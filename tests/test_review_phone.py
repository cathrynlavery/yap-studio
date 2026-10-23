"""`review --phone`: one link, over Tailscale when connected, otherwise the local Wi-Fi."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from yap_studio.cli import main
from yap_studio.review import phone
from yap_studio.review import server as server_module
from yap_studio.review.media import ReviewMedia
from yap_studio.review.server import ReviewServer
from yap_studio.review.store import ReviewStore


@pytest.fixture
def video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """One synthetic video, a private review store and a recording stand-in socket."""
    path = tmp_path / "sample.mp4"
    path.write_bytes(b"synthetic video bytes")
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(
        "yap_studio.review.media.probe_media",
        lambda item: ReviewMedia(Path(item), "sample.mp4", 2.0, None, False, "h264", 64, 36),
    )
    monkeypatch.setattr("yap_studio.cli.webbrowser.open", lambda url: None)
    return path


@pytest.fixture
def binds(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    addresses: list[tuple[str, int]] = []

    def bind(address: tuple[str, int], _handler: object) -> SimpleNamespace:
        addresses.append(address)
        return SimpleNamespace(server_port=address[1] or 12345, server_close=lambda: None,
                               serve_forever=lambda: None)

    monkeypatch.setattr(server_module, "_QuietThreadingHTTPServer", bind)
    return addresses


@pytest.mark.parametrize(("value", "lan"), [
    ("192.168.1.20", True), ("10.0.0.7", True), ("172.20.4.1", True),
    ("127.0.0.1", False), ("0.0.0.0", False), ("8.8.8.8", False), ("100.101.102.103", False),
    ("169.254.3.4", False), ("fd00::1", False), ("box.local", False), ("", False),
])
def test_only_private_network_addresses_count(value: str, lan: bool) -> None:
    assert phone.is_lan_address(value) is lan


def test_lan_address_prefers_wifi_over_vpn_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phone.sys, "platform", "darwin")
    calls: list[tuple[str, ...]] = []

    def run(args: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(tuple(args))
        assert kwargs["timeout"] == 5
        if args[0] == "networksetup":
            return SimpleNamespace(returncode=0, stdout=(
                "Hardware Port: Ethernet\nDevice: en0\n\n"
                "Hardware Port: VPN\nDevice: utun3\n\n"
                "Hardware Port: Wi-Fi\nDevice: en1\n\n"
                "Hardware Port: USB Ethernet\nDevice: en5\n"))
        return SimpleNamespace(returncode=0, stdout={
            "en1": "192.168.1.20\n", "en0": "10.0.0.7\n", "en5": "172.20.4.1\n",
        }[args[-1]])

    monkeypatch.setattr(phone.subprocess, "run", run)
    monkeypatch.setattr(phone.socket, "socket", lambda *_: pytest.fail("UDP probe not needed"))
    assert phone.lan_address() == "192.168.1.20"
    assert calls == [("networksetup", "-listallhardwareports"),
                     ("ipconfig", "getifaddr", "en1")]


def test_lan_address_tries_other_hardware_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phone.sys, "platform", "darwin")
    calls: list[tuple[str, ...]] = []

    def run(args: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(tuple(args))
        if args[0] == "networksetup":
            return SimpleNamespace(returncode=0, stdout=(
                "Hardware Port: Wi-Fi\nDevice: en1\n\n"
                "Hardware Port: Ethernet\nDevice: en0\n\n"
                "Hardware Port: Tunnel\nDevice: utun3\n"))
        return SimpleNamespace(returncode=1 if args[-1] == "en1" else 0,
                               stdout="172.20.4.1\n")

    monkeypatch.setattr(phone.subprocess, "run", run)
    monkeypatch.setattr(phone.socket, "socket", lambda *_: pytest.fail("UDP probe not needed"))
    assert phone.lan_address() == "172.20.4.1"
    assert calls == [("networksetup", "-listallhardwareports"),
                     ("ipconfig", "getifaddr", "en1"),
                     ("ipconfig", "getifaddr", "en0")]


def test_lan_address_falls_back_when_hardware_ports_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(phone.sys, "platform", "darwin")

    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("networksetup")

    class Probe:
        def __enter__(self) -> Probe:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def connect(self, address: tuple[str, int]) -> None:
            assert address == ("10.254.254.254", 9)

        def getsockname(self) -> tuple[str, int]:
            return "192.168.1.21", 1234

    monkeypatch.setattr(phone.subprocess, "run", unavailable)
    monkeypatch.setattr(phone.socket, "socket", lambda *_: Probe())
    assert phone.lan_address() == "192.168.1.21"


@pytest.mark.parametrize("host", ["0.0.0.0", "8.8.8.8", "100.101.102.103", "box.local"])
def test_server_refuses_public_and_wildcard_binds(
    video: Path, binds: list, tmp_path: Path, host: str,
) -> None:
    store = ReviewStore(tmp_path / "data")
    media = ReviewMedia(video, "sample.mp4", 2.0, None, False, "h264", 64, 36)
    with pytest.raises(ValueError, match="127.0.0.1 or a private network address"):
        ReviewServer(media, "a" * 64, None, store, host=host)
    assert binds == []


def test_server_on_wifi_address_links_and_accepts_that_host(
    video: Path, binds: list, tmp_path: Path,
) -> None:
    store = ReviewStore(tmp_path / "data")
    media = ReviewMedia(video, "sample.mp4", 2.0, None, False, "h264", 64, 36)
    server = ReviewServer(media, "a" * 64, None, store, host="192.168.1.20")
    assert binds == [("192.168.1.20", 0)]
    assert server.url == f"http://192.168.1.20:12345/?t={server.token}"
    assert server._accepts_host("192.168.1.20:12345")
    assert not server._accepts_host("192.168.1.21:12345")
    assert not server._accepts_host("192.168.1.20:12346")


def test_phone_without_tailscale_prints_one_wifi_link(
    video: Path, binds: list, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(phone, "tailscale_name", lambda: None)
    monkeypatch.setattr(phone, "lan_address", lambda: "192.168.1.20")
    assert main(["review", str(video), "--phone", "--no-open"]) == 0
    assert binds == [("192.168.1.20", 0)]
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("Open on your phone: Not encrypted (same Wi-Fi): "
                               "http://192.168.1.20:12345/?t=")
    assert lines[1:] == ["Press Ctrl-C to stop."]


def test_phone_with_tailscale_uses_tailnet_link(
    video: Path, binds: list, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ports: list[int] = []
    monkeypatch.setattr(phone, "tailscale_name", lambda: "box.tailnet.example.invalid")

    def origin(port: int) -> tuple[str, None]:
        ports.append(port)
        return f"https://box.tailnet.example.invalid:{port}", None

    monkeypatch.setattr(phone, "tailscale_origin", origin)
    monkeypatch.setattr(phone, "lan_address", lambda: pytest.fail("Wi-Fi not needed"))
    assert main(["review", str(video), "--phone", "--no-open", "--json"]) == 0
    assert binds == [("127.0.0.1", 0)] and ports == [12345]
    document = json.loads(capsys.readouterr().out)
    url = urlsplit(document["phone"]["url"])
    assert document["phone"]["via"] == "tailscale"
    assert f"{url.scheme}://{url.netloc}" == "https://box.tailnet.example.invalid:12345"
    assert url.query.startswith("t=")
    assert document["videos"][0]["url"] == document["phone"]["url"]
    local = urlsplit(document["videos"][0]["local_url"])
    assert (local.scheme, local.netloc) == ("http", "127.0.0.1:12345")
    assert local.query == url.query


def test_tailscale_server_accepts_the_tailnet_host(
    video: Path, binds: list, monkeypatch: pytest.MonkeyPatch,
) -> None:
    servers: list[ReviewServer] = []
    monkeypatch.setattr(phone, "tailscale_name", lambda: "box.tailnet.example.invalid")
    monkeypatch.setattr(phone, "tailscale_origin",
                        lambda port: (f"https://box.tailnet.example.invalid:{port}", None))
    monkeypatch.setattr(ReviewServer, "serve_forever", lambda self: servers.append(self))
    assert main(["review", str(video), "--phone", "--no-open"]) == 0
    assert servers[0]._accepts_host("box.tailnet.example.invalid:12345")
    assert servers[0]._accepts_host("BOX.tailnet.example.invalid")
    assert not servers[0]._accepts_host("other.tailnet.example.invalid:12345")


def test_failed_tailscale_link_falls_back_to_wifi(
    video: Path, binds: list, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def refuse(port: int) -> str:
        raise OSError("HTTPS is not enabled for this tailnet")

    monkeypatch.setattr(phone, "tailscale_name", lambda: "box.tailnet.example.invalid")
    monkeypatch.setattr(phone, "tailscale_origin", refuse)
    monkeypatch.setattr(phone, "lan_address", lambda: "10.0.0.7")
    assert main(["review", str(video), "--phone", "--no-open"]) == 0
    assert binds == [("127.0.0.1", 0), ("10.0.0.7", 0)]
    captured = capsys.readouterr()
    assert "Tailscale link failed (HTTPS is not enabled for this tailnet)" in captured.err
    assert "http://10.0.0.7:12345/?t=" in captured.out


def test_no_network_exits_two_with_a_fix(
    video: Path, binds: list, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(phone, "tailscale_name", lambda: None)
    monkeypatch.setattr(phone, "lan_address", lambda: None)
    assert main(["review", str(video), "--phone", "--no-open"]) == 2
    error = capsys.readouterr().err
    assert "No network your phone can reach" in error
    assert "fix: Join the same Wi-Fi as your phone, or set up Tailscale" in error
    assert binds == []


def test_phone_and_public_base_are_exclusive(
    video: Path, binds: list, capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["review", str(video), "--phone", "--public-base", "https://box.example.net",
                 "--no-open"]) == 2
    assert "use --phone or --public-base, not both" in capsys.readouterr().err
    assert binds == []


class FakeTailscale:
    """Answer status and serve commands like the Tailscale CLI, recording serve changes."""

    def __init__(self, serve: dict, fail_off: bool = False) -> None:
        self.serve = serve
        self.fail_off = fail_off
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, binary: str, *args: str) -> str:
        self.calls.append(args)
        if args == ("status", "--json"):
            return json.dumps({"BackendState": "Running",
                               "Self": {"DNSName": "box.tailnet.example.invalid."}})
        if args == ("serve", "status", "--json"):
            return json.dumps(self.serve)
        if self.fail_off and args[:1] == ("serve",) and args[-1:] == ("off",):
            raise OSError("serve is unavailable")
        return ""


@pytest.mark.parametrize("interrupt", [False, True])
def test_new_tailscale_mapping_is_removed_after_review_stops(
    video: Path, binds: list, monkeypatch: pytest.MonkeyPatch, interrupt: bool,
) -> None:
    fake = FakeTailscale({})
    monkeypatch.setattr(phone, "_tailscale_binary", lambda: "/usr/bin/tailscale")
    monkeypatch.setattr(phone, "_tailscale", fake)

    def serve(_server: ReviewServer) -> None:
        assert ("serve", "--https=12345", "off") not in fake.calls
        if interrupt:
            raise KeyboardInterrupt

    monkeypatch.setattr(ReviewServer, "serve_forever", serve)
    assert main(["review", str(video), "--phone", "--no-open"]) == 0
    assert binds == [("127.0.0.1", 0)]
    assert ("serve", "--bg", "--https=12345", "http://127.0.0.1:12345") in fake.calls
    assert fake.calls[-1] == ("serve", "--https=12345", "off")


def test_sigterm_stops_review_and_removes_tailscale_mapping(
    video: Path, binds: list, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeTailscale({})
    monkeypatch.setattr(phone, "_tailscale_binary", lambda: "/usr/bin/tailscale")
    monkeypatch.setattr(phone, "_tailscale", fake)
    previous = signal.getsignal(signal.SIGTERM)
    received: list[int] = []

    def prior_handler(signum: int, _frame: object) -> None:
        received.append(signum)

    def serve(_server: ReviewServer) -> None:
        os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(ReviewServer, "serve_forever", serve)
    signal.signal(signal.SIGTERM, prior_handler)
    try:
        assert main(["review", str(video), "--phone", "--no-open"]) == 0
        assert binds == [("127.0.0.1", 0)]
        assert fake.calls[-1] == ("serve", "--https=12345", "off")
        assert signal.getsignal(signal.SIGTERM) is prior_handler
        assert received == []
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_sigterm_during_mapping_setup_removes_it_and_restores_handler(
    video: Path, binds: list, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeTailscale({})
    monkeypatch.setattr(phone, "_tailscale_binary", lambda: "/usr/bin/tailscale")

    def tailscale(binary: str, *args: str) -> str:
        result = fake(binary, *args)
        if args[:2] == ("serve", "--bg"):
            os.kill(os.getpid(), signal.SIGTERM)
        return result

    monkeypatch.setattr(phone, "_tailscale", tailscale)
    monkeypatch.setattr(ReviewServer, "serve_forever",
                        lambda self: pytest.fail("review should stop before serving"))
    previous = signal.getsignal(signal.SIGTERM)
    received: list[int] = []

    def prior_handler(signum: int, _frame: object) -> None:
        received.append(signum)

    signal.signal(signal.SIGTERM, prior_handler)
    try:
        assert main(["review", str(video), "--phone", "--no-open"]) == 0
        assert fake.calls[-1] == ("serve", "--https=12345", "off")
        assert signal.getsignal(signal.SIGTERM) is prior_handler
        assert received == []
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_reused_tailscale_mapping_is_left_alone(
    video: Path, binds: list, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeTailscale({"TCP": {"8470": {"HTTPS": True}}, "Web": {
        "box.tailnet.example.invalid:8470": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:12345"}}},
    }})
    monkeypatch.setattr(phone, "_tailscale_binary", lambda: "/usr/bin/tailscale")
    monkeypatch.setattr(phone, "_tailscale", fake)
    assert main(["review", str(video), "--phone", "--no-open"]) == 0
    assert binds == [("127.0.0.1", 0)]
    assert not any(call[-1:] == ("off",) for call in fake.calls)


def test_failed_tailscale_removal_warns_and_review_succeeds(
    video: Path, binds: list, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeTailscale({}, fail_off=True)
    monkeypatch.setattr(phone, "_tailscale_binary", lambda: "/usr/bin/tailscale")
    monkeypatch.setattr(phone, "_tailscale", fake)
    assert main(["review", str(video), "--phone", "--no-open"]) == 0
    assert binds == [("127.0.0.1", 0)]
    assert fake.calls[-1] == ("serve", "--https=12345", "off")
    assert capsys.readouterr().err.splitlines() == [
        "warning: could not remove Tailscale mapping; run tailscale serve --https=12345 off"
    ]


def test_tailscale_origin_reuses_an_existing_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeTailscale({"TCP": {"8470": {"HTTPS": True}}, "Web": {
        "box.tailnet.example.invalid:8470": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:50123"}}},
    }})
    monkeypatch.setattr(phone, "_tailscale_binary", lambda: "/usr/bin/tailscale")
    monkeypatch.setattr(phone, "_tailscale", fake)
    origin, cleanup = phone.tailscale_origin(50123)
    assert origin == "https://box.tailnet.example.invalid:8470"
    assert cleanup is None
    assert not any(call[:2] == ("serve", "--bg") for call in fake.calls)


def test_tailscale_origin_does_not_reuse_funnel_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeTailscale({
        "TCP": {"50123": {"HTTPS": True}},
        "Web": {"box.tailnet.example.invalid:50123": {
            "Handlers": {"/": {"Proxy": "http://127.0.0.1:50123"}},
        }},
        "AllowFunnel": {"box.tailnet.example.invalid:50123": True},
    })
    monkeypatch.setattr(phone, "_tailscale_binary", lambda: "/usr/bin/tailscale")
    monkeypatch.setattr(phone, "_tailscale", fake)
    origin, cleanup = phone.tailscale_origin(50123)
    assert origin == "https://box.tailnet.example.invalid:18000"
    assert cleanup is not None
    assert ("serve", "--bg", "--https=18000", "http://127.0.0.1:50123") in fake.calls
    cleanup()
    assert fake.calls[-1] == ("serve", "--https=18000", "off")


@pytest.mark.parametrize(("serve", "external"), [
    ({}, 50123),
    ({"TCP": {"50123": {"HTTPS": True}}, "Web": {
        "box.tailnet.example.invalid:50123": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:3000"}}},
    }}, 18000),
])
def test_tailscale_origin_starts_a_tailnet_only_mapping(
    monkeypatch: pytest.MonkeyPatch, serve: dict, external: int,
) -> None:
    fake = FakeTailscale(serve)
    monkeypatch.setattr(phone, "_tailscale_binary", lambda: "/usr/bin/tailscale")
    monkeypatch.setattr(phone, "_tailscale", fake)
    origin, cleanup = phone.tailscale_origin(50123)
    assert origin == f"https://box.tailnet.example.invalid:{external}"
    assert cleanup is not None
    assert ("serve", "--bg", f"--https={external}", "http://127.0.0.1:50123") in fake.calls
    assert not any("funnel" in call for call in fake.calls)
    cleanup()
    assert fake.calls[-1] == ("serve", f"--https={external}", "off")


def test_tailscale_name_needs_a_running_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phone, "_tailscale_binary", lambda: "/usr/bin/tailscale")
    monkeypatch.setattr(phone, "_tailscale", lambda binary, *args: json.dumps(
        {"BackendState": "Stopped", "Self": {"DNSName": "box.tailnet.example.invalid."}}))
    assert phone.tailscale_name() is None
    monkeypatch.setattr(phone, "_tailscale_binary", lambda: None)
    assert phone.tailscale_name() is None


@pytest.mark.parametrize("failure", ["exit", "oserror", "timeout", "json", "shape"])
@pytest.mark.parametrize("recovers", [True, False])
def test_tailscale_name_retries_failed_status_with_a_bound(
    monkeypatch: pytest.MonkeyPatch, failure: str, recovers: bool,
) -> None:
    monkeypatch.setattr(phone, "_tailscale_binary", lambda: "/usr/bin/tailscale")
    pauses: list[float] = []
    monkeypatch.setattr("time.sleep", pauses.append)
    timeouts: list[float] = []

    def run(args: list[str], **kwargs: object) -> SimpleNamespace:
        assert args == ["/usr/bin/tailscale", "status", "--json"]
        timeouts.append(kwargs["timeout"])
        if recovers and len(timeouts) == 3:
            return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
                "BackendState": "Running", "Self": {"DNSName": "box.tailnet.example.invalid."},
            }))
        if failure == "oserror":
            raise OSError("starting")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        return SimpleNamespace(returncode=1 if failure == "exit" else 0,
                               stderr="starting", stdout="[]" if failure == "shape" else "bad")

    monkeypatch.setattr(phone.subprocess, "run", run)
    assert phone.tailscale_name() == ("box.tailnet.example.invalid" if recovers else None)
    assert len(timeouts) == 3
    assert pauses == [1, 1]
    assert sum(timeouts) + sum(pauses) <= 17


@pytest.mark.parametrize("state", ["Stopped", "NeedsLogin", "Starting"])
def test_tailscale_name_does_not_retry_a_not_running_status(
    monkeypatch: pytest.MonkeyPatch, state: str,
) -> None:
    monkeypatch.setattr(phone, "_tailscale_binary", lambda: "/usr/bin/tailscale")
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(phone, "_tailscale", lambda binary, *args, **kwargs: (
        calls.append(args) or json.dumps({"BackendState": state, "Self": {}})))
    monkeypatch.setattr("time.sleep", lambda seconds: pytest.fail("must not retry"))
    assert phone.tailscale_name() is None
    assert calls == [("status", "--json")]
