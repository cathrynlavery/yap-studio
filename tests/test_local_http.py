"""Local page servers bind without a reverse DNS lookup."""

from __future__ import annotations

import re
import socket
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

from yap_studio.local_http import LocalHTTPServer

SRC = Path(__file__).parents[1] / "src" / "yap_studio"


def test_bind_skips_reverse_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    def lookup(*_args):
        raise AssertionError("server_bind must not resolve the host name")

    monkeypatch.setattr(socket, "getfqdn", lookup)
    server = LocalHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    try:
        assert server.server_name == "127.0.0.1"
        assert server.server_port == server.socket.getsockname()[1] > 0
    finally:
        server.server_close()


def test_page_servers_use_local_server() -> None:
    from yap_studio.gallery.server import _GalleryHTTPServer
    from yap_studio.review.server import _QuietThreadingHTTPServer

    assert issubclass(_GalleryHTTPServer, LocalHTTPServer)
    assert issubclass(_QuietThreadingHTTPServer, LocalHTTPServer)
    stock = re.compile(r"\b(?:Threading)?HTTPServer\b")
    users = [path.relative_to(SRC).as_posix() for path in SRC.rglob("*.py")
             if path.name != "local_http.py" and stock.search(path.read_text(encoding="utf-8"))]
    assert users == []
