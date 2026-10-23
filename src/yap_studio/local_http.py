"""HTTP servers for Yap Studio's local pages."""

from __future__ import annotations

import socketserver
from http.server import ThreadingHTTPServer


class LocalHTTPServer(ThreadingHTTPServer):
    """Bind without the reverse DNS lookup that HTTPServer.server_bind makes.

    socket.getfqdn on the bound address can wait on a multicast DNS lookup for tens of
    seconds on some Macs, so a page would start long after its caller gave up.
    """

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = port
