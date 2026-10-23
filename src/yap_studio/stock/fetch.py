"""Bounded HTTPS downloads from Pexels hosts, including every redirect."""

from __future__ import annotations

import os
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

from yap_studio import __version__, paths
from yap_studio.errors import ValidationError
from yap_studio.stock.provider import StockUnavailable

CLIP_MAX_BYTES = 600 * 1024 * 1024
CLIP_TIMEOUT_S = 900
PREVIEW_MAX_BYTES = 5 * 1024 * 1024
PREVIEW_TIMEOUT_S = 30
HOSTS = frozenset({"videos.pexels.com", "images.pexels.com", "www.pexels.com"})


def allowed_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlsplit(url)
        return (
            parsed.scheme == "https"
            and parsed.hostname in HOSTS
            and parsed.port in {None, 443}
            and parsed.username is None
            and parsed.password is None
            and not any(ord(c) < 33 for c in url)
        )
    except (ValueError, TypeError):
        return False


def check_url(url: str) -> None:
    if not allowed_url(url):
        raise ValidationError("url", "expected HTTPS on an allowed Pexels download host")


class CheckedRedirect(urllib.request.HTTPRedirectHandler):
    """Check the destination and a per-request redirect budget before following."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        count = getattr(req, "stock_redirects", 0) + 1
        if count > 3:
            raise ValidationError("url.redirect", "at most 3 redirects are allowed")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.stock_redirects = count
        return redirected


def fetch(
    url: str,
    target: Path,
    *,
    max_bytes: int,
    timeout_s: float,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Stream a bounded response, replacing the destination only after success."""
    check_url(url)
    target = Path(target)
    part = target.with_name(target.name + ".part")
    with paths.operation():
        paths._guard_write(target)
        paths._guard_write(part)
        target.parent.mkdir(parents=True, exist_ok=True)
        created = False
        try:
            deadline = clock() + timeout_s
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": f"yap-studio/{__version__}",
                },
            )
            opener = urllib.request.build_opener(CheckedRedirect())
            with opener.open(request, timeout=timeout_s) as response:
                length = response.headers.get("Content-Length")
                try:
                    length = int(length)
                except (ValueError, TypeError):
                    length = None
                if length is not None and length > max_bytes:
                    raise ValidationError("max_bytes", "download exceeds the size limit")
                with part.open("xb") as stream:
                    created = True
                    count = 0
                    while chunk := response.read(64 * 1024):
                        if clock() > deadline:
                            raise TimeoutError("stock download timed out")
                        count += len(chunk)
                        if count > max_bytes:
                            raise ValidationError("max_bytes", "download exceeds the size limit")
                        stream.write(chunk)
            paths._guard_write(target)
            os.replace(part, target)
        except urllib.error.HTTPError as exc:
            raise StockUnavailable("download_failed", http_status=exc.code) from exc
        except (
            TimeoutError,
            urllib.error.URLError,
            ConnectionError,
            socket.gaierror,
            ssl.SSLError,
        ) as exc:
            raise StockUnavailable("offline") from exc
        finally:
            if created:
                part.unlink(missing_ok=True)
