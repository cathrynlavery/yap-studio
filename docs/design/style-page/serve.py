#!/usr/bin/env python3
"""Serve the style page and pull colors from a website.

    python3 serve.py            # http://127.0.0.1:8792, or the next free port
    python3 serve.py --port 9000

GET /api/site-colors?url=example.com gathers a site's colors from two places:

1. The optional context.dev brand lookup, when context-dev-pp-cli is installed and
   signed in. It returns named logo and site colors. Each lookup costs credits,
   so results are cached while the server runs.
   Without the CLI or a successful brand lookup, only HTML and stylesheets are used.
2. The page's own HTML and up to six stylesheets. Colors on buttons and links
   count extra, the body background becomes the card color, and browser default
   link colors are ignored.

It merges near-duplicates and suggests which color to use for captions, the
highlight, cards and card text. A browser cannot do this on its own because
other sites block cross-origin reads.

This is a prototype helper. It only listens on 127.0.0.1, only connects to
public http and https addresses (checked on every connection, redirects
included), and caps time and size for every request.
"""

from __future__ import annotations

import argparse
import colorsys
import http.client
import ipaddress
import json
import re
import shutil
import socket
import ssl
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIMEOUT_S = 8
# The whole lookup stops after this many seconds; the page waits a little longer.
DEADLINE_S = 25
BRAND_TIMEOUT_S = 14
# A lookup that outlives its deadline keeps its thread until its sockets give
# up, so only this many may run at once.
MAX_LOOKUPS = 4
LOOKUP_SLOTS = threading.BoundedSemaphore(MAX_LOOKUPS)
MAX_HTML = 1_500_000
MAX_CSS = 800_000
MAX_SHEETS = 6
THEME_METAS = ("theme-color", "msapplication-tilecolor")
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126 Safari/537.36"
)

HEX = re.compile(r"#([0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3,4})\b")
RGB = re.compile(r"rgba?\(\s*(\d{1,3})[\s,]+(\d{1,3})[\s,]+(\d{1,3})(?:[\s,/]+([\d.]+%?))?\s*\)")
RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")
INTERACTIVE = re.compile(r"button|\bbtn|\ba\b|:hover|cta|submit|\blink", re.I)
CANVAS = re.compile(r"(^|[\s,])(html|body|:root|main)\b", re.I)
BG_DECL = re.compile(r"background(?:-color)?\s*:\s*([^;]+)", re.I)
VAR_DEF = re.compile(r"(--[\w-]+)\s*:\s*([^;}{]+)")
VAR_REF = re.compile(r"var\(\s*(--[\w-]+)\s*(?:,\s*([^()]*))?\)")
DOMAIN = re.compile(r"^(?=.{1,253}$)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")
# Browser default link and visited colors say nothing about a brand.
IGNORED = {"#0000EE", "#551A8B", "#0000FF", "#800080"}
BRAND_CACHE: dict[str, tuple[list[tuple[str, float, str]], str]] = {}
# One lookup per host at a time, so two requests never pay for the same host.
BRAND_LOCKS: dict[str, threading.Lock] = {}
BRAND_LOCKS_GUARD = threading.Lock()
BRAND_VAR = re.compile(
    r"--[\w-]*(primary|brand|accent|main|highlight|cta|button|link|theme)[\w-]*\s*:\s*([^;}{]+)",
    re.I,
)


class FetchError(Exception):
    pass


def _public_ip(host: str, port: int) -> str:
    """Resolve once, refuse any non-public answer, and return the address to use."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise FetchError("That address could not be found.") from exc
    addresses = [info[4][0] for info in infos]
    if not addresses or not all(ipaddress.ip_address(a).is_global for a in addresses):
        raise FetchError("That address is not a public website.")
    return addresses[0]


# Each lookup runs in its own server thread and records its absolute deadline
# here, so every connection, redirect hop and read can check the same clock.
_LOOKUP = threading.local()


def _remaining() -> float:
    return getattr(_LOOKUP, "deadline", float("inf")) - time.monotonic()


def _connect_timeout(timeout: object) -> float:
    left = _remaining()
    if left <= 0.5:
        raise FetchError("That site took too long.")
    base = timeout if isinstance(timeout, (int, float)) else TIMEOUT_S
    return min(base, left)


# Connections go to the exact address that passed the check, so a second DNS
# answer (rebinding) or a redirect to a private host never gets a connection.
class _PinnedHTTPConnection(http.client.HTTPConnection):
    def connect(self):
        addr = (_public_ip(self.host, self.port), self.port)
        self.sock = socket.create_connection(addr, _connect_timeout(self.timeout))


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def connect(self):
        addr = (_public_ip(self.host, self.port), self.port)
        sock = socket.create_connection(addr, _connect_timeout(self.timeout))
        self.sock = _TLS.wrap_socket(sock, server_hostname=self.host)


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Follow at most a few redirects, and only to http or https addresses."""

    max_redirections = 4

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme not in ("http", "https"):
            fp.close()
            raise FetchError("The site redirected to an address we can't read.")
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        # urllib reads the whole redirect body before following it; closing the
        # response here makes that read return nothing, so a slow body can't stall us.
        fp.close()
        return new


class _PinnedHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_PinnedHTTPConnection, req)


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_PinnedHTTPSConnection, req)


_TLS = ssl.create_default_context()
# Only these handlers: no proxies, no file or ftp, redirects through the pinned ones.
_OPENER = urllib.request.OpenerDirector()
for _handler in (
    _PinnedHTTPHandler(),
    _PinnedHTTPSHandler(),
    _SafeRedirect(),
    urllib.request.HTTPDefaultErrorHandler(),
    urllib.request.HTTPErrorProcessor(),
):
    _OPENER.add_handler(_handler)


def _normalize(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        raise FetchError("Enter a website address.")
    if "://" not in raw:
        raw = "https://" + raw
    parts = urllib.parse.urlsplit(raw)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FetchError("Use a website address like example.com.")
    return raw


def _get(url: str, limit: int, timeout: float = TIMEOUT_S) -> tuple[str, str]:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FetchError("Unsupported address.")
    headers = {"User-Agent": UA, "Accept": "text/html,text/css,*/*;q=0.8"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            final = resp.geturl()
            charset = resp.headers.get_content_charset() or "utf-8"
            # read1 does one socket read at a time, so a slow trickle can't
            # hold the lookup past its deadline.
            chunks: list[bytes] = []
            size = 0
            while size < limit:
                if _remaining() <= 0:
                    raise FetchError("That site took too long.")
                chunk = resp.read1(min(65536, limit - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
            data = b"".join(chunks)
    except urllib.error.HTTPError as exc:
        raise FetchError(f"The site answered with an error ({exc.code}).") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FetchError("The site did not respond.") from exc
    try:
        return data.decode(charset, errors="replace"), final
    except LookupError:
        # The site named a charset Python doesn't know; read it as UTF-8.
        return data.decode("utf-8", errors="replace"), final


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sheets: list[str] = []
        self.styles: list[str] = []
        self.theme: list[str] = []
        self.title = ""
        self._in_style = False
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "link" and "stylesheet" in a.get("rel", "").lower() and a.get("href"):
            self.sheets.append(a["href"])
        elif tag == "meta" and a.get("name", "").lower() in THEME_METAS:
            self.theme.append(a.get("content", ""))
        elif tag == "style":
            self._in_style = True
        elif tag == "title":
            self._in_title = True
        if a.get("style"):
            self.styles.append(a["style"])

    def handle_endtag(self, tag):
        if tag == "style":
            self._in_style = False
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_style:
            self.styles.append(data)
        elif self._in_title and len(self.title) < 120:
            self.title += data


def _to_hex(text: str) -> str | None:
    m = HEX.fullmatch(text.strip())
    if m:
        h = m.group(1)
        if len(h) in (3, 4):
            if len(h) == 4 and int(h[3], 16) < 8:
                return None
            h = "".join(c * 2 for c in h[:3])
        elif len(h) == 8:
            if int(h[6:8], 16) < 128:
                return None
            h = h[:6]
        return "#" + h.upper()
    m = RGB.fullmatch(text.strip())
    if m:
        r, g, b, alpha = m.groups()
        if alpha:
            a = float(alpha.rstrip("%")) / (100 if alpha.endswith("%") else 1)
            if a < 0.5:
                return None
        vals = [int(r), int(g), int(b)]
        if any(v > 255 for v in vals):
            return None
        return "#" + "".join(f"{v:02X}" for v in vals)
    return None


def _add_colors(text: str, w: float, into: Counter) -> None:
    for m in list(HEX.finditer(text)) + list(RGB.finditer(text)):
        hx = _to_hex(m.group(0))
        if hx and hx not in IGNORED:
            into[hx] += w


CanvasDecl = tuple[str, dict[str, str]]


def _var_level(selector: str) -> int:
    """How close a rule's variable definitions sit to the page background."""
    if re.search(r"\bbody\b", selector, re.I):
        return 2
    return 1 if CANVAS.search(selector) else 0


def _colors_in(
    css: str,
    weight: float,
    counts: Counter,
    canvas_decls: list[CanvasDecl],
    css_vars: dict[str, tuple[int, str]],
) -> None:
    _add_colors(css, weight, counts)
    for m in BRAND_VAR.finditer(css):
        _add_colors(m.group(2), weight * 6, counts)
    rules = RULE.findall(css) or [("", css)]
    for selector, body in rules:
        level = _var_level(selector)
        local = {name: value.strip() for name, value in VAR_DEF.findall(body)}
        # CSS variable names are case-sensitive. A definition nearer the body
        # replaces one on :root; at the same level the first one stays, since
        # later ones are often media-query overrides this parser can't see.
        for name, value in local.items():
            if name not in css_vars or level > css_vars[name][0]:
                css_vars[name] = (level, value)
        if INTERACTIVE.search(selector):
            _add_colors(body, weight * 3, counts)
        if CANVAS.search(selector):
            canvas_decls.extend((bg, local) for bg in BG_DECL.findall(body))


def _resolve_vars(text: str, local: dict[str, str], css_vars: dict[str, tuple[int, str]]) -> str:
    """Replace var(--x) with its definition (or its fallback), a few levels deep.

    A definition in the same rule as the background wins, then the page-wide one.
    """

    def value(m: re.Match) -> str:
        name = m.group(1)
        if name in local:
            return local[name]
        if name in css_vars:
            return css_vars[name][1]
        return m.group(2) or ""

    for _ in range(4):
        resolved = VAR_REF.sub(value, text)
        if resolved == text:
            break
        text = resolved
    return text


def _rgb(hx: str) -> tuple[int, int, int]:
    n = int(hx[1:], 16)
    return (n >> 16) & 255, (n >> 8) & 255, n & 255


def _dist(a: str, b: str) -> float:
    return sum((x - y) ** 2 for x, y in zip(_rgb(a), _rgb(b), strict=True)) ** 0.5


def _hsv(hx: str) -> tuple[float, float, float]:
    r, g, b = (v / 255 for v in _rgb(hx))
    return colorsys.rgb_to_hsv(r, g, b)


def _luminance(hx: str) -> float:
    def ch(v: int) -> float:
        c = v / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = _rgb(hx)
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _cluster(counts: Counter) -> list[tuple[str, float]]:
    groups: list[list] = []
    for hx, w in counts.most_common():
        for g in groups:
            if _dist(g[0], hx) < 28:
                g[1] += w
                break
        else:
            groups.append([hx, w])
    return [(g[0], g[1]) for g in sorted(groups, key=lambda g: -g[1])]


def _kind(hx: str) -> str:
    _, s, v = _hsv(hx)
    if s < 0.18 or v < 0.12:
        return "light" if v > 0.85 else "dark" if v < 0.3 else "neutral"
    return "color"


def suggest(colors: list[tuple[str, float]], canvas: str | None) -> dict:
    # The highlight should be a real color: saturated and not too dark. Among
    # those, frequent and vivid wins. None means the site is black and white.
    vivid = [
        (c, w * _hsv(c)[1] * _hsv(c)[2])
        for c, w in colors
        if _hsv(c)[1] >= 0.35 and _hsv(c)[2] >= 0.35
    ]
    accent = max(vivid, key=lambda cw: cw[1])[0] if vivid else None
    lights = [c for c, _ in colors if _kind(c) == "light"]
    if canvas:
        paper = canvas
    else:
        paper = next((c for c in lights if c != "#FFFFFF"), lights[0] if lights else "#FFFFFF")
    # Card text: the most used color that reads on the card.
    ink = next((c for c, _ in colors if _contrast(c, paper) >= 4.5 and _kind(c) != "color"), None)
    if ink is None:
        ink = "#111111" if _luminance(paper) > 0.4 else "#FFFFFF"
    caption = next((c for c in lights if _luminance(c) > 0.85), "#FFFFFF")
    return {"caption": caption, "accent": accent, "paper": paper, "ink": ink}


def brand_lookup(host: str) -> tuple[list[tuple[str, float, str]], str]:
    """Optional context.dev colors; return none so HTML and CSS work alone."""
    host = host.lower().removeprefix("www.")
    cli = shutil.which("context-dev-pp-cli")
    if not cli or not DOMAIN.match(host):
        return [], ""
    with BRAND_LOCKS_GUARD:
        lock = BRAND_LOCKS.setdefault(host, threading.Lock())
    with lock:
        if host in BRAND_CACHE:
            return BRAND_CACHE[host]
        cmd = [cli, "brand", "list-retrievesimplified", "--domain", host]
        cmd += ["--timeout-ms", str((BRAND_TIMEOUT_S - 2) * 1000), "--json", "--no-input"]
        cmd += ["--no-color"]
        try:
            done = subprocess.run(
                cmd, capture_output=True, text=True, timeout=BRAND_TIMEOUT_S, check=False
            )
            brand = json.loads(done.stdout)["results"]["brand"]
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
            return [], ""
        found: list[tuple[str, float, str]] = []
        for c in brand.get("colors") or []:
            hx = _to_hex(c.get("hex", ""))
            if hx:
                found.append((hx, 60 if c.get("source") == "logo" else 45, c.get("name", "")))
        for backdrop in brand.get("backdrops") or []:
            for c in backdrop.get("colors") or []:
                hx = _to_hex(c.get("hex", ""))
                if hx:
                    found.append((hx, 4, c.get("name", "")))
        result = (found, brand.get("title") or "")
        BRAND_CACHE[host] = result
        return result


def lookup_with_cutoff(raw_url: str) -> dict:
    """Answer within the deadline no matter where the lookup is stuck.

    The lookup runs in a worker thread. Socket timeouts can't cover DNS or a
    wait on another request's brand lookup, so the request itself stops waiting
    at the deadline and ignores anything the worker finishes later.
    """
    if not LOOKUP_SLOTS.acquire(blocking=False):
        raise FetchError("Too many color lookups are running. Try again in a moment.")
    box: dict[str, object] = {}
    done = threading.Event()

    def work() -> None:
        try:
            box["result"] = site_colors(raw_url)
        except FetchError as exc:
            box["error"] = exc
        except Exception:
            traceback.print_exc()
            box["error"] = FetchError("That site could not be read.")
        finally:
            LOOKUP_SLOTS.release()
            done.set()

    threading.Thread(target=work, daemon=True).start()
    if not done.wait(DEADLINE_S + 1):
        raise FetchError("That site took too long.")
    if "error" in box:
        raise box["error"]
    return box["result"]


def site_colors(raw_url: str) -> dict:
    url = _normalize(raw_url)
    deadline = time.monotonic() + DEADLINE_S
    _LOOKUP.deadline = deadline
    try:
        return _site_colors(url, deadline)
    finally:
        del _LOOKUP.deadline


def _site_colors(url: str, deadline: float) -> dict:
    host = urllib.parse.urlsplit(url).hostname or ""
    counts: Counter = Counter()
    canvas_decls: list[CanvasDecl] = []
    css_vars: dict[str, tuple[int, str]] = {}
    names: dict[str, str] = {}
    title = ""
    sources = []
    brand, brand_title = brand_lookup(host)
    if brand:
        sources.append("brand lookup")
        title = brand_title
        for hx, w, name in brand:
            counts[hx] += w
            if name:
                names.setdefault(hx, name)

    def left() -> float:
        return min(TIMEOUT_S, deadline - time.monotonic())

    try:
        if left() < 1:
            raise FetchError("That site took too long.")
        html, final = _get(url, MAX_HTML, left())
    except FetchError:
        if not brand:
            raise
    else:
        sources.append("page styles")
        host = urllib.parse.urlsplit(final).hostname or host
        parser = _PageParser()
        parser.feed(html)
        title = title or " ".join(parser.title.split())[:80]
        for theme in parser.theme:
            hx = _to_hex(theme)
            if hx:
                counts[hx] += 40
        for block in parser.styles:
            _colors_in(block, 1, counts, canvas_decls, css_vars)
        for href in parser.sheets[:MAX_SHEETS]:
            # Out of time: answer with what we have rather than keep the page waiting.
            if left() < 1:
                break
            try:
                css, _ = _get(urllib.parse.urljoin(final, href), MAX_CSS, left())
            except FetchError:
                continue
            _colors_in(css, 1, counts, canvas_decls, css_vars)
    if not counts:
        raise FetchError("No colors found on that page.")
    colors = _cluster(counts)
    canvas: Counter = Counter()
    for decl, local in canvas_decls:
        _add_colors(_resolve_vars(decl, local, css_vars), 1, canvas)
    top_canvas = canvas.most_common(1)[0][0] if canvas else None

    def name_for(hx: str) -> str:
        near = [(n, _dist(hx, k)) for k, n in names.items()]
        best = min(near, key=lambda nd: nd[1], default=("", 99))
        return best[0] if best[1] < 12 else ""

    shown = [c for c, _ in colors[:10]]
    return {
        "site": host.removeprefix("www."),
        "title": title,
        "sources": sources,
        "colors": [{"hex": c, "kind": _kind(c), "name": name_for(c)} for c in shown],
        "suggestion": suggest(colors, top_canvas),
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def do_GET(self):
        parts = urllib.parse.urlsplit(self.path)
        if parts.path != "/api/site-colors":
            return super().do_GET()
        query = urllib.parse.parse_qs(parts.query)
        try:
            body, status = lookup_with_cutoff(query.get("url", [""])[0]), 200
        except FetchError as exc:
            body, status = {"error": str(exc)}, 422
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8792)
    args = parser.parse_args()
    # Other local tools often hold nearby ports, so try the next few.
    for port in range(args.port, args.port + 10):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit(f"No free port between {args.port} and {args.port + 9}.")
    print(f"Style page: http://127.0.0.1:{port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
