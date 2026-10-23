"""Authenticated local HTTP routes for the creator's home page."""

from __future__ import annotations

import hmac
import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

from yap_studio import video_type
from yap_studio.errors import ValidationError, YapError
from yap_studio.homepage import bridge
from yap_studio.homepage.state import (
    MEMO_SUFFIXES,
    OFFERED_TYPES,
    VIDEO_SUFFIXES,
    HomeState,
    inbox_files,
    public_value,
)
from yap_studio.local_http import LocalHTTPServer
from yap_studio.style.answers import ANSWERS_SCHEMA, apply_answers, load_answers, validate_answers

_STATIC = Path(__file__).parent / "static"
_PACKAGE = Path(__file__).parents[1]
_EXTENSIONS = {".css", ".js", ".mjs", ".svg", ".png", ".jpg", ".webp", ".woff2"}
_MAX_BODY = 1024 * 1024
_MAX_UPLOAD = 16 * 1024**3
_MIN_FREE = 2 * 1024**3
_CSP = (
    "default-src 'self'; img-src 'self' data: blob:; font-src 'self'; "
    "media-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


class _RequestError(Exception):
    def __init__(
        self, status: int, message: str, code: str = "home_request", *, html: bool = False
    ) -> None:
        self.status, self.message, self.code = status, message, code
        self.html = html
        self.local_link: str | None = None


def launch_style_page(workspace: Path, phone: bool) -> dict:
    """Start or reuse the existing detached style picker without opening a browser."""
    command = [
        sys.executable,
        "-m",
        "yap_studio",
        "style",
        "pick",
        "--workspace",
        str(workspace),
        "--inbox",
        str(workspace / "inbox"),
        "--detach",
        "--json",
        "--no-open",
    ]
    if phone:
        command.append("--phone")
    result = subprocess.run(command, capture_output=True, text=True, timeout=150)
    if result.returncode:
        raise YapError(
            "The style page could not open. Ask your agent to check it.", code="home_style_page"
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _sources(data: dict) -> tuple[str, list[str]]:
    kind = data.get("video_type")
    sources = data.get("broll_sources")
    if kind not in OFFERED_TYPES:
        raise ValidationError("video_type", "Choose talking head or voiceover.")
    if not isinstance(sources, list) or any(not isinstance(s, str) for s in sources):
        raise ValidationError("broll_sources", "Choose where your pictures may come from.")
    sources = video_type.parse_sources(sources)
    return kind, sources


def _name(name: str) -> bool:
    return bool(
        name
        and not name.startswith(".")
        and "/" not in name
        and "\\" not in name
        and not any(ord(c) < 32 or ord(c) == 127 for c in name)
    )


class HomeServer:
    """Serve a workspace without performing edits, renders or paid actions."""

    def __init__(
        self,
        workspace: Path,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        allowed_hosts: tuple[str, ...] = (),
        phone: dict | None = None,
    ) -> None:
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValidationError("port", "Choose a port from 0 to 65535.")
        self.workspace = bridge.workspace_dir(workspace)
        bridge.directory(self.workspace)
        with bridge.writer(self.workspace) as root:
            if not (root / "seen.json").exists():
                bridge.write_json(
                    root / "seen.json", bridge._seed(*bridge._history(self.workspace))
                )
        self.state = HomeState(self.workspace, phone=phone)
        self.host = host
        self.allowed_hosts = {name.lower() for name in allowed_hosts}
        self.token = secrets.token_urlsafe(24)
        self._http = LocalHTTPServer((host, port), self._handler_type())
        self._http.daemon_threads = True
        self.port = self._http.server_port
        self.cookie_name = f"yap_home_{self.port}"
        link_host = "127.0.0.1" if host == "0.0.0.0" else host
        self.url = f"http://{link_host}:{self.port}/?t={quote(self.token, safe='')}"

    def serve_forever(self) -> None:
        """Serve until explicitly stopped."""
        self._http.serve_forever()

    def shutdown(self) -> None:
        """Stop a running HTTP loop from another thread."""
        self._http.shutdown()

    def server_close(self) -> None:
        """Release the listening socket."""
        self._http.server_close()

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        home = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format, *args):
                pass

            def _headers(
                self, status: int, length: int, mime: str, disposition: str | None = None,
                *, location: str | None = None,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(length))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Security-Policy", _CSP)
                query = parse_qs(urlsplit(self.path).query)
                if query.get("t") == [home.token]:
                    self.send_header(
                        "Set-Cookie",
                        f"{home.cookie_name}={home.token}; HttpOnly; SameSite=Strict; Path=/",
                    )
                if disposition:
                    self.send_header("Content-Disposition", disposition)
                if location:
                    self.send_header("Location", location)
                self.end_headers()

            def _send(self, status: int, payload: bytes, mime: str) -> None:
                self._headers(status, len(payload), mime)
                if self.command != "HEAD":
                    self.wfile.write(payload)

            def _json(self, status: int, data: dict) -> None:
                self._send(
                    status,
                    json.dumps(data, ensure_ascii=False).encode(),
                    "application/json; charset=utf-8",
                )

            def _length(self, maximum: int) -> int:
                lengths = self.headers.get_all("Content-Length", [])
                if (
                    len(lengths) != 1
                    or not lengths[0].isascii()
                    or not lengths[0].isdecimal()
                    or self.headers.get("Transfer-Encoding") is not None
                ):
                    raise _RequestError(400, "Send a file size with Content-Length.")
                if len(lengths[0]) > 20 or int(lengths[0]) > maximum:
                    raise _RequestError(413, "This file or request is too large.")
                return int(lengths[0])

            def _body(self) -> dict:
                if self.headers.get_content_type() != "application/json":
                    raise _RequestError(415, "Send this request as application/json.")
                length = self._length(_MAX_BODY)
                self.connection.settimeout(30)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise _RequestError(400, "The request was interrupted. Try again.")
                try:
                    data = json.loads(raw)
                except (ValueError, UnicodeError):
                    raise _RequestError(
                        400, "The request is not valid JSON. Refresh and try again."
                    ) from None
                if not isinstance(data, dict):
                    raise _RequestError(400, "Send a JSON object.")
                return data

            def _authorize(self) -> tuple[str, bool]:
                hosts = self.headers.get_all("Host", [])
                host = hosts[0].lower() if len(hosts) == 1 else ""
                local = {f"127.0.0.1:{home.port}", f"localhost:{home.port}"}
                valid = {f"{home.host}:{home.port}"}
                if home.host in {"127.0.0.1", "localhost", "0.0.0.0"}:
                    valid |= local
                parsed_host = urlsplit("//" + host)
                if host not in valid and not (
                    parsed_host.hostname in home.allowed_hosts
                    and re.fullmatch(r"[a-z0-9.-]+(?::[0-9]+)?", host)
                ):
                    raise _RequestError(403, "Open the home page using its original link.")
                parsed = urlsplit(self.path)
                if parsed.scheme or parsed.netloc:
                    raise _RequestError(403, "Open a page on this home server.")
                cookies = dict(
                    part.strip().split("=", 1)
                    for part in self.headers.get("Cookie", "").split(";")
                    if "=" in part
                )
                token = (
                    self.headers.get("X-Home-Token")
                    if self.command == "POST"
                    else parse_qs(parsed.query).get("t", [None])[0] or cookies.get(home.cookie_name)
                )
                if not token or not hmac.compare_digest(token.encode(), home.token.encode()):
                    error = _RequestError(
                        403, "Open the home page using its private link.",
                        html=self.command == "GET" and unquote(parsed.path) == "/",
                    )
                    # A page opened on the tailnet name may be on this Mac, which has the cookie
                    # for the loopback address. The link carries no token.
                    if home.host == "127.0.0.1" and host not in local:
                        error.local_link = f"http://127.0.0.1:{home.port}/"
                    raise error
                if self.command == "POST":
                    origin = self.headers.get("Origin")
                    if origin is not None and origin not in {f"http://{host}", f"https://{host}"}:
                        raise _RequestError(403, "Refresh the home page before trying again.")
                return unquote(parsed.path), host in local

            def _file(
                self, path: Path, *, mime: str | None = None, attachment: bool | str = False
            ) -> None:
                if path.is_symlink() or not path.is_file():
                    raise _RequestError(404, "This file is not available. Refresh the page.")
                with path.open("rb") as source:
                    disposition = None
                    if attachment:
                        # A named attachment saves as the project, not as the version's file name.
                        filename = attachment if isinstance(attachment, str) else path.name
                        safe = re.sub(r"[^A-Za-z0-9_. -]", "_", filename)
                        disposition = f'attachment; filename="{safe}"'
                    self._headers(
                        200,
                        os.fstat(source.fileno()).st_size,
                        mime or mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                        disposition,
                    )
                    if self.command != "HEAD":
                        shutil.copyfileobj(source, self.wfile, 1024 * 1024)

            def _asset(self, root: Path, name: str, extensions: set[str]) -> None:
                target = root / name
                if (
                    not _name(name)
                    or target.suffix.lower() not in extensions
                    or not target.resolve().is_relative_to(root.resolve())
                ):
                    raise _RequestError(404, "This file is not available.")
                self._file(target)

            def _upload(self) -> None:
                import fcntl

                if self.headers.get_content_type() != "application/octet-stream":
                    raise _RequestError(415, "Send the file as application/octet-stream.")
                encoded_name = self.headers.get("X-File-Name", "")
                if re.search(r"%(?![0-9A-Fa-f]{2})", encoded_name):
                    raise _RequestError(400, "The file name is not encoded correctly. Try again.")
                try:
                    name = unquote(encoded_name, errors="strict")
                except UnicodeError:
                    raise _RequestError(
                        400, "The file name is not encoded correctly. Try again."
                    ) from None
                if (
                    not _name(name)
                    or Path(name).suffix.lower() not in VIDEO_SUFFIXES | MEMO_SUFFIXES
                ):
                    raise _RequestError(
                        400, "Choose a video or audio file with a simple file name."
                    )
                length = self._length(_MAX_UPLOAD)
                inbox = home.workspace / "inbox"
                if inbox.is_symlink():
                    raise _RequestError(400, "Replace the linked inbox with a folder.")
                if shutil.disk_usage(inbox).free - length < _MIN_FREE:
                    raise _RequestError(413, "Free up space first. Keep at least 2 GiB available.")
                root = bridge.directory(home.workspace)
                fd = os.open(root / "upload.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, "a+b") as lock:
                    os.fchmod(lock.fileno(), 0o600)
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    if shutil.disk_usage(inbox).free - length < _MIN_FREE:
                        raise _RequestError(413, "Free up space, then upload the file again.")
                    temporary = inbox / f".{name}.part"
                    created = False
                    try:
                        self.connection.settimeout(30)
                        output_fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                        created = True
                        with os.fdopen(output_fd, "wb") as output:
                            remaining = length
                            while remaining:
                                chunk = self.rfile.read(min(1024 * 1024, remaining))
                                if not chunk:
                                    raise _RequestError(
                                        400, "The upload stopped. Try the file again."
                                    )
                                output.write(chunk)
                                remaining -= len(chunk)
                            output.flush()
                            os.fsync(output.fileno())
                        if shutil.disk_usage(inbox).free < _MIN_FREE:
                            raise _RequestError(413, "Free up space, then upload the file again.")
                        destination = inbox / name
                        number = 2
                        while True:
                            try:
                                # Hard-link publication is atomic and cannot overwrite a file.
                                os.link(temporary, destination)
                                break
                            except FileExistsError:
                                destination = (
                                    inbox / f"{Path(name).stem} ({number}){Path(name).suffix}"
                                )
                                number += 1
                        temporary.unlink()
                    except FileExistsError:
                        raise _RequestError(
                            409, "This file is already uploading. Try again shortly."
                        ) from None
                    finally:
                        if created:
                            temporary.unlink(missing_ok=True)
                self._json(201, home.state.take_entry(destination))

            def _create_request(self, data: dict) -> dict:
                kind = data.get("kind")
                if kind == "learn":
                    return bridge.create_request(home.workspace, kind, {})
                if kind == "final":
                    return bridge.create_request(
                        home.workspace, kind, {"project": data.get("project")}
                    )
                if kind == "message":
                    return bridge.create_request(
                        home.workspace,
                        kind,
                        {"project": data.get("project"), "text": data.get("text")},
                    )
                if kind != "edit":
                    raise ValidationError(
                        "kind", "Choose make my video, learn my style, final or message."
                    )
                video, sources = _sources(data)
                takes = data.get("takes")
                if (
                    not isinstance(takes, list)
                    or any(not isinstance(p, str) for p in takes)
                    or len(takes) != len(set(takes))
                ):
                    raise ValidationError("takes", "Choose each inbox file once.")
                available, _ = inbox_files(home.workspace)
                by_name = {p.name: p for p in available}
                if any(name not in by_name for name in takes):
                    raise ValidationError(
                        "takes", "These files changed. Refresh and choose them again."
                    )
                kinds = {home.state.metadata(by_name[name])[0] for name in takes}
                if len(kinds) > 1:
                    raise ValidationError("takes", "Choose only video takes or only audio memos.")
                if not takes and video != "voiceover":
                    raise ValidationError("takes", "Add a video take to your inbox first.")
                if not takes and not {"album", "folder"}.intersection(sources):
                    raise ValidationError(
                        "broll", "Choose a Photos album or folder for your voiceover."
                    )
                video_type.check_fit(video, sources, voiceover=(not takes or kinds == {"memo"}))
                shape = data.get("shape", "vertical")
                if shape not in ("vertical", "horizontal"):
                    raise ValidationError("shape", "Choose vertical or horizontal.")
                for name in takes:
                    media_kind, _, width, height = home.state.metadata(by_name[name])
                    if media_kind != "take":
                        continue
                    if not width or not height:
                        if shape == "horizontal":
                            raise ValidationError(
                                "takes", f"Cannot read the shape of {name}. "
                                "Add a readable horizontal take.",
                            )
                        continue
                    take_shape = (
                        "horizontal" if width > height
                        else "vertical" if height > width else "square"
                    )
                    if take_shape != shape and "horizontal" in (shape, take_shape):
                        fix = f"Replace this take with a {shape} take."
                        if take_shape != "square":
                            fix = f"Choose {take_shape} or replace it with a {shape} take."
                        raise ValidationError(
                            "takes", f"{name} is {take_shape}, but this video is {shape}. {fix}",
                        )
                album = data.get("album")
                if album is not None and (not isinstance(album, str) or len(album) > 1000):
                    raise ValidationError("album", "Use the name of your Photos album.")
                hook = data.get("text_hook", "")
                if not isinstance(hook, str):
                    raise ValidationError("text_hook", "Write the Text hook as plain text.")
                hook = " ".join(hook.split())
                if hook:
                    from yap_studio.text_hook import fit_text_hook

                    fit_text_hook(hook)
                return bridge.create_request(
                    home.workspace,
                    "edit",
                    {"video_type": video, "broll_sources": sources, "takes": takes, "album": album,
                     "shape": shape,
                     **({"captions": data.get("captions", False)} if shape == "horizontal" else {}),
                     **({"text_hook": hook} if hook else {})},
                )

            def _post(self, route: str, on_this_mac: bool) -> None:
                if route == "/api/inbox":
                    self._upload()
                    return
                if route not in {
                    "/api/video-type",
                    "/api/answers",
                    "/api/style-page",
                    "/api/requests",
                    "/api/reveal",
                    "/api/open",
                } and not re.fullmatch(r"/api/requests/r[1-9][0-9]*/(?:cancel|resume)", route):
                    raise _RequestError(404, "This action is not available.")
                data = self._body()
                if route == "/api/video-type":
                    kind, sources = _sources(data)
                    video_type.save_default(kind, sources)
                    self._json(200, {"video_type": kind, "broll_sources": sources})
                elif route == "/api/answers":
                    answers = data.get("answers")
                    if not isinstance(answers, dict):
                        raise ValidationError("answers", "Send your answers as an object.")
                    with bridge.writer(home.workspace):
                        target = home.workspace / "style/answers.json"
                        previous = load_answers(target) if target.is_file() else {}
                        merged = {**previous, **answers, "schema": ANSWERS_SCHEMA}
                        merged = validate_answers(merged)
                        target.parent.mkdir(exist_ok=True)
                        bridge.write_json(target, merged)
                        result = apply_answers(target, workspace=home.workspace)
                    self._json(
                        200,
                        {
                            "applied": result["fields_set"],
                            "pending": list(result["pending"].values()),
                        },
                    )
                elif route == "/api/style-page":
                    # From the Mac, keep the style page on this computer's address so the way
                    # back lands on the same page; a phone gets the phone link.
                    phone = bool(home.state.phone) and not on_this_mac
                    self._json(200, launch_style_page(home.workspace, phone))
                elif route == "/api/requests":
                    self._json(201, self._create_request(data))
                elif route.endswith("/cancel"):
                    self._json(200, bridge.cancel(home.workspace, route.split("/")[3]))
                elif route.endswith("/resume"):
                    self._json(200, bridge.resume(home.workspace, route.split("/")[3]))
                else:
                    if not on_this_mac:
                        raise _RequestError(403, "Open this page on your Mac to show the file.")
                    value = data.get("project")
                    if not isinstance(value, str):
                        raise ValidationError("project", "Choose a project from this workspace.")
                    relative = bridge.relative_project(home.workspace, value)
                    target = home.state.approved_file(Path(relative).name)
                    if target is None:
                        raise _RequestError(
                            404, "Approve the finished video before showing its file."
                        )
                    if sys.platform != "darwin":
                        raise _RequestError(400, "Download the finished video on this computer.")
                    command = (
                        ["open", "-R", str(target)]
                        if route == "/api/reveal"
                        else ["open", str(target)]
                    )
                    subprocess.run(command, check=True, timeout=10)
                    self._json(200, {"revealed" if route == "/api/reveal" else "opened": True})

            def _get(self, route: str, on_this_mac: bool) -> None:
                if route == "/":
                    if self.command == "GET" and "t" not in parse_qs(
                        urlsplit(self.path).query, keep_blank_values=True
                    ):
                        self._headers(
                            303, 0, "text/html; charset=utf-8", location=f"/?t={home.token}"
                        )
                    else:
                        self._file(_STATIC / "home.html", mime="text/html; charset=utf-8")
                elif route == "/api/state":
                    self._json(200, home.state.snapshot(on_this_mac=on_this_mac))
                elif route.startswith("/static/"):
                    self._asset(_STATIC, route.removeprefix("/static/"), _EXTENSIONS)
                elif route.startswith("/fonts/"):
                    self._asset(_PACKAGE / "fonts", route.removeprefix("/fonts/"), {".woff2"})
                elif route == "/pictures/presenter.jpg":
                    self._file(_PACKAGE / "gallery/sample-still.jpg")
                elif route == "/pictures/shot.jpg":
                    self._file(_PACKAGE / "gallery/static/shot.jpg")
                elif route.startswith("/thumb/"):
                    target = home.state.thumbnail(route.removeprefix("/thumb/"))
                    if target is None:
                        raise _RequestError(404, "This take has no preview image.")
                    self._file(target, mime="image/jpeg")
                elif route.startswith("/api/download/"):
                    name = route.removeprefix("/api/download/")
                    target = home.state.approved_file(name) if _name(name) else None
                    if target is None:
                        raise _RequestError(
                            404, "Approve the finished video before downloading it."
                        )
                    self._file(target, mime="video/mp4", attachment=f"{name}{target.suffix}")
                else:
                    raise _RequestError(404, "This page or file is not available.")

            def _dispatch(self) -> None:
                try:
                    route, local = self._authorize()
                    if self.command == "POST":
                        self._post(route, local)
                    else:
                        self._get(route, local)
                except bridge.UnknownRequestError as exc:
                    self._json(404, {"error": {"code": exc.code, "message": exc.message}})
                except bridge.ConflictError as exc:
                    self._json(409, {"error": {"code": exc.code, "message": exc.message}})
                except _RequestError as exc:
                    if exc.html:
                        local_link = (
                            f'<p><a href="{exc.local_link}">Open Yap Studio on this Mac</a></p>'
                            if exc.local_link
                            else ""
                        )
                        self._send(
                            exc.status,
                            (
                                '<!doctype html><html lang="en"><meta charset="utf-8">'
                                "<title>Yap Studio</title><p>This page needs its private link. "
                                f"Ask Claude to open Yap Studio for you.</p>{local_link}</html>"
                            ).encode(),
                            "text/html; charset=utf-8",
                        )
                    else:
                        self._json(
                            exc.status, {"error": {"code": exc.code, "message": exc.message}}
                        )
                except (YapError, ValueError) as exc:
                    self._json(
                        400,
                        {
                            "error": {
                                "code": getattr(exc, "code", "home_invalid"),
                                "message": public_value(str(exc), home.workspace),
                            }
                        },
                    )
                except (OSError, subprocess.SubprocessError):
                    self._json(
                        400,
                        {
                            "error": {
                                "code": "home_io",
                                "message": "Could not open the file. Refresh and try again.",
                            }
                        },
                    )

            do_GET = _dispatch
            do_HEAD = _dispatch
            do_POST = _dispatch

        return Handler
