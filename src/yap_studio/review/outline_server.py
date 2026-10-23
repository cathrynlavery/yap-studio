"""Local story outline review using the video review server's security rules."""

from __future__ import annotations

import errno
import hmac
import json
import re
import secrets
import threading
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

from yap_studio.errors import ValidationError, YapError
from yap_studio.feedback.batch import FeedbackError
from yap_studio.feedback.outline import (
    _load_outline,
    add_note,
    approve_outline,
    delete_note,
    list_notes,
    outline_status,
    send_notes,
    set_wait,
    update_note,
)
from yap_studio.paths import atomic_write_text
from yap_studio.project.lock import project_lock
from yap_studio.project.model import load_project
from yap_studio.review.server import (
    _CSP,
    _STATIC,
    _STATIC_NAME,
    _STATIC_TYPES,
    _QuietThreadingHTTPServer,
    accepts_query_token,
    parse_allowed_host,
)

_LINK_SCHEMA = "yap-studio/outline-link@1"
_NOTE_PATH = re.compile(r"/api/outline/notes/([^/]+)\Z")
_STALE = "The story changed. The agent is updating the outline."


class _RequestError(YapError):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message, error_type=code, code=code)
        self.status = status


def _load_link(path: Path) -> dict | None:
    """Read the private link directly into the server without logging its token."""
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(value, dict) or set(value) != {"schema", "token", "port"}
                or value["schema"] != _LINK_SCHEMA
                or not isinstance(value["token"], str) or not value["token"]
                or type(value["port"]) is not int or not 1 <= value["port"] <= 65535):
            raise ValueError("invalid link")
        return value
    except (OSError, UnicodeError, ValueError) as exc:
        raise YapError("Cannot read the saved story outline link.",
                       error_type="input_error", code="invalid_outline_link",
                       suggested_action="Restore .yap/outline-link.json and retry.") from exc


class OutlineServer:
    """Serve an existing outline, keeping the token and preferred port across restarts."""

    def __init__(self, project_dir: Path, *, host: str = "127.0.0.1", port: int = 0,
                 token: str | None = None, allowed_hosts: Sequence[str] = ()) -> None:
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValidationError("port", "must be an integer from 0 to 65535")
        if token is not None and (not isinstance(token, str) or not token):
            raise ValidationError("token", "must be a nonempty string")
        self.project_dir = Path(project_dir)
        self._allowed_hosts = tuple(parse_allowed_host(value) for value in allowed_hosts)
        self._requests = threading.RLock()
        link_path = self.project_dir / ".yap" / "outline-link.json"
        with project_lock(self.project_dir):
            self.project = load_project(self.project_dir / "project.json").name
            link = _load_link(link_path)
            self.token = token if token is not None else (
                link["token"] if link else secrets.token_urlsafe(24)
            )
            preferred = link["port"] if link else 0
            try:
                self._http = _QuietThreadingHTTPServer(
                    (host, port or preferred), self._handler_type(),
                )
            except OSError as exc:
                if port or not preferred or exc.errno != errno.EADDRINUSE:
                    raise
                self._http = _QuietThreadingHTTPServer((host, 0), self._handler_type())
            self.port = self._http.server_port
            try:
                # atomic_write_text uses mkstemp (0600) before atomic replacement.
                atomic_write_text(link_path, json.dumps({
                    "schema": _LINK_SCHEMA, "token": self.token, "port": self.port,
                }) + "\n")
            except BaseException:
                self._http.server_close()
                raise
        self.url = f"http://127.0.0.1:{self.port}/?t={quote(self.token, safe='')}"

    def serve_forever(self) -> None:
        """Handle requests until stopped from another thread."""
        self._http.serve_forever()

    def shutdown(self) -> None:
        """Stop serving from another thread."""
        self._http.shutdown()

    def server_close(self) -> None:
        """Release the listening socket."""
        self._http.server_close()

    def _accepts_host(self, host: str | None) -> bool:
        if host in {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}:
            return True
        if host is None or not self._allowed_hosts:
            return False
        try:
            name, port = parse_allowed_host(host)
        except ValidationError:
            return False
        return any(name == allowed_name and (allowed_port is None or port == allowed_port)
                   for allowed_name, allowed_port in self._allowed_hosts)

    def _require_fresh(self) -> None:
        state = outline_status(self.project_dir)["state"]
        if state == "stale":
            raise _RequestError(409, "outline_changed", _STALE)
        if state == "missing":
            raise _RequestError(409, "outline_missing", "Build the story outline first.")

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        review = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *args: object) -> None:
                pass

            def _headers(self, status: int, content_type: str | None, length: int,
                         extra: dict[str, str] | None = None) -> None:
                self.send_response(status)
                if content_type is not None:
                    self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(length))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Cache-Control", (extra or {}).get("Cache-Control", "no-store"))
                if content_type and content_type.startswith("text/html"):
                    self.send_header("Content-Security-Policy", _CSP)
                for key, value in (extra or {}).items():
                    if key != "Cache-Control":
                        self.send_header(key, value)
                self.end_headers()

            def _bytes(self, status: int, payload: bytes, content_type: str | None = None,
                       extra: dict[str, str] | None = None) -> None:
                self._headers(status, content_type, len(payload), extra)
                if self.command != "HEAD" and payload:
                    self.wfile.write(payload)

            def _json(self, status: int, value: object,
                      extra: dict[str, str] | None = None) -> None:
                self._bytes(status, (json.dumps(value, ensure_ascii=False) + "\n").encode(),
                            "application/json; charset=utf-8", extra)

            def _body(self, *, required: bool = True) -> dict:
                length_text = self.headers.get("Content-Length")
                if length_text is None and not required:
                    return {}
                if length_text is None or not length_text.isdecimal():
                    raise ValidationError("body", "Content-Length is required and must be valid")
                try:
                    length = int(length_text)
                except ValueError as exc:
                    raise ValidationError("body", "Content-Length must be valid") from exc
                if length > 65536:
                    raise _RequestError(413, "payload_too_large", "request body exceeds 64 KB")
                if not length and not required:
                    return {}
                try:
                    value = json.loads(self.rfile.read(length))
                except (UnicodeError, ValueError, RecursionError) as exc:
                    raise ValidationError("body", "must be valid JSON") from exc
                if not isinstance(value, dict):
                    raise ValidationError("body", "must be a JSON object")
                return value

            def _authorized(self, query: dict[str, list[str]]) -> bool:
                supplied = self.headers.get("X-Yap-Token")
                if supplied is None and accepts_query_token(self.command):
                    supplied = query.get("t", [None])[0]
                return supplied is not None and hmac.compare_digest(
                    supplied.encode("utf-8"), review.token.encode("utf-8"),
                )

            def _route(self) -> None:
                if not review._accepts_host(self.headers.get("Host")):
                    raise _RequestError(403, "forbidden", "invalid Host header")
                parsed = urlsplit(self.path)
                path = parsed.path
                query = parse_qs(parsed.query, keep_blank_values=True)
                if (path.startswith("/api/") or not accepts_query_token(self.command)) and not (
                    self._authorized(query)
                ):
                    raise _RequestError(403, "forbidden", "invalid review token")
                note_match = _NOTE_PATH.fullmatch(path)
                if path in {"/", "/api/outline"} or path.startswith("/static/"):
                    allowed = {"GET", "HEAD"}
                elif path in {"/api/outline/notes", "/api/outline/send",
                              "/api/outline/approve", "/api/outline/wait"}:
                    allowed = {"POST"}
                elif note_match:
                    allowed = {"PATCH", "DELETE"}
                else:
                    raise _RequestError(404, "not_found", "route not found")
                if self.command not in allowed:
                    error = _RequestError(405, "method_not_allowed", "method not allowed")
                    self._json(405, {"error": error.to_dict()},
                               {"Allow": ", ".join(sorted(allowed))})
                    return
                root = review.project_dir
                if path == "/api/outline":
                    with project_lock(root):
                        outline = _load_outline(root)
                        notes = list_notes(root)
                        current = [note for note in notes
                                   if outline and note["outline"] == outline["id"]]
                        self._json(200, {"project": review.project, "outline": outline,
                                         "status": outline_status(root), "notes": current,
                                         "older_notes": len(notes) - len(current)})
                elif path.startswith("/api/"):
                    needs_body = self.command == "PATCH" or path in {
                        "/api/outline/notes", "/api/outline/wait",
                    }
                    body = self._body(required=needs_body)
                    review._require_fresh()
                    if path == "/api/outline/notes":
                        if set(body) != {"beat", "kind", "text"}:
                            raise ValidationError("note", "requires beat, kind, and text")
                        self._json(201, add_note(root, **body))
                    elif note_match:
                        note_id = note_match[1]
                        note = next((note for note in list_notes(root)
                                     if note["id"] == note_id), None)
                        if note is None:
                            raise _RequestError(404, "not_found", "Unknown outline note.")
                        if note["batch"] is not None:
                            raise _RequestError(409, "note_sent", "Sent notes cannot be changed.")
                        if note["outline"] != outline_status(root)["outline"]:
                            raise _RequestError(409, "older_outline",
                                                "The note belongs to an older outline.")
                        if self.command == "PATCH":
                            if not body or set(body) - {"kind", "text"}:
                                raise ValidationError("note", "only kind and text can be updated")
                            self._json(200, update_note(root, note_id, **body))
                        else:
                            delete_note(root, note_id)
                            self._bytes(204, b"")
                    elif path == "/api/outline/send":
                        batch = send_notes(root)
                        self._json(200, {"batch": batch["id"], "notes": len(batch["items"])})
                    elif path == "/api/outline/approve":
                        self._json(200, approve_outline(root))
                    else:
                        if set(body) != {"minutes"}:
                            raise ValidationError("wait", "requires minutes")
                        self._json(200, set_wait(root, body["minutes"]))
                else:
                    name = "outline.html" if path == "/" else path.removeprefix("/static/")
                    if path != "/" and _STATIC_NAME.fullmatch(name) is None:
                        raise _RequestError(404, "not_found", "file not found")
                    target = _STATIC / name
                    if not target.is_file() or target.resolve().parent != _STATIC.resolve():
                        raise _RequestError(404, "not_found", "file not found")
                    content_type = "text/html; charset=utf-8" if path == "/" else (
                        _STATIC_TYPES[target.suffix]
                    )
                    self._bytes(200, target.read_bytes(), content_type)

            def _dispatch(self) -> None:
                try:
                    with review._requests:
                        try:
                            self._route()
                        except _RequestError as exc:
                            self._json(exc.status, {"error": exc.to_dict()})
                        except FeedbackError as exc:
                            message = exc.message
                            if exc.code == "outline_changed":
                                exc = _RequestError(409, "outline_changed", _STALE)
                                code = 409
                            elif "locked" in message or message.startswith((
                                "No unsent notes", "The note belongs to an older outline",
                            )):
                                code = 409
                            elif message.startswith("Unknown outline note"):
                                code = 404
                            else:
                                code = 400
                            self._json(code, {"error": exc.to_dict()})
                        except YapError as exc:
                            code = 409 if exc.error_type == "project_conflict" else 400
                            self._json(code, {"error": exc.to_dict()})
                        except Exception:
                            error = _RequestError(500, "internal_error", "internal server error")
                            self._json(500, {"error": error.to_dict()})
                except OSError:
                    pass

            def do_GET(self) -> None:
                self._dispatch()

            def do_HEAD(self) -> None:
                self._dispatch()

            def do_POST(self) -> None:
                self._dispatch()

            def do_PATCH(self) -> None:
                self._dispatch()

            def do_DELETE(self) -> None:
                self._dispatch()

            def __getattr__(self, name: str):
                if name.startswith("do_"):
                    return self._dispatch
                raise AttributeError(name)

        return Handler
