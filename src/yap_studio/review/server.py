"""Loopback HTTP server for local video reviews."""

from __future__ import annotations

import errno
import hmac
import json
import math
import os
import re
import secrets
import socketserver
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from fractions import Fraction
from html import escape
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

from yap_studio import paths, receipt
from yap_studio.captions import _stripped
from yap_studio.checks.model import format_time
from yap_studio.errors import InputError, ValidationError, YapError
from yap_studio.feedback.clarify import (
    CarriedQuestionError,
    UnknownQuestionError,
    answer_clarification,
    answered_questions,
    pending_questions,
)
from yap_studio.feedback.moments import map_moment
from yap_studio.feedback.release import approval_record, approve, clear_approval, rollback
from yap_studio.feedback.removed import removed_ranges
from yap_studio.feedback.versions import (
    VersionError,
    current_version,
    find_version_by_media,
    load_version,
    version_layers,
    version_report,
)
from yap_studio.homepage import bridge as home_bridge
from yap_studio.local_http import LocalHTTPServer
from yap_studio.project.lock import project_lock, write_json_cas
from yap_studio.project.model import CutWords, TextHook, load_words, parse_document
from yap_studio.project.word_fixes import change_word_fix
from yap_studio.review.captions import caption_phrases
from yap_studio.review.export import (
    comment_to_dict,
    earlier_threads,
    export_filename,
    format_text,
    format_threads,
    format_threads_summary,
    series_export,
    to_export_json,
)
from yap_studio.review.findings import findings_for_media
from yap_studio.review.media import KnownMedia, ReviewMedia, file_identity
from yap_studio.review.model import (
    conform_comment,
    frame_at,
    frame_start,
    normalize_input,
    sorted_comments,
)
from yap_studio.review.phone import link_label
from yap_studio.review.rounds import comment_states
from yap_studio.review.send import send_round
from yap_studio.review.series import thread_to_dict
from yap_studio.review.store import (
    ReviewStore,
    RoundConflictError,
    UnknownCommentError,
    UnknownRoundError,
    round_to_dict,
)
from yap_studio.review.video_map import video_map
from yap_studio.taste import TasteRefusal, UnknownProposalError, decide, inbox_view

_STATIC = Path(__file__).parent / "static"
_STATIC_NAME = re.compile(r"[A-Za-z0-9_-]+\.(css|mjs|svg)\Z")
_COMMENT_PATH = re.compile(r"/api/comments/(c[1-9][0-9]*)\Z")
_RETRY_PATH = re.compile(r"/api/rounds/(n[1-9][0-9]*)/retry\Z")
_VIDEO_PATH = re.compile(r"/v/([1-9][0-9]*)(/.*)\Z")
_RANGE = re.compile(r"bytes=(\d*)-(\d*)\Z")
_HOST = re.compile(
    r"([A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*)(?::([0-9]+))?\Z"
)
_CSP = (
    "default-src 'self'; img-src 'self' data:; media-src 'self'; style-src 'self'; "
    "script-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)
_STATIC_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}
_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
}
_SAFE_FILENAME_CHARACTER = re.compile(r"[^A-Za-z0-9._ -]")
_MEDIA_CHANGED_MESSAGE = (
    "The video changed on disk after this review started. "
    "Restart yap-studio review to review the new version."
)


def accepts_query_token(method: str) -> bool:
    """Accept link tokens for GET and HEAD; writes require the token header."""
    # Links and media elements cannot send headers. The link holder is the reviewer.
    return method in {"GET", "HEAD"}


def parse_allowed_host(value: str) -> tuple[str, int | None]:
    """Validate a plain host with an optional port and normalize it for matching."""
    detail = "must be a hostname or IPv4 address, optionally with a port from 1 to 65535"
    match = _HOST.fullmatch(value)
    if match is None:
        raise ValidationError("allow-host", detail)
    name, port_text = match.groups()
    try:
        port = int(port_text) if port_text is not None else None
    except ValueError as exc:
        raise ValidationError("allow-host", detail) from exc
    if port is not None and not 1 <= port <= 65535:
        raise ValidationError("allow-host", detail)
    return name.lower(), port


class _QuietThreadingHTTPServer(LocalHTTPServer):
    daemon_threads = True

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        self.server_name = self.server_address[0]
        self.server_port = self.server_address[1]

    def handle_error(self, request: object, client_address: tuple[str, int]) -> None:
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


@dataclass(frozen=True)
class ReviewVersion:
    """One media identity and optional occurrence map in a review series."""

    number: int
    media: ReviewMedia
    sha256: str
    media_identity: tuple[int, int, int, int, int]
    words: CutWords | None
    version_id: str | None
    _known_media: KnownMedia = field(default_factory=KnownMedia, compare=False, repr=False)

    def _media_changed(self, stream=None) -> bool:
        return self._known_media.changed(self.media.path, self.sha256, self.media_identity, stream)


@dataclass(frozen=True)
class ReviewSeries:
    """The ordered versions served together under one video's token."""

    id: str
    versions: tuple[ReviewVersion, ...]
    project_dir: Path | None = None
    preferred_port: int | None = None


def _start_send(work: Callable[[], None]) -> None:
    threading.Thread(target=work, daemon=True).start()


@dataclass(frozen=True)
class ReviewVideo:
    """Keep a video's authorization and launch identity together."""

    media: ReviewMedia
    sha256: str
    previous_sha: str | None = None
    token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    media_identity: tuple[int, int, int, int, int] | None = None
    series: ReviewSeries | None = None
    _known_media: KnownMedia = field(default_factory=KnownMedia, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.series is not None:
            newest = self.series.versions[-1]
            object.__setattr__(self, "media", newest.media)
            object.__setattr__(self, "sha256", newest.sha256)
            object.__setattr__(self, "media_identity", newest.media_identity)
            object.__setattr__(self, "previous_sha", None)
        if self.media_identity is None:
            identity = file_identity(self.media.path)
            if identity is None:
                raise InputError(str(self.media.path))
            object.__setattr__(self, "media_identity", identity)

    def _media_changed(self, stream=None) -> bool:
        return self._known_media.changed(self.media.path, self.sha256, self.media_identity, stream)


class ReviewServer:
    """Serve isolated video reviews on one loopback or private-network HTTP port."""

    def __init__(
        self,
        media: ReviewMedia,
        sha256: str,
        previous_sha: str | None,
        store: ReviewStore,
        host: str = "127.0.0.1",
        port: int = 0,
        token: str | None = None,
        media_identity: tuple | None = None,
        allowed_hosts: Sequence[str] = (),
        additional_videos: Sequence[ReviewVideo] = (),
        *,
        series: ReviewSeries | None = None,
        send_runner: Callable[[Callable[[], None]], None] | None = None,
    ) -> None:
        from yap_studio.review.phone import is_lan_address

        # A private network address serves a phone on the same Wi-Fi; the token still guards it.
        if host != "127.0.0.1" and not is_lan_address(host):
            raise ValueError("review server must bind to 127.0.0.1 or a private network address")
        self.host = host
        self._allowed_hosts = tuple(parse_allowed_host(entry) for entry in allowed_hosts)
        self.store = store
        self.owner = {"pid": os.getpid(), "instance": secrets.token_hex(16)}
        self._send_runner = send_runner if send_runner is not None else _start_send
        link = store.series_link(series.id) if series is not None else None
        self.token = token if token is not None else (
            link["token"] if link is not None else secrets.token_urlsafe(24)
        )
        first = ReviewVideo(media, sha256, previous_sha, self.token, media_identity, series)
        self.media, self.sha256, self.previous_sha = first.media, first.sha256, first.previous_sha
        self.series = first.series
        self.phone: dict | None = None
        self._media_identity = first.media_identity
        self._known_media = first._known_media
        self.videos = (
            first,
            *additional_videos,
        )
        if len({video.sha256 for video in self.videos}) != len(self.videos):
            raise ValidationError("media", "duplicate video; give each video's contents only once")
        self.list_token = secrets.token_urlsafe(24) if len(self.videos) > 1 else None
        recovered = set()
        for video in self.videos:
            if video.series is not None and video.series.id not in recovered:
                store.recover_rounds(video.series.id, instance=self.owner["instance"])
                recovered.add(video.series.id)
        preferred = (series.preferred_port or link["port"]) if series is not None else None
        binding_port = port or preferred or 0
        try:
            self._http = _QuietThreadingHTTPServer((host, binding_port), self._handler_type())
        except OSError as exc:
            if port != 0 or not preferred or exc.errno != errno.EADDRINUSE:
                raise
            self._http = _QuietThreadingHTTPServer((host, 0), self._handler_type())
        self.port = self._http.server_port
        try:
            if series is not None and self.port:
                store.save_link_port(series.id, self.port)
        except Exception:
            self._http.server_close()
            raise
        self.list_url = (
            f"http://{self.host}:{self.port}/?t={self.list_token}"
            if self.list_token is not None else None
        )
        self.url = self.list_url or self.video_urls[0]

    @property
    def video_urls(self) -> tuple[str, ...]:
        """Return each token link in argument order."""
        origin = f"http://{self.host}:{self.port}"
        if len(self.videos) == 1:
            return (f"{origin}/?t={self.token}",)
        return tuple(
            f"{origin}/v/{number}/?t={video.token}"
            for number, video in enumerate(self.videos, 1)
        )

    @property
    def media_identity(self) -> tuple:
        """Return the first video's identity captured before serving."""
        return self._media_identity

    def serve_forever(self) -> None:
        """Handle requests until shutdown is called from another thread."""
        self._http.serve_forever()

    def shutdown(self) -> None:
        """Stop a running server from another thread."""
        self._http.shutdown()

    def server_close(self) -> None:
        """Release the listening socket."""
        self._http.server_close()

    def _media_changed(self, stream=None) -> bool:
        if not hasattr(self, "_known_media"):
            self._known_media = KnownMedia()
        return self._known_media.changed(self.media.path, self.sha256, self._media_identity,
                                         stream)

    def _accepts_host(self, host: str | None) -> bool:
        if host in {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}:
            return True
        if host == f"{self.host}:{self.port}":
            return True
        if host is None or not self._allowed_hosts:
            return False
        try:
            name, port = parse_allowed_host(host)
        except ValidationError:
            return False
        return any(
            name == allowed_name and (allowed_port is None or port == allowed_port)
            for allowed_name, allowed_port in self._allowed_hosts
        )

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        review = self

        class Handler(BaseHTTPRequestHandler):
            _video: ReviewVideo | ReviewServer = review
            _selected: ReviewVideo | ReviewVersion | ReviewServer = review
            _version_number: int | None = None

            def log_message(self, _format: str, *args: object) -> None:
                pass

            def _headers(
                self,
                status: int,
                content_type: str | None,
                length: int,
                extra: dict[str, str] | None = None,
            ) -> None:
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

            def _bytes(
                self,
                status: int,
                payload: bytes,
                content_type: str | None = None,
                extra: dict[str, str] | None = None,
            ) -> None:
                self._headers(status, content_type, len(payload), extra)
                if self.command != "HEAD" and payload:
                    self.wfile.write(payload)

            def _json(
                self, status: int, value: object, extra: dict[str, str] | None = None
            ) -> None:
                payload = (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
                self._bytes(status, payload, "application/json; charset=utf-8", extra)

            def _error(
                self,
                status: int,
                kind: str,
                message: str,
                extra: dict[str, str] | None = None,
            ) -> None:
                self._json(
                    status, {"error": {"type": kind, "code": kind, "message": message}}, extra
                )

            def _body(self) -> dict:
                length_text = self.headers.get("Content-Length")
                if length_text is None or not length_text.isdecimal():
                    raise ValidationError("body", "Content-Length is required and must be valid")
                length = int(length_text)
                if length > 65536:
                    self._error(413, "payload_too_large", "request body exceeds 64 KB")
                    raise _Handled()
                try:
                    value = json.loads(self.rfile.read(length))
                except (UnicodeError, ValueError) as exc:
                    raise ValidationError("body", "must be valid JSON") from exc
                if not isinstance(value, dict):
                    raise ValidationError("body", "must be a JSON object")
                return value

            def _authorized(self, query: dict[str, list[str]], token: str | None = None) -> bool:
                supplied = self.headers.get("X-Yap-Token")
                if supplied is None and accepts_query_token(self.command):
                    supplied = query.get("t", [None])[0]
                expected = self._video.token if token is None else token
                return supplied is not None and hmac.compare_digest(
                    supplied.encode("utf-8"), expected.encode("utf-8")
                )

            def _previous(self) -> tuple | None:
                if self._video.previous_sha is None:
                    return None
                comments = review.store.comments(self._video.previous_sha)
                if not comments:
                    return None
                facts = review.store.review_facts(self._video.previous_sha)
                if facts is None:
                    facts = self._video.media.facts(self._video.previous_sha)
                conformed = [conform_comment(comment, facts) for comment in comments]
                return facts, sorted_comments(conformed)

            def _current(self) -> tuple:
                facts = self._selected.media.facts(self._selected.sha256)
                saved = review.store.comments(self._selected.sha256)
                comments = sorted_comments([conform_comment(comment, facts) for comment in saved])
                return facts, comments

            def _series_json(self) -> dict | None:
                series = self._video.series
                if series is None:
                    return None
                versions = []
                current = current_version(series.project_dir) if series.project_dir else None
                approval = approval_record(series.project_dir, verify_media=False) if (
                    series.project_dir
                ) else None
                numbers = {item.version_id: item.number for item in series.versions}
                for item in series.versions:
                    manifest = load_version(series.project_dir, item.version_id) if (
                        series.project_dir is not None) else None
                    report = version_report(manifest) if manifest is not None else None
                    value = {"number": item.number, "name": item.media.name,
                             "sha256": item.sha256, "version_id": item.version_id,
                             "note": manifest.note if manifest is not None else None,
                             "final": bool(manifest and
                                           manifest.media["source"].startswith("final/")),
                             "report_status": report["status"] if report else None}
                    value["youtube_chapters"] = (((report or {}).get("qa") or {}).get(
                        "youtube_chapters") if value["final"] else None)
                    value.update(duration_s=item.media.duration_s, fps=None,
                                 cfr=item.media.cfr, word_map=item.words is not None,
                                 comment_count=len(review.store.comments(item.sha256)))
                    if item.media.fps is not None:
                        value["fps"] = (
                            f"{item.media.fps.numerator}/{item.media.fps.denominator}"
                        )
                    versions.append(value)
                threads = review.store.threads(series.id, self._version_number)
                return {
                    "id": series.id,
                    "title": series.project_dir.name if series.project_dir else None,
                    "current": self._version_number, "newest": len(series.versions),
                    "current_version": numbers.get(current.id) if current else len(series.versions),
                    "approved": {"version": numbers[approval["version"]],
                                 "approved_at": approval["approved_at"],
                                 "report_status": approval["report_status"]} if (
                        approval and approval["version"] in numbers
                    ) else None,
                    "project": series.project_dir is not None,
                    "versions": versions,
                    "threads": [thread_to_dict(item) for item in threads],
                }

            def _layers(self, sha256: str, version_id: str | None = None):
                series = self._video.series
                root = series.project_dir if series is not None else next(
                    (parent for parent in self._video.media.path.parents
                     if (parent / "project.json").is_file()), None,
                )
                if root is None:
                    return None
                manifest = (load_version(root, version_id) if version_id is not None else
                            find_version_by_media(root, sha256))
                return version_layers(manifest) if manifest is not None else None

            def _text(self, facts, comments, reviewer, exported_on: date, *,
                      earlier: bool = False) -> str:
                series = self._video.series
                version = (
                    (self._version_number, len(series.versions)) if series is not None else None
                )
                version_id = (self._selected.version_id
                              if isinstance(self._selected, ReviewVersion) else None)
                text = format_text(facts, comments, reviewer, exported_on, version=version,
                                   layers=self._layers(facts.sha256, version_id))
                if series is not None:
                    threads = review.store.threads(series.id, self._version_number)
                    if threads:
                        text += "\n" + format_threads_summary(threads)
                        if earlier:
                            facts_by_version = {item.number: item.media.facts(item.sha256)
                                                for item in series.versions}
                            layers_by_version = {
                                item.number: layers for item in series.versions
                                if (layers := self._layers(item.sha256, item.version_id))
                                is not None
                            }
                            text += format_threads(
                                threads, facts_by_version, layers_by_version=layers_by_version,
                                as_of=self._version_number,
                            )
                return text

            def _review_json(self) -> dict:
                from yap_studio.homepage.link import for_path
                from yap_studio.install_info import workspace_agent

                facts, comments = self._current()
                reviewer = review.store.reviewer()
                previous = self._previous()
                series = self._series_json()
                records = [comment_to_dict(item) for item in comments]
                if series is not None:
                    if self._video.series.project_dir is not None:
                        series["questions"] = self._questions()
                        series["taste"] = self._taste_json()
                    rounds = review.store.rounds(self._video.series.id)
                    series["rounds"] = [round_to_dict(item) for item in
                                        sorted(rounds, key=lambda item: int(item.id[1:]),
                                               reverse=True)]
                    states = comment_states(comments, [item for item in rounds
                                                       if item.version == self._version_number])
                    records = [{**item, "send": states[item["id"]]} for item in records]
                earlier = None
                if previous is not None:
                    old_facts, old_comments = previous
                    earlier = {
                        "sha256": old_facts.sha256,
                        "count": len(old_comments),
                        "comments": [comment_to_dict(item) for item in old_comments],
                        "text": format_text(old_facts, old_comments, reviewer, date.today(),
                                            layers=self._layers(old_facts.sha256)),
                    }
                threads = (review.store.threads(self._video.series.id, self._version_number)
                           if series is not None else [])
                earlier_count = len(earlier_threads(threads, self._version_number))
                phone = None
                if review.phone:
                    origin = urlsplit(review.phone["url"])
                    path = urlsplit(self.path).path.removesuffix("api/review")
                    query = f"t={self._video.token}"
                    if self._version_number is not None:
                        query += f"&version={self._version_number}"
                    phone = {**review.phone, "url": origin._replace(
                        path=path, query=query, fragment="",
                    ).geturl()}
                return {
                    "agent": workspace_agent(self._video.series.project_dir)
                    if self._video.series is not None and self._video.series.project_dir is not None
                    else "claude-code",
                    "media_changed": self._selected._media_changed(),
                    "text_hook": self._text_hook(),
                    "phone": phone,
                    "home": for_path(self._video.series.project_dir
                                     if self._video.series is not None else None),
                    "media": {
                        "name": facts.name,
                        "sha256": facts.sha256,
                        "duration_s": facts.duration_s,
                        "fps": (
                            f"{facts.fps.numerator}/{facts.fps.denominator}"
                            if facts.fps is not None
                            else None
                        ),
                        "cfr": facts.cfr,
                        "codec": self._selected.media.codec,
                        "width": self._selected.media.width,
                        "height": self._selected.media.height,
                    },
                    "reviewer": reviewer,
                    "comments": records,
                    "text": self._text(facts, comments, reviewer, date.today()),
                    "text_earlier": (self._text(facts, comments, reviewer, date.today(),
                                                earlier=True) if threads else None),
                    "earlier_count": earlier_count,
                    "previous": earlier,
                    "series": series,
                    "findings": findings_for_media(
                        self._selected.media.path, self._selected.sha256,
                        project_dir=(self._video.series.project_dir
                                     if self._video.series is not None else None),
                        version_id=(self._selected.version_id
                                    if self._video.series is not None else None),
                    ),
                }

            def _text_hook(self) -> dict | None:
                series = self._video.series
                if series is None or series.project_dir is None:
                    return None
                target = series.project_dir / "plans/text-hook.json"
                if not target.is_file():
                    return None
                return parse_document("text-hook", json.loads(target.read_bytes())).to_dict()

            def _choose_text_hook(self, body: dict):
                series = self._video.series
                if (series is None or series.project_dir is None
                        or self._version_number != len(series.versions)):
                    raise ValidationError("text_hook",
                                          "Choose the text at the top on the newest review.")
                value = body.get("text_hook")
                if (set(body) != {"text_hook"} or not isinstance(value, dict)
                        or set(value) != {"text", "source"}
                        or not isinstance(value.get("source"), str)
                        or value.get("source") not in {"idea", "typed"}
                        or not isinstance(value.get("text"), str)):
                    raise ValidationError("text_hook", "Choose an idea or write your own text.")
                with project_lock(series.project_dir):
                    plan = self._text_hook()
                    if plan is None:
                        raise ValidationError("text_hook",
                                              "No ideas for text at the top are ready yet.")
                    plan.update(chosen=" ".join(value["text"].split()), source=value["source"])
                    chosen = parse_document("text-hook", plan)
                    assert isinstance(chosen, TextHook)
                    label = "Opening title" if chosen.kind == "opening" else "Text hook"
                    comment = review.store.add_comment(
                        self._selected.sha256, self._selected.media.facts(self._selected.sha256),
                        {"kind": "fix", "target": "element",
                         **({"start_frame": 0} if self._selected.media.cfr else {"start_s": 0}),
                         "text": f'{label}: "{chosen.chosen}"'},
                    )
                    paths.atomic_write_text(series.project_dir / "plans/text-hook.json",
                                            json.dumps(chosen.to_dict(), indent=2) + "\n")
                return comment

            def _taste_json(self) -> dict:
                series = self._video.series
                try:
                    taste = inbox_view(series.project_dir)
                except YapError as exc:
                    return {"available": False, "error": exc.to_dict(), "proposals": []}
                versions = {item.sha256: item.number for item in series.versions}
                for item in taste["proposals"]:
                    for source in item["sources"]:
                        source["version"] = versions.get(source.get("media_sha256"))
                return {"available": True, **taste}

            def _decide_taste(self) -> None:
                series = self._video.series
                if series is None or series.project_dir is None:
                    self._error(409, "no_project", "Open the project folder with --versions.")
                    return
                body = self._body()
                required = {"proposal", "action"}
                if not required <= body.keys() or body.keys() - required - {"sentence"}:
                    raise ValidationError("body", "must name a proposal and action")
                if not isinstance(body["proposal"], str):
                    raise ValidationError("proposal", "expected string")
                if (not isinstance(body["action"], str)
                        or body["action"] not in {"approve", "reject", "revoke"}):
                    raise ValidationError("action", "choose one of approve, reject, revoke")
                if "sentence" in body:
                    if body["action"] != "approve":
                        raise ValidationError("sentence", "only allowed with approve")
                    if not isinstance(body["sentence"], str):
                        raise ValidationError("sentence", "expected string")
                decide(series.project_dir, body["proposal"], body["action"],
                       sentence=body.get("sentence"))
                self._json(200, {"taste": self._taste_json()})

            def _questions(self) -> list[dict]:
                series = self._video.series
                if series is None or series.project_dir is None:
                    return []
                versions = {item.version_id: item.number for item in series.versions
                            if item.version_id is not None}
                questions = []
                for state, records in (
                    ("asked", pending_questions(series.project_dir)),
                    ("answered", answered_questions(series.project_dir)),
                ):
                    for item in records:
                        number = versions.get(item["comment"]["version"])
                        if number is not None:
                            questions.append({**item, "state": state,
                                              "answer": item.get("answer"),
                                              "option": item.get("option"),
                                              "comment_version": item["comment"]["version"],
                                              "comment": {"version": number,
                                                          "id": item["comment"]["id"]}})
                return questions

            def _removed(self) -> dict:
                series = self._video.series
                if series is None or series.project_dir is None:
                    return {"available": False, "ranges": []}
                manifest = load_version(series.project_dir, self._selected.version_id)
                return {"available": True, "version": self._version_number,
                        "ranges": removed_ranges(series.project_dir, manifest)}

            def _video_map(self) -> dict:
                series = self._video.series
                manifest = (load_version(series.project_dir, self._selected.version_id)
                            if series is not None and series.project_dir is not None else None)
                return {"version": self._version_number, "lanes": video_map(manifest)}

            def _words(self) -> dict:
                words = getattr(self._selected, "words", None)
                return {"available": words is not None, "version": self._version_number,
                        "words": [{key: getattr(word, key)
                                   for key in ("id", "segment", "text", "start", "end")}
                                  for word in words.words] if words is not None else []}

            def _captions(self) -> dict:
                series = self._video.series
                data = {"available": False, "phrases": []}
                if series is not None and series.project_dir is not None:
                    manifest = load_version(series.project_dir, self._selected.version_id)
                    data = caption_phrases(series.project_dir, manifest)
                return {**data, "version": self._version_number}

            def _caption_fix(self) -> None:
                series = self._video.series
                if series is None or series.project_dir is None:
                    self._error(409, "no_project", "Open the project folder with --versions.")
                    return
                body = self._body()
                root = series.project_dir
                target = root / "project.json"
                facts = self._selected.media.facts(self._selected.sha256)
                with project_lock(root):
                    digest = receipt.sha256_file(target)
                    data = json.loads(target.read_bytes())
                    captions = self._captions()
                    if not captions["available"]:
                        self._error(409, "captions_unavailable",
                                    "Caption words are unavailable for this version. "
                                    "Fix them with a note.")
                        return
                    if set(body) == {"undo"}:
                        notes = review.store.comments(self._selected.sha256)
                        note = next((note for note in notes if note.id == body["undo"]), None)
                        if note is None or (note.request or {}).get("type") != "caption_fix":
                            raise ValidationError("undo", "choose a caption fix note")
                        request = note.request
                        field = "word_fixes" if request["heard"] else "word_text_fixes"
                        key = request["heard"] or request["word"]
                        newer = any(
                            (datetime.fromisoformat(other.created_at)
                             > datetime.fromisoformat(note.created_at)
                             or (version.sha256 == self._selected.sha256
                                 and int(other.id[1:]) > int(note.id[1:])))
                            and (other.request or {}).get("type") == "caption_fix"
                            and other.request["heard"] == request["heard"]
                            and (request["heard"] or other.request["word"] == request["word"])
                            for version in series.versions
                            for other in review.store.comments(version.sha256)
                        )
                        if newer or data.get(field, {}).get(key) != request["to"]:
                            self._error(409, "fix_changed",
                                        "This word has another fix. Undo it first.")
                            return
                        original = json.loads(target.read_bytes())
                        change_word_fix(root, data,
                                        word=None if request["heard"] else request["word"],
                                        heard=request["heard"], to=request["previous"],
                                        remove=request["previous"] is None)
                        after = write_json_cas(target, data, base_sha256=digest)
                        try:
                            review.store.delete_comment(self._selected.sha256, note.id)
                        except Exception:
                            write_json_cas(target, original, base_sha256=after)
                            raise
                        self._json(200, {"undone": note.id})
                        return
                    if (not {"word", "start", "to"} <= body.keys()
                            or body.keys() - {"word", "start", "to", "every_time"}
                            or not isinstance(body["word"], str)
                            or not isinstance(body["to"], str)
                            or type(body["start"]) not in (int, float)
                            or not math.isfinite(body["start"]) or body["start"] < 0
                            or type(body.get("every_time", False)) is not bool):
                        raise ValidationError("body", "choose a word and replacement text")
                    word = min((word for phrase in captions["phrases"]
                                for word in phrase["words"] if word["id"] == body["word"]),
                               key=lambda word: abs(word["start"] - body["start"]), default=None)
                    rate = facts.fps or Fraction(
                        load_version(root, self._selected.version_id).media["fps"])
                    if word is None or abs(word["start"] - body["start"]) > 0.5 / float(rate):
                        raise ValidationError("word", "choose a word in this version's captions")
                    if not word["editable"]:
                        self._error(409, "caption_merged", word["reason"])
                        return
                    every_time = body.get("every_time", False)
                    if every_time and word["repeats"] < 2:
                        raise ValidationError("every_time", "this heard word does not repeat")
                    one_offs = data.get("word_text_fixes", {})
                    if every_time and any(
                        source.id in one_offs and _stripped(source.text) == _stripped(word["heard"])
                        for take in {identifier.split(":")[0] for identifier in one_offs}
                        for source in load_words(
                            paths.validate_input(root / f"ingest/{take}/words.json")).words
                    ):
                        self._error(409, "fix_changed",
                                    "This word has a one-off fix. Undo it first.")
                        return
                    original = json.loads(target.read_bytes())
                    field, key = change_word_fix(root, data,
                                                word=None if every_time else word["id"],
                                                heard=word["heard"] if every_time else None,
                                                to=body["to"])
                    request = {"type": "caption_fix", "word": word["id"],
                               "heard": key if every_time else None,
                               "from": word["text"], "to": body["to"].strip(),
                               "previous": original.get(field, {}).get(key)}
                    timing = ({"start_frame": frame_at(word["start"], facts.fps)} if facts.cfr
                              else {"start_s": word["start"]})
                    fields = {"kind": "fix", "target": "caption", "request": request, **timing,
                              "text": f'Caption: {word["text"]} -> {request["to"]} '
                                      f'at {format_time(word["start"])}'
                                      + (" (every time)" if every_time else "")}
                    normalize_input(fields, facts)
                    note = review.store.add_comment(self._selected.sha256, facts, fields)
                    try:
                        write_json_cas(target, data, base_sha256=digest)
                    except Exception:
                        review.store.delete_comment(self._selected.sha256, note.id)
                        raise
                self._json(201, comment_to_dict(note))

            def _cut(self) -> None:
                words = getattr(self._selected, "words", None)
                if words is None:
                    self._error(409, "no_words", "This version has no word list. Open the "
                                "project folder with --versions to cut by words.")
                    return
                facts = self._selected.media.facts(self._selected.sha256)
                body = self._body()
                edges = {"start_frame", "end_frame"} if facts.cfr else {"start_s", "end_s"}
                if not edges <= body.keys() or body.keys() - edges - {"text"}:
                    raise ValidationError("body", "must name a start and an end, "
                                          "with optional text")
                text = body.get("text", "")
                if not isinstance(text, str):
                    raise ValidationError("text", "must be a string")
                timing = {key: body[key] for key in edges}
                fields = {"kind": "fix", "target": "story", "text": text.strip() or "Cut this.",
                          **timing}
                normalized = normalize_input(fields, facts)
                if normalized["end_s"] is None:
                    raise ValidationError("range", "Cut this needs a range with a start and an end")
                start, end = (
                    (frame_start(normalized["start_frame"], facts.fps),
                     frame_start(normalized["end_frame"], facts.fps)) if facts.cfr else
                    (Fraction(str(normalized["start_s"])), Fraction(str(normalized["end_s"])))
                )
                covered = [{"id": word.id, "segment": word.segment, "text": word.text}
                           for word in words.words
                           if start <= (Fraction(str(word.start))
                                        + Fraction(str(word.end))) / 2 < end]
                if not covered:
                    raise ValidationError("range", "No spoken words in this range")
                if len(covered) > 200:
                    raise ValidationError("range", "Select at most 200 words to cut in one note")
                fields["request"] = {"type": "cut", "start": normalized["start_s"],
                                     "end": normalized["end_s"], "words": covered}
                comment = review.store.add_comment(self._selected.sha256, facts, fields)
                self._json(201, comment_to_dict(comment))

            def _restore(self) -> None:
                removed = self._removed()
                if not removed["available"]:
                    self._error(409, "no_project", "Open the project folder with --versions.")
                    return
                body = self._body()
                if set(body) != {"range"} or not isinstance(body["range"], str):
                    raise ValidationError("body", "must name a removed range")
                stretch = next((item for item in removed["ranges"]
                                if item["id"] == body["range"]), None)
                if stretch is None:
                    self._error(404, "not_found", "removed range not found")
                    return
                text = stretch["text"][:500] if stretch["text"] is not None else None
                request = {"type": "restore", "range": stretch["id"],
                           **{key: stretch[key] for key in ("take", "start", "end")},
                           "text": text}
                facts = self._selected.media.facts(self._selected.sha256)
                timing = ({"start_frame": stretch["at_frame"]} if facts.cfr else
                          {"start_s": stretch["at_s"]})
                comment = review.store.add_comment(self._selected.sha256, facts, {
                    "kind": "fix", "target": "story", "request": request, **timing,
                    "text": f'Bring back: "{text}"' if text is not None else
                    f'Bring back the removed part at {format_time(stretch["at_s"])}',
                })
                self._json(201, comment_to_dict(comment))

            def _answer_question(self) -> None:
                series = self._video.series
                if series is None or series.project_dir is None:
                    self._error(409, "no_project", "Open the project folder with --versions.")
                    return
                body = self._body()
                required = {"batch", "comment_version", "comment_id"}
                if not required <= body.keys() or body.keys() - required - {"answer", "option"}:
                    raise ValidationError("body", "must name a batch, comment, "
                                          "and answer or option")
                if (not isinstance(body["comment_version"], str)
                        or body["comment_version"] not in {item.version_id
                                                           for item in series.versions}
                        or not isinstance(body["comment_id"], str)):
                    raise ValidationError("comment", "must name a comment on this series")
                answer_clarification(series.project_dir, body["batch"], body["comment_version"],
                                     body["comment_id"], answer=body.get("answer"),
                                     option=body.get("option"))
                self._json(200, {"questions": self._questions()})

            def _export(self, path: str, query: dict[str, list[str]]) -> None:
                if path.endswith(".txt") and "earlier" in query and query["earlier"] != ["1"]:
                    raise ValidationError("earlier", "must be 1")
                if query.get("version") == ["previous"]:
                    selected = self._previous()
                    if selected is None:
                        self._error(404, "not_found", "previous review not found")
                        return
                    facts, comments = selected
                else:
                    facts, comments = self._current()
                reviewer = review.store.reviewer()
                extension = "txt" if path.endswith(".txt") else "json"
                if extension == "txt":
                    payload = self._text(facts, comments, reviewer, date.today(),
                                         earlier=query.get("earlier") == ["1"]).encode("utf-8")
                    kind = "text/plain; charset=utf-8"
                else:
                    value = to_export_json(facts, comments, reviewer, datetime.now(UTC))
                    series = self._video.series
                    value["series"] = (series_export(review.store, series, self._version_number)
                                       if series is not None else None)
                    payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode(
                        "utf-8"
                    )
                    kind = "application/json; charset=utf-8"
                filename = export_filename(facts.name, extension)
                safe_filename = _SAFE_FILENAME_CHARACTER.sub("_", filename)
                disposition = (
                    f'attachment; filename="{safe_filename}"; '
                    f"filename*=UTF-8''{quote(filename, safe='')}"
                )
                self._bytes(
                    200,
                    payload,
                    kind,
                    {"Content-Disposition": disposition},
                )

            def _release(self, path: str) -> None:
                series = self._video.series
                if self.command == "DELETE":
                    clear_approval(series.project_dir)
                    self._bytes(204, b"")
                    return
                body = self._body()
                if set(body) != {"version"} or type(body["version"]) is not int:
                    raise ValidationError("body", "must name a series version number")
                version = self._series_version([str(body["version"])], "version")
                action = rollback if path == "/api/rollback" else approve
                self._json(200, action(series.project_dir, version.version_id))

            def _final(self) -> None:
                from yap_studio.homepage.link import workspace_for_path

                series = self._video.series
                body = self._body()
                if set(body) != {"version"} or type(body["version"]) is not int:
                    raise ValidationError("body", "must name a series version number")
                version = self._series_version([str(body["version"])], "version")
                workspace = workspace_for_path(series.project_dir)
                if workspace is None:
                    raise home_bridge.ConflictError(
                        "Open Yap Studio home for this workspace before asking for the final video."
                    )
                relative = home_bridge.relative_project(workspace, series.project_dir)
                row = home_bridge.create_request(workspace, "final", {"project": relative},
                                                 version_id=version.version_id)
                self._json(200, {"request": row})

            def _download(self, path: str) -> None:
                series = self._video.series
                approval = approval_record(series.project_dir, verify_media=False)
                if approval is None or approval["version"] != self._selected.version_id:
                    self._error(403, "not_approved", "Approve this version to download it.")
                    return
                if self._selected._media_changed():
                    self._error(409, "media_changed", _MEDIA_CHANGED_MESSAGE)
                    return
                manifest = load_version(series.project_dir, self._selected.version_id)
                suffix = "-checks.json" if path.endswith("/report") else manifest.media_path.suffix
                filename = _SAFE_FILENAME_CHARACTER.sub(
                    "_", f"{manifest.project}-{manifest.id}{suffix}",
                )
                disposition = {"Content-Disposition": f'attachment; filename="{filename}"'}
                if path.endswith("/report"):
                    self._json(200, version_report(manifest), disposition)
                else:
                    self._media(disposition)

            def _media(self, attachment: dict[str, str] | None = None) -> None:
                try:
                    stream = self._selected.media.path.open("rb")
                except OSError:
                    self._error(409, "media_changed", _MEDIA_CHANGED_MESSAGE)
                    return
                with stream:
                    stat = os.fstat(stream.fileno())
                    if self._selected._media_changed(stream):
                        self._error(409, "media_changed", _MEDIA_CHANGED_MESSAGE)
                        return
                    size = stat.st_size
                    requested = self.headers.get("Range")
                    start, end = 0, size - 1
                    if requested is not None:
                        match = _RANGE.fullmatch(requested)
                        if match is None or (not match.group(1) and not match.group(2)):
                            self._range_error(size)
                            return
                        first, last = match.groups()
                        if first:
                            start = int(first)
                            end = min(int(last), size - 1) if last else size - 1
                        else:
                            count = int(last)
                            start = max(0, size - count)
                        if start >= size or end < start or (not first and int(last) == 0):
                            self._range_error(size)
                            return
                    length = end - start + 1
                    extra = {"Accept-Ranges": "bytes", "Cache-Control": "no-cache"}
                    extra.update(attachment or {})
                    if requested is not None:
                        extra["Content-Range"] = f"bytes {start}-{end}/{size}"
                    kind = _MEDIA_TYPES.get(
                        self._selected.media.path.suffix.lower(), "application/octet-stream"
                    )
                    self._headers(206 if requested is not None else 200, kind, length, extra)
                    if self.command == "HEAD":
                        return
                    try:
                        stream.seek(start)
                        remaining = length
                        while remaining:
                            chunk = stream.read(min(remaining, 256 * 1024))
                            if self._selected._media_changed(stream):
                                self.close_connection = True
                                return
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                    except OSError:
                        self.close_connection = True

            def _range_error(self, size: int) -> None:
                self._error(
                    416,
                    "range_not_satisfiable",
                    "range not satisfiable",
                    {"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes"},
                )

            def _series_version(self, values: list[str] | None, parameter: str) -> ReviewVersion:
                series = self._video.series
                if (series is None or values is None or len(values) != 1
                        or re.fullmatch(r"[1-9][0-9]*", values[0]) is None):
                    raise ValidationError(parameter, "must name a version in this series")
                try:
                    number = int(values[0])
                except ValueError as exc:
                    raise ValidationError(parameter, "must name a version in this series") from exc
                if number > len(series.versions):
                    raise ValidationError(parameter, "must name a version in this series")
                return series.versions[number - 1]

            def _select_version(self, path: str, query: dict[str, list[str]]) -> None:
                self._selected = self._video
                self._version_number = None
                values = query.get("version")
                if getattr(self._video, "series", None) is not None:
                    selected = self._series_version(
                        values if values is not None else [str(len(self._video.series.versions))],
                        "version",
                    )
                    self._selected = selected
                    self._version_number = selected.number
                elif values is not None and path not in {
                    "/api/removed", "/api/restore", "/api/words", "/api/cut",
                    "/api/captions", "/api/caption-fixes",
                } and not (
                    path in {"/api/export.txt", "/api/export.json"} and values == ["previous"]
                ):
                    self._error(404, "not_found", "version not found")
                    raise _Handled()

            def _map(self, query: dict[str, list[str]]) -> None:
                if self._video.series is None:
                    self._error(404, "not_found", "route not found")
                    return
                source = self._series_version(query.get("from"), "from")
                target = self._series_version(query.get("to"), "to")
                if source._media_changed() or target._media_changed():
                    self._error(409, "media_changed", _MEDIA_CHANGED_MESSAGE)
                    return
                if source.media.cfr and source.media.fps is not None:
                    values = query.get("frame", [])
                    if (len(values) != 1 or re.fullmatch(r"[0-9]+", values[0]) is None):
                        raise ValidationError("frame", "must be a non-negative integer")
                    try:
                        time_s = float(frame_start(int(values[0]), source.media.fps))
                    except (ValueError, OverflowError) as exc:
                        raise ValidationError(
                            "frame", "must be a finite non-negative integer"
                        ) from exc
                else:
                    try:
                        values = query.get("time", [])
                        if len(values) != 1:
                            raise ValueError("missing time")
                        time_s = float(values[0])
                        if not math.isfinite(time_s) or time_s < 0:
                            raise ValueError("invalid time")
                    except (ValueError, OverflowError) as exc:
                        raise ValidationError(
                            "time", "must be finite non-negative seconds"
                        ) from exc
                result = map_moment(source.words, target.words, time_s,
                                    target_duration=target.media.duration_s)
                self._json(200, {**result.to_dict(), "from": source.number, "to": target.number,
                                 "frame": frame_at(result.time_s, target.media.fps)
                                 if target.media.cfr and target.media.fps is not None else None})

            def _send_notes(self, round_id: str | None = None) -> None:
                series = self._video.series
                if series is None or series.project_dir is None:
                    self._error(409, "no_project",
                                "Open the project folder with --versions to send notes.")
                    return
                has_body = round_id is None or self.headers.get("Content-Length") is not None
                if has_body and self._body():
                    raise ValidationError("body", "must be an empty JSON object")
                if round_id is None:
                    comments = review.store.comments(self._selected.sha256)
                    rounds = [item for item in review.store.rounds(series.id)
                              if item.version == self._version_number]
                    states = comment_states(comments, rounds)
                    drafts = [item for item in comments if states[item.id]["state"] == "draft"]
                    answers = [{"batch": item["batch"], "comment": {
                        "version": item["comment_version"], "id": item["comment"]["id"],
                    }} for item in self._questions() if item["state"] == "answered"]
                    round = review.store.start_round(series.id, self._version_number, drafts,
                                                     answers=answers, owner=review.owner)
                else:
                    current = next((item for item in review.store.rounds(series.id)
                                    if item.id == round_id
                                    and item.version == self._version_number),
                                   None)
                    if current is None:
                        raise UnknownRoundError(round_id)
                    round = review.store.retry_round(series.id, round_id, owner=review.owner)
                version = self._selected
                review._send_runner(lambda: send_round(
                    review.store, series, version, round.id, instance=review.owner["instance"],
                ))
                self._json(202, round_to_dict(round))

            def _list_page(self) -> None:
                items: list[str] = []
                for number, video in enumerate(review.videos, 1):
                    duration = video.media.duration_s
                    length = "Unknown duration" if duration is None else f"{duration:.1f} s"
                    if review.phone:
                        length += f" · {link_label(review.phone['via'])}"
                    link = escape(f"v/{number}/?t={video.token}", quote=True)
                    items.append(
                        f'<li><a href="{link}">{escape(video.media.name)}</a>'
                        f'<span class="muted">{length}</span></li>'
                    )
                page = (_STATIC / "list.html").read_text(encoding="utf-8")
                payload = page.replace("<!-- videos -->", "\n".join(items)).encode("utf-8")
                self._bytes(200, payload, "text/html; charset=utf-8")

            def _route(self) -> None:
                self._video = review
                host = self.headers.get("Host")
                if not review._accepts_host(host):
                    self._error(403, "forbidden", "invalid Host header")
                    return
                parsed = urlsplit(self.path)
                path = parsed.path
                query = parse_qs(parsed.query, keep_blank_values=True)
                multiple = len(getattr(review, "videos", ())) > 1
                if multiple:
                    if path == "/":
                        if not self._authorized(query, review.list_token):
                            self._error(403, "forbidden", "invalid review token")
                        elif self.command != "GET":
                            self._error(
                                405, "method_not_allowed", "method not allowed", {"Allow": "GET"}
                            )
                        else:
                            self._list_page()
                        return
                    if not path.startswith("/static/"):
                        selected = _VIDEO_PATH.fullmatch(path)
                        if selected is None or int(selected.group(1)) > len(review.videos):
                            self._error(404, "not_found", "route not found")
                            return
                        self._video = review.videos[int(selected.group(1)) - 1]
                        path = selected.group(2)
                protected = path == "/media" or path.startswith("/api/") or (
                    multiple and path == "/"
                )
                if protected and not self._authorized(query):
                    self._error(403, "forbidden", "invalid review token")
                    return
                match = _COMMENT_PATH.fullmatch(path)
                retry = _RETRY_PATH.fullmatch(path)
                if path in {"/media", "/api/download", "/api/download/report"}:
                    allowed = {"GET", "HEAD"}
                elif path == "/api/approve":
                    allowed = {"POST", "DELETE"}
                elif path in {"/api/comments", "/api/rounds", "/api/questions/answer",
                              "/api/restore", "/api/cut", "/api/rollback", "/api/taste",
                              "/api/final", "/api/caption-fixes"} or retry:
                    allowed = {"POST"}
                elif match:
                    allowed = {"PATCH", "DELETE"}
                elif path in {
                    "/",
                    "/api/review",
                    "/api/export.txt",
                    "/api/export.json",
                    "/api/map",
                    "/api/removed",
                    "/api/words",
                    "/api/captions",
                    "/api/video-map",
                } or path.startswith("/static/"):
                    allowed = {"GET"}
                else:
                    self._error(404, "not_found", "route not found")
                    return
                if self.command not in allowed:
                    self._error(
                        405,
                        "method_not_allowed",
                        "method not allowed",
                        {"Allow": ", ".join(sorted(allowed))},
                    )
                    return
                if path in {"/api/rollback", "/api/approve", "/api/download",
                            "/api/download/report", "/api/final"} and (
                    self._video.series is None or self._video.series.project_dir is None
                ):
                    self._error(409, "no_project", "Open the project folder with --versions.")
                    return
                if path == "/media" or path.startswith("/api/"):
                    self._select_version(path, query)
                guarded = (path in {"/media", "/api/comments", "/api/rounds",
                                    "/api/questions/answer", "/api/restore", "/api/cut",
                                    "/api/captions", "/api/caption-fixes"}
                           or match is not None or retry is not None)
                if guarded and self._selected._media_changed():
                    self._error(409, "media_changed", _MEDIA_CHANGED_MESSAGE)
                    return
                if path == "/media":
                    self._media()
                elif path == "/api/review":
                    self._json(200, self._review_json())
                elif path in {"/api/rollback", "/api/approve"}:
                    self._release(path)
                elif path == "/api/final":
                    self._final()
                elif path in {"/api/download", "/api/download/report"}:
                    self._download(path)
                elif path == "/api/map":
                    self._map(query)
                elif path == "/api/removed":
                    self._json(200, self._removed())
                elif path == "/api/restore":
                    self._restore()
                elif path == "/api/words":
                    self._json(200, self._words())
                elif path == "/api/captions":
                    self._json(200, self._captions())
                elif path == "/api/caption-fixes":
                    self._caption_fix()
                elif path == "/api/video-map":
                    self._json(200, self._video_map())
                elif path == "/api/cut":
                    self._cut()
                elif path in {"/api/export.txt", "/api/export.json"}:
                    self._export(path, query)
                elif path == "/api/rounds" or retry:
                    self._send_notes(retry.group(1) if retry else None)
                elif path == "/api/questions/answer":
                    self._answer_question()
                elif path == "/api/taste":
                    self._decide_taste()
                elif path == "/api/comments":
                    body = self._body()
                    if isinstance(body.get("request"), dict) and (
                        body["request"].get("type") == "caption_fix"
                    ):
                        raise ValidationError("request", "use the caption panel to save fixes")
                    if "text_hook" in body:
                        comment = self._choose_text_hook(body)
                    else:
                        comment = review.store.add_comment(
                            self._selected.sha256,
                            self._selected.media.facts(self._selected.sha256), body,
                        )
                    self._json(201, comment_to_dict(comment))
                elif match:
                    comment_id = match.group(1)
                    note = next((note for note in review.store.comments(self._selected.sha256)
                                 if note.id == comment_id), None)
                    if note is not None and (note.request or {}).get("type") == "caption_fix":
                        raise ValidationError("request", "use Undo caption fix to remove this fix")
                    if self.command == "PATCH":
                        body = self._body()
                        if isinstance(body.get("request"), dict) and (
                            body["request"].get("type") == "caption_fix"
                        ):
                            raise ValidationError("request", "use the caption panel to save fixes")
                        comment = review.store.update_comment(
                            self._selected.sha256,
                            self._selected.media.facts(self._selected.sha256),
                            comment_id,
                            body,
                        )
                        self._json(200, comment_to_dict(comment))
                    else:
                        review.store.delete_comment(self._selected.sha256, comment_id)
                        self._bytes(204, b"")
                else:
                    name = "index.html" if path == "/" else path.removeprefix("/static/")
                    if path != "/" and _STATIC_NAME.fullmatch(name) is None:
                        self._error(404, "not_found", "file not found")
                        return
                    target = _STATIC / name
                    if not target.is_file() or target.resolve().parent != _STATIC.resolve():
                        self._error(404, "not_found", "file not found")
                        return
                    content_type = (
                        "text/html; charset=utf-8" if path == "/" else _STATIC_TYPES[target.suffix]
                    )
                    self._bytes(200, target.read_bytes(), content_type)

            def _dispatch(self) -> None:
                try:
                    try:
                        self._route()
                    except _Handled:
                        pass
                    except UnknownCommentError as exc:
                        self._json(404, {"error": exc.to_dict()})
                    except UnknownRoundError as exc:
                        self._json(404, {"error": exc.to_dict()})
                    except UnknownQuestionError as exc:
                        self._json(404, {"error": exc.to_dict()})
                    except UnknownProposalError as exc:
                        self._json(404, {"error": exc.to_dict()})
                    except TasteRefusal as exc:
                        self._json(409, {"error": exc.to_dict()})
                    except CarriedQuestionError as exc:
                        self._json(409, {"error": exc.to_dict()})
                    except RoundConflictError as exc:
                        self._json(409, {"error": exc.to_dict()})
                    except home_bridge.ConflictError as exc:
                        self._json(409, {"error": exc.to_dict()})
                    except VersionError as exc:
                        self._json(409, {"error": exc.to_dict()})
                    except ValidationError as exc:
                        self._json(400, {"error": exc.to_dict()})
                    except YapError as exc:
                        self._json(500, {"error": exc.to_dict()})
                    except Exception:
                        self._error(500, "internal_error", "internal server error")
                except OSError:
                    pass

            def do_GET(self) -> None:
                self._dispatch()

            def do_POST(self) -> None:
                self._dispatch()

            def do_PATCH(self) -> None:
                self._dispatch()

            def do_DELETE(self) -> None:
                self._dispatch()

            def do_HEAD(self) -> None:
                self._dispatch()

            def __getattr__(self, name: str):
                if name.startswith("do_"):
                    return self._dispatch
                raise AttributeError(name)

        return Handler


class _Handled(Exception):
    """Mark a request whose response has already been sent."""
