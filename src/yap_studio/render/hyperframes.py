"""Install pinned dependencies and render exact-length layers with local assets."""

from __future__ import annotations

import codecs
import errno
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote
from uuid import uuid4

from yap_studio import paths, tools
from yap_studio.cache import cache_root
from yap_studio.errors import (
    DependencyError,
    ProcessingError,
    RendererBlockedError,
    ValidationError,
    YapError,
)
from yap_studio.render import alpha

HYPERFRAMES_VERSION = "0.8.184"
# From 0.8.151 caption bands render the same at any worker count.
# Let HyperFrames choose the caption count.
# Cards and titles stay at one worker. A stat card chunk that starts during
# its scale pulse keeps the number about a pixel off.
# Screenshot capture keeps text and colours consistent across machines.
RENDER_WORKERS = 1
CAPTION_WORKERS = "auto"
RENDER_CAPTURE = "screenshot"
# Under load the GPU can draw clip edges and rounded corners differently between renders.
# Software drawing gives cards the same pixels every time.
CARD_BROWSER_GPU = False
# Captions draw in software too, so the same layer gives the same pixels under load.
CAPTION_BROWSER_GPU = False
# Renderer serves 127.0.0.1 but loads localhost; another program on [::1] can break page loads.
LOCAL_SERVER_ERRORS = (
    "net::ERR_INVALID_HTTP_RESPONSE at http://localhost:",
    "net::ERR_CONNECTION_RESET at http://localhost:",
    "net::ERR_EMPTY_RESPONSE at http://localhost:",
)
GSAP_VERSION = "3.14.2"
GSAP_SHA256 = "c174bfce53a729418d57a8ad8625e7247c793a22fef8e2851e3cfa3de9cd8280"
GSAP_SCRIPT = "gsap.min.js"
PINNED_GSAP_PLUGINS = {
    "CustomEase.min.js": "2dae8ef6daba66ce7b527b1f46a5695ba325b21ad5b4d73e4a843753c65415f6",
}
LOCK_DIR = Path(__file__).parent / "renderer-lock"
RENDERER_ENV = {
    "HYPERFRAMES_NO_TELEMETRY": "1",
    "HYPERFRAMES_NO_UPDATE_CHECK": "1",
    "HYPERFRAMES_NO_AUTO_INSTALL": "1",
    "HYPERFRAMES_SKIP_SKILLS": "1",
    # CI stops HyperFrames' warm check (#5943) background process and home writes on non-TTY checks.
    "CI": "1",
}


def renderer_env() -> dict[str, str]:
    """Keep renderer scripts and their children on the working Node."""
    return tools.node_env(RENDERER_ENV)


@dataclass(frozen=True)
class RendererInstall:
    """A completed renderer installation and its pinned browser."""

    version: str
    prefix: Path
    executable: Path
    browser_path: str | None
    node_version: str


@dataclass(frozen=True)
class LayerResult:
    """A converted alpha layer at its composition's own box size."""

    path: Path
    frames: int
    width: int
    height: int
    duration_s: float
    checks: dict | None = None
    render_retries: int = 0


def install_prefix() -> Path:
    """Return the versioned cache prefix, refusing paths that enable self-update."""
    prefix = cache_root() / "renderer" / f"hyperframes-{HYPERFRAMES_VERSION}"
    executable = prefix / "node_modules" / "hyperframes" / "bin" / "hyperframes.mjs"
    if any(
        fragment in executable.resolve().as_posix()
        for fragment in ("/npm/node_modules/hyperframes/", "/lib/node_modules/hyperframes/")
    ):
        raise DependencyError(
            "Renderer cache path would enable self-update.",
            error_type="dependency_error",
            code="renderer_cache_path",
            suggested_action="Set YAP_STUDIO_CACHE to a dedicated cache folder and try again.",
        )
    return prefix


def _lock_sha256() -> str:
    digest = hashlib.sha256()
    digest.update((LOCK_DIR / "package.json").read_bytes())
    digest.update(b"\0")
    digest.update((LOCK_DIR / "package-lock.json").read_bytes())
    return digest.hexdigest()


def _install_metadata(prefix: Path) -> dict | None:
    """Read valid metadata independently of package and browser availability."""
    try:
        metadata = json.loads((prefix / "install.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    if (
        not isinstance(metadata, dict)
        or metadata.get("version") != HYPERFRAMES_VERSION
        or metadata.get("lock_sha256") != _lock_sha256()
        or not isinstance(metadata.get("node_version"), str)
        or not isinstance(metadata.get("browser_path"), (str, type(None)))
    ):
        return None
    return metadata


def _npm_installed() -> RendererInstall | None:
    """Return intact pinned packages even when their browser needs repair."""
    prefix = install_prefix()
    executable = prefix / "node_modules" / ".bin" / "hyperframes"
    metadata = _install_metadata(prefix)
    if metadata is None or not executable.is_file() or not os.access(executable, os.X_OK):
        return None
    return RendererInstall(
        HYPERFRAMES_VERSION,
        prefix,
        executable,
        metadata.get("browser_path"),
        metadata["node_version"],
    )


def installed() -> RendererInstall | None:
    """Return a completed installation with an available browser, or None."""
    result = _npm_installed()
    if result is None or result.browser_path is None or not Path(result.browser_path).exists():
        return None
    return result


def gsap_file(install: RendererInstall) -> Path:
    """Return the installed GSAP script only when its pinned checksum matches."""
    script = install.prefix / "node_modules" / "gsap" / "dist" / GSAP_SCRIPT
    try:
        if script.is_file() and hashlib.sha256(script.read_bytes()).hexdigest() == GSAP_SHA256:
            return script
    except OSError:
        pass
    raise DependencyError(
        f"Pinned GSAP {GSAP_VERSION} is missing or altered in renderer cache {install.prefix}.",
        error_type="dependency_error",
        code="renderer_gsap_missing",
        suggested_action=(
            f"Delete the renderer cache folder {install.prefix} and run the command again "
            "so the pinned renderer reinstalls."
        ),
    )


@contextmanager
def _install_lock(timeout_s: float, *, directory: Path | None = None) -> Iterator[None]:
    """Serialize the full installation across processes and threads."""
    directory = paths.validate_output_dir(directory if directory is not None
                                          else cache_root() / "renderer")
    descriptor = os.open(directory / ".install.lock", os.O_RDWR | os.O_CREAT, 0o600)
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            deadline = time.monotonic() + timeout_s
            while True:
                try:
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
                    break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise DependencyError(
                            "Another process holds the renderer installation lock.",
                            error_type="dependency_error",
                            code="renderer_install_locked",
                            suggested_action="Wait for the installation to finish and retry.",
                        ) from exc
                    time.sleep(0.05)
        else:
            import fcntl

            deadline = time.monotonic() + timeout_s
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                        raise
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise DependencyError(
                            "Another process holds the renderer installation lock.",
                            error_type="dependency_error",
                            code="renderer_install_locked",
                            suggested_action="Wait for the installation to finish and retry.",
                        ) from exc
                    time.sleep(min(0.05, remaining))
        locked = True
        yield
    finally:
        try:
            if locked:
                if os.name == "nt":
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _renderer_blocked(stderr: str) -> bool:
    lowered = stderr.lower()
    return (
        "failed to launch the browser process" in lowered
        or "uv_os_setpriority returned eperm" in lowered
        or (
            any(text in lowered for text in ("operation not permitted", "sandbox", "eperm"))
            and any(text in lowered for text in ("chrome", "chromium", "browser", "mach"))
        )
    )


@contextmanager
def _renderer_env() -> Iterator[dict[str, str]]:
    # HyperFrames 0.8.152 and later checkpoint project history on every render and check.
    # Our layers are disposable, so their history is too.
    with tempfile.TemporaryDirectory(
        prefix="hyperframes-history-", ignore_cleanup_errors=True
    ) as history_root:
        yield {**renderer_env(), "HYPERFRAMES_HISTORY_ROOT": history_root}


def _run_renderer(executable: Path, args: list[str], *, timeout_s: float) -> str:
    try:
        with _renderer_env() as env:
            result = tools.run_tool(
                "hyperframes", args, timeout_s=timeout_s, env=env, executable=executable
            )
    except ProcessingError as exc:
        if _renderer_blocked(exc.full_stderr):
            raise RendererBlockedError() from exc
        if "variable validation failed" in exc.full_stderr.lower():
            raise YapError(
                "Renderer variable validation failed.",
                error_type="validation_error",
                code="renderer_variables",
                suggested_action="Declare each variable on the composition's html element.",
            ) from exc
        raise
    return result.stdout


def render_with_retry(attempt: Callable[[], object]) -> dict | None:
    """Retry one local server collision and return its detail after success."""
    try:
        attempt()
    except ProcessingError as exc:
        detail = next((
            line.strip()[:200] for line in exc.full_stderr.splitlines()
            if any(marker in line for marker in LOCAL_SERVER_ERRORS)
        ), None)
        if detail is None:
            raise
        attempt()
        return {"reason": "renderer_local_server", "detail": detail}
    return None


def ensure_installed(
    *, notice: Callable[[str], None] | None = None, timeout_s: float = 900
) -> RendererInstall:
    """Install locked packages and the browser in staging, then publish the cache."""
    existing = installed()
    if existing is not None:
        return existing
    with paths.operation(), _install_lock(timeout_s):
        existing = installed()
        if existing is not None:
            return existing
        return _install_locked(notice=notice, timeout_s=timeout_s)


def _ensure_browser(prefix: Path, executable: Path, node_version: str, timeout_s: float) -> str:
    """Ensure the browser and record the installation's current browser path."""
    stdout = _run_renderer(executable, ["browser", "ensure"], timeout_s=timeout_s)
    browser = re.search(r"^[ \t]*Path:[ \t]*(\S[^\r\n]*?)[ \t]*\r?$", stdout, re.MULTILINE)
    if browser is None:
        raise DependencyError(
            "Renderer did not report its browser path.",
            error_type="dependency_error",
            code="renderer_browser_missing",
            suggested_action="Check the Chrome download and retry installation.",
        )
    metadata = {
        "schema": "yap-studio/renderer-install@1",
        "version": HYPERFRAMES_VERSION,
        "browser_path": browser[1],
        "node_version": node_version,
        "lock_sha256": _lock_sha256(),
    }
    paths.atomic_write_text(prefix / "install.json", json.dumps(metadata) + "\n")
    return browser[1]


def _install_locked(*, notice: Callable[[str], None] | None, timeout_s: float) -> RendererInstall:
    """Repair the browser or install packages while holding the cache lock."""
    packages = _npm_installed()
    if packages is not None:
        browser_path = _ensure_browser(
            packages.prefix, packages.executable, packages.node_version, timeout_s
        )
        return RendererInstall(
            packages.version, packages.prefix, packages.executable,
            browser_path, packages.node_version,
        )
    prefix = install_prefix()
    if _install_metadata(prefix) is not None:
        if prefix.is_symlink() or not prefix.is_dir():
            prefix.unlink()
        else:
            shutil.rmtree(prefix)
    node_version = tools.node_version() or ""
    match = re.fullmatch(r"v?(\d+)\.\d+\.\d+(?:[-+].*)?", node_version)
    if match is None or int(match[1]) < tools.MIN_NODE_MAJOR or tools.find_tool("npm") is None:
        raise DependencyError.missing_node()
    with paths.operation():
        paths.validate_output_dir(prefix.parent)
        temporary = Path(tempfile.mkdtemp(dir=prefix.parent, prefix=".hyperframes-"))
        try:
            for name in ("package.json", "package-lock.json"):
                with paths.atomic_output(temporary / name) as staged:
                    shutil.copyfile(LOCK_DIR / name, staged)
            if notice is not None:
                notice("First install downloads the pinned renderer and its Chrome browser.")
            tools.run_tool(
                "npm",
                ["ci", "--no-audit", "--no-fund", "--ignore-scripts"],
                timeout_s=timeout_s,
                env=renderer_env(),
                cwd=temporary,
            )
            executable = temporary / "node_modules" / ".bin" / "hyperframes"
            browser_path = _ensure_browser(temporary, executable, node_version, timeout_s)
            if prefix.exists():
                if prefix.is_symlink() or not prefix.is_dir():
                    prefix.unlink()
                else:
                    shutil.rmtree(prefix)
            os.replace(temporary, prefix)
            return RendererInstall(
                HYPERFRAMES_VERSION,
                prefix,
                prefix / "node_modules" / ".bin" / "hyperframes",
                browser_path,
                node_version,
            )
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def layer_duration_s(frames: int, fps: int = 30) -> float:
    """Choose a millisecond duration inside the window for exactly frames frames."""
    if frames < 1 or fps < 1:
        raise ValueError("frames and fps must be positive")
    return math.floor((frames * 1000 - 1) / fps) / 1000


def renderer_frame_count(duration_s: float, fps: int = 30) -> int:
    """Calculate the pinned renderer's frame count for a duration."""
    if not math.isfinite(duration_s) or duration_s < 0 or fps < 1:
        raise ValueError("duration must be finite and nonnegative; fps must be positive")
    # Mirrors HyperFrames' totalFrames = Math.max(1, Math.ceil(duration * fps)).
    return max(1, math.ceil(duration_s * fps))


_CSS_STRING = r""""(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'"""
_HOST_CHARACTER = r"[a-zA-Z0-9_\[\u0080-\uffff]"
_EXTERNAL_URL = re.compile(
    r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s\"'`<>(){};,]*"
    rf"|(?<![^\s=(,;\"'`])//{_HOST_CHARACTER}[^\s\"'`<>(){{}};,]*"
)
_CSS_COMMENT_OR_STRING = re.compile(rf"/\*.*?(?:\*/|$)|{_CSS_STRING}", re.DOTALL)
_CSS_ASSET = re.compile(
    rf"\burl\s*\(\s*(?P<url>{_CSS_STRING}|(?:\\.|[^)\\])*)\s*\)"
    rf"|@import\s+(?P<import>{_CSS_STRING})"
    r"|(?P<image_set>\b(?:-webkit-)?image-set\s*\()"
    rf"|(?P<string>{_CSS_STRING})"
    r"|(?P<open>\()|(?P<close>\))"
    rf"|(?P<literal>{_EXTERNAL_URL.pattern})",
    re.IGNORECASE | re.DOTALL,
)
_CSS_ESCAPE = re.compile(r"\\(?:([0-9a-fA-F]{1,6})\s?|(\r\n|[\n\r\f])|(.))", re.DOTALL)
_HTML_ATTRIBUTE = re.compile(
    r"""([^\s/>=]+)(?:\s*=\s*(?:"(?P<double>[^"]*)"|'(?P<single>[^']*)'"""
    r"|(?P<bare>[^\s>]*)))?"
)
_URL_ATTRIBUTES = frozenset({
    "src", "href", "xlink:href", "data", "poster", "background", "action", "formaction",
    "cite", "longdesc", "usemap", "manifest", "codebase", "lowsrc", "dynsrc", "icon",
    "profile", "data-composition-src",
})
_CSS_ATTRIBUTES = frozenset({
    "style", "fill", "stroke", "filter", "clip-path", "mask", "marker-start", "marker-mid",
    "marker-end", "cursor",
})
_NETWORK_SCRIPT = re.compile(
    rf"(?:https?|wss?|ftp)://[^\s\"'`<>]*"
    rf"|[\"'`]//{_HOST_CHARACTER}[^\s\"'`<>]*"
    r"|\b(?:fetch\s*\(|XMLHttpRequest\b|WebSocket\b|EventSource\b|sendBeacon\b"
    r"|importScripts\b|new\s+(?:Worker|SharedWorker)\s*\(|RTCPeerConnection\b"
    r"|serviceWorker\b|import\s*\()",
    re.IGNORECASE,
)
_XML_NAMESPACES = frozenset({
    "http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink",
    "http://www.w3.org/1999/xhtml", "http://www.w3.org/XML/1998/namespace",
    "http://www.w3.org/2000/xmlns/",
})
_SCRIPT_STRING = re.compile(r'''"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`''')
_NAMESPACE_CALL = re.compile(r"\b[A-Za-z_$][\w$]*NS\s*\(\s*$")
_NAMESPACE_DECLARATION = re.compile(r"\b(?:var|let|const)\s+([A-Za-z_$][\w$]*)\s*=\s*$")


def _css_unescape(value: str) -> str:
    def unescape(match: re.Match[str]) -> str:
        if match[1] is not None:
            point = int(match[1], 16)
            return chr(point) if 0 < point <= 0x10FFFF else "\ufffd"
        return "" if match[2] is not None else match[3]

    return _CSS_ESCAPE.sub(unescape, value)


def _css_urls(source: str, *, literals: bool = True) -> Iterator[tuple[str, int]]:
    """Yield CSS references and other URL literals with their source offsets."""
    # Preserve offsets and lines while ignoring comments, including unclosed comments.
    source = _CSS_COMMENT_OR_STRING.sub(
        lambda match: re.sub(r"[^\n]", " ", match[0])
        if match[0].startswith("/*") else match[0],
        source,
    )
    parentheses: list[bool] = []
    for match in _CSS_ASSET.finditer(source):
        group = match.lastgroup
        if group == "image_set":
            parentheses.append(True)
        elif group == "open":
            parentheses.append(False)
        elif group == "close":
            if parentheses:
                parentheses.pop()
        elif group in {"url", "import", "string"}:
            raw = match[group]
            offset = match.start(group) + len(raw) - len(raw.lstrip())
            value = raw.strip()
            if value.startswith(("'", '"')):
                value = value[1:-1]
            value = _css_unescape(value)
            if group != "string" or any(parentheses):
                yield value, offset
            elif literals:
                for literal in _EXTERNAL_URL.finditer(value):
                    yield literal[0], offset + 1 + literal.start()
            else:
                # Attribute values can wrap CSS functions in additional quoted strings.
                for item, nested_offset in _css_urls(value, literals=False):
                    yield item, offset + 1 + nested_offset
        elif group == "literal" and literals:
            yield match[0], match.start()


def _srcset_urls(source: str) -> Iterator[tuple[str, int]]:
    # URLs may contain commas, notably data URIs; descriptors end at the next comma.
    position = 0
    while position < len(source):
        while position < len(source) and (source[position].isspace() or source[position] == ","):
            position += 1
        start = position
        while position < len(source) and not source[position].isspace():
            position += 1
        value = source[start:position]
        if not value:
            break
        yield value.rstrip(","), start
        if value.endswith(","):
            continue
        parentheses = 0
        while position < len(source):
            character = source[position]
            position += 1
            if character == "," and parentheses == 0:
                break
            if character == "(":
                parentheses += 1
            elif character == ")":
                parentheses = max(0, parentheses - 1)


def _source_label(composition_dir: Path, source_file: Path, line: int) -> str:
    return f"{source_file.relative_to(composition_dir).as_posix()}:{line}"


def _external_asset(
    value: str, composition_dir: Path, source_file: Path, line: int,
    *, reason: str = "Layer asset must be local",
) -> YapError:
    return YapError(
        f"{_source_label(composition_dir, source_file, line)}: {reason}: {value[:120]!r}.",
        error_type="validation_error",
        code="layer_external_asset",
        suggested_action=(
            "Copy assets into the composition folder and reference them by a relative path. "
            "Base elements and srcdoc attributes are not supported in layer compositions."
        ),
    )


def _validate_asset(
    value: str, composition_dir: Path, source_file: Path, line: int,
) -> None:
    line += value.count("\n", 0, len(value) - len(value.lstrip()))
    normalized = re.sub(r"[\t\r\n]", "", value.strip())
    if normalized.lower().startswith("data:") or normalized.startswith("#"):
        return
    asset_path = unquote(normalized.split("#", 1)[0].split("?", 1)[0])
    if (
        re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", normalized)
        or asset_path.startswith("/")
        or "\\" in normalized
        or "\\" in asset_path
        or "\0" in asset_path
        or ".." in asset_path.split("/")
    ):
        raise _external_asset(value, composition_dir, source_file, line)
    try:
        resolved = (source_file.parent / asset_path).resolve()
        local = resolved.is_relative_to(composition_dir.resolve())
    except (OSError, RuntimeError, ValueError):
        local = False
    if not local:
        raise _external_asset(value, composition_dir, source_file, line)


def _validate_css(
    source: str, composition_dir: Path, source_file: Path, line: int = 1,
    *, literals: bool = True,
) -> None:
    for value, offset in _css_urls(source, literals=literals):
        _validate_asset(value, composition_dir, source_file, line + source.count("\n", 0, offset))


def _namespace_argument(source: str, start: int, end: int) -> bool:
    return (_NAMESPACE_CALL.search(source[:start]) is not None
            and re.match(r"\s*[,)]", source[end:]) is not None)


def _namespace_literal(source: str, match: re.Match[str]) -> bool:
    """Allow namespaces only as DOM namespace arguments or restricted declarations."""
    if match[0][0] not in {"'", '"'} or match[0][1:-1] not in _XML_NAMESPACES:
        return False
    if _namespace_argument(source, match.start(), match.end()):
        return True
    declaration = _NAMESPACE_DECLARATION.search(source[:match.start()])
    tail = source[match.end():]
    if (declaration is None or re.match(r"[^\S\r\n]*(?:[;,\r\n]|$)", tail) is None
            or re.match(r"\s*[-+*/%&|^?:.\[(`]", tail) is not None):
        return False
    identifier = re.escape(declaration[1])
    for use in re.finditer(rf"(?<![\w$]){identifier}(?![\w$])", source):
        if use.span() == declaration.span(1) or match.start() <= use.start() < match.end():
            continue
        if not _namespace_argument(source, use.start(), use.end()):
            return False
    return True


def _validate_script(source: str, composition_dir: Path, source_file: Path, line: int = 1) -> None:
    # Namespace arguments are data, while other URL literals and network calls remain blocked.
    def namespace_literal(match: re.Match[str]) -> str:
        if _namespace_literal(source, match):
            return " " * len(match[0])
        return match[0]

    scanned = _SCRIPT_STRING.sub(namespace_literal, source)
    match = _NETWORK_SCRIPT.search(scanned)
    if match is not None:
        line += source.count("\n", 0, match.start())
        raise YapError(
            f"{_source_label(composition_dir, source_file, line)}: "
            f"Layer network script pattern: {match[0][:120]!r}.",
            error_type="validation_error",
            code="layer_network_script",
            suggested_action=(
                "A layer composition must not make network requests. "
                "Assets belong in the composition folder."
            ),
        )


class _CompositionMetadataParser(HTMLParser):
    """Collect root metadata before deciding whether GSAP ownership applies."""

    def __init__(self) -> None:
        super().__init__()
        self.roots: list[dict[str, str | None]] = []
        self.uses_gsap = False
        self._elements: list[tuple[str, bool]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        composition = "data-composition-id" in attributes
        if composition and not any(root for _, root in self._elements):
            self.roots.append(attributes)
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
                       "meta", "param", "source", "track", "wbr"}:
            self._elements.append((tag, composition))
        if tag == "script" and attributes.get("src") == GSAP_SCRIPT:
            self.uses_gsap = True

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self._elements) - 1, -1, -1):
            if self._elements[index][0] == tag:
                del self._elements[index:]
                break


class _CompositionParser(_CompositionMetadataParser):
    """Validate each HTML or SVG reference while collecting composition metadata."""

    def __init__(self, composition_dir: Path, source_file: Path) -> None:
        super().__init__()
        self.composition_dir = composition_dir
        self.source_file = source_file
        self._text_tag: str | None = None
        self._text: list[str] = []
        self._text_line = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        super().handle_starttag(tag, attrs)
        attributes = dict(attrs)
        line = self.getpos()[0]
        if tag == "base":
            raise _external_asset(
                attributes.get("href") or "<base>", self.composition_dir, self.source_file, line,
                reason="Base elements are not supported in layer compositions",
            )
        raw = self.get_starttag_text() or ""
        tag_end = re.match(r"<\s*[^\s/>]+", raw)
        positions = list(_HTML_ATTRIBUTE.finditer(raw, tag_end.end() if tag_end else 0))
        for index, (name, value) in enumerate(attrs):
            position = positions[index] if index < len(positions) else None
            offset = position.start(position.lastgroup or 1) if position is not None else 0
            attribute_line = line + raw.count(
                "\n", 0, offset
            )
            if name == "srcdoc":
                raise _external_asset(
                    value or "srcdoc", self.composition_dir, self.source_file, attribute_line,
                    reason="Srcdoc attributes are not supported in layer compositions",
                )
            if name == "xmlns" or name.startswith("xmlns:"):
                continue
            if value is None:
                continue
            if name.startswith("on"):
                _validate_script(value, self.composition_dir, self.source_file, attribute_line)
            if name in _URL_ATTRIBUTES:
                _validate_asset(value, self.composition_dir, self.source_file, attribute_line)
            elif name in {"archive", "ping"}:
                for item in re.finditer(r"\S+", value):
                    _validate_asset(
                        item[0], self.composition_dir, self.source_file,
                        attribute_line + value.count("\n", 0, item.start()),
                    )
            elif name in {"srcset", "imagesrcset"}:
                for item, offset in _srcset_urls(value):
                    _validate_asset(
                        item, self.composition_dir, self.source_file,
                        attribute_line + value.count("\n", 0, offset),
                    )
            elif name != "style":
                literal = _EXTERNAL_URL.search(value)
                if literal is not None and value not in _XML_NAMESPACES:
                    raise _external_asset(
                        value, self.composition_dir, self.source_file,
                        attribute_line + value.count("\n", 0, literal.start()),
                    )
            if name in _CSS_ATTRIBUTES:
                _validate_css(
                    value, self.composition_dir, self.source_file, attribute_line,
                    literals=name == "style",
                )
        if tag == "style" or (tag == "script" and "src" not in attributes):
            self._text_tag = tag
            self._text = []
            self._text_line = line + raw.count("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        super().handle_endtag(tag)
        if tag == self._text_tag:
            validate = _validate_css if tag == "style" else _validate_script
            validate("".join(self._text), self.composition_dir, self.source_file, self._text_line)
            self._text_tag = None

    def handle_data(self, data: str) -> None:
        if self._text_tag is not None:
            self._text.append(data)

    def close(self) -> None:
        super().close()
        if self._text_tag is not None:
            # Older interpreter patch releases leave unterminated CDATA text unflushed at EOF.
            if self.rawdata:
                self._text.append(self.rawdata)
                self.rawdata = ""
            self.handle_endtag(self._text_tag)


def _bom_encoding(data: bytes) -> str | None:
    if data.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if data.startswith((codecs.BOM_UTF16_BE, codecs.BOM_UTF16_LE)):
        return "utf-16"
    return None


def _css_encoding(data: bytes) -> str:
    encoding = _bom_encoding(data)
    if encoding is not None:
        return encoding
    charset = re.match(br'@charset "([\x00-\x21\x23-\x7f]+)";', data)
    if charset is not None:
        label = charset[1].decode("ascii").strip().lower()
        if label in {"utf-16be", "utf-16le", "utf-16"}:
            return "utf-8"
        if label in {"iso-8859-1", "latin1", "l1", "ascii", "us-ascii", "windows-1252"}:
            return "cp1252"
        try:
            return codecs.lookup(label).name
        except LookupError:
            pass
    return "utf-8"


def _decode_css(data: bytes) -> str:
    """Decode a stylesheet strictly using its BOM or exact charset declaration."""
    return data.decode(_css_encoding(data), errors="strict")


def _read_source(composition_dir: Path, source_file: Path) -> str:
    data = source_file.read_bytes()
    css = source_file.suffix.lower() == ".css"
    encoding = _css_encoding(data) if css else _bom_encoding(data) or "utf-8"
    try:
        return _decode_css(data) if css else data.decode(encoding, errors="strict")
    except (UnicodeError, LookupError, ValueError, TypeError) as exc:
        raise YapError(
            f"{source_file.relative_to(composition_dir).as_posix()}: "
            f"Layer source could not be decoded with encoding {encoding!r}.",
            error_type="validation_error",
            code="layer_source_encoding",
            suggested_action=(
                "Save the file as UTF-8 or declare its real encoding with @charset."
            ),
        ) from exc


def scan_files(composition_dir: Path, files: Iterable[Path]) -> dict[Path, _CompositionParser]:
    """Apply offline preflight validators to local files without requiring a root page."""
    parsers: dict[Path, _CompositionParser] = {}
    for source_file in sorted(files):
        try:
            local = source_file.resolve().is_relative_to(composition_dir.resolve())
        except (OSError, RuntimeError):
            local = False
        if not local:
            raise _external_asset(
                str(source_file), composition_dir, source_file, 1,
                reason="Layer symlink must resolve inside the composition folder",
            )
        if source_file.is_dir():
            continue
        pinned_hash = ({GSAP_SCRIPT: GSAP_SHA256, **PINNED_GSAP_PLUGINS}).get(source_file.name)
        if (
            pinned_hash is not None
            and not source_file.is_symlink()
            and source_file.is_file()
            and hashlib.sha256(source_file.read_bytes()).hexdigest() == pinned_hash
        ):
            continue
        suffix = source_file.suffix.lower()
        if suffix not in {".html", ".htm", ".svg", ".css", ".js", ".mjs", ".cjs"}:
            continue
        paths.validate_input(source_file)
        source = _read_source(composition_dir, source_file)
        if suffix in {".html", ".htm", ".svg"}:
            nested = _CompositionParser(composition_dir, source_file)
            nested.feed(source)
            nested.close()
            parsers[source_file] = nested
        elif suffix == ".css":
            _validate_css(source, composition_dir, source_file)
        else:
            _validate_script(source, composition_dir, source_file)
    return parsers


def _scan_composition_tree(composition_dir: Path) -> _CompositionParser:
    """Scan local source files and refuse escaping symlinks without traversing them."""
    files: list[Path] = []
    for directory, subdirs, names in os.walk(composition_dir, followlinks=False):
        files.extend(Path(directory) / name for name in sorted(subdirs + names))
    index = composition_dir / "index.html"
    paths.validate_input(index)
    return scan_files(composition_dir, files)[index]


@dataclass(frozen=True)
class _Composition:
    duration_s: float
    width: int
    height: int
    uses_gsap: bool


def _composition(
    composition_dir: Path, frames: int, fps: int,
    *, parser: _CompositionMetadataParser | None = None,
) -> _Composition:
    if frames < 1 or fps < 1:
        raise ValueError("frames and fps must be positive")
    if parser is None:
        index = composition_dir / "index.html"
        paths.validate_input(index)
        parser = _CompositionMetadataParser()
        parser.feed(_read_source(composition_dir, index))
        parser.close()
    timeline_action = (
        "Remove data-no-timeline from the root and register the GSAP timeline."
        if parser.uses_gsap
        else "Set data-no-timeline on the root for a composition without GSAP."
    )
    try:
        if len(parser.roots) != 1:
            raise ValueError("exactly one composition root is required")
        root = parser.roots[0]
        if parser.uses_gsap and "data-no-timeline" in root:
            raise ValueError("a GSAP composition root must not carry data-no-timeline")
        if not parser.uses_gsap and "data-no-timeline" not in root:
            raise ValueError("the composition root requires data-no-timeline")
        duration = float(root["data-duration"] or "")
        width = int(root["data-width"] or "")
        height = int(root["data-height"] or "")
        if width < 1 or height < 1:
            raise ValueError("composition dimensions must be positive")
        actual = renderer_frame_count(duration, fps)
    except (KeyError, ValueError) as exc:
        raise YapError(
            f"Invalid layer composition: {exc}.",
            error_type="validation_error",
            code="layer_composition",
            suggested_action=timeline_action + " Set valid duration, width and height on the root.",
        ) from exc
    if actual != frames:
        raise YapError(
            f"Composition duration gives {actual} frames; expected {frames}.",
            error_type="validation_error",
            code="layer_frame_mismatch",
            suggested_action="Set data-duration with layer_duration_s(frames, fps) and try again.",
        )
    return _Composition(duration, width, height, parser.uses_gsap)


def preflight(composition_dir: Path, frames: int, fps: int = 30) -> float:
    """Validate the composition tree, timeline and frames before any tools.

    Static scanning cannot prove a script makes no request; it blocks common network
    patterns. Engine-owned templates contain no network code, and the render tests
    run under a network-denied sandbox on macOS.
    """
    parser = _scan_composition_tree(composition_dir)
    return _composition(composition_dir, frames, fps, parser=parser).duration_s


def _existing_gsap(script: Path) -> bool:
    """Accept only a regular pinned script, preserving any conflicting caller file."""
    try:
        if not script.is_symlink():
            if not script.exists():
                return False
            if script.is_file() and hashlib.sha256(script.read_bytes()).hexdigest() == GSAP_SHA256:
                return True
    except OSError:
        pass
    raise YapError(
        f"Layer GSAP file conflicts with the pinned script: {script}.",
        error_type="validation_error",
        code="layer_gsap_conflict",
        suggested_action="Remove or rename the file so the pinned GSAP can be used.",
    )


def _copy_gsap(source: Path, script: Path) -> bool:
    """Publish a complete copy atomically without replacing an existing caller file."""
    if _existing_gsap(script):
        return False
    paths.validate_output(script, allowed_existing_suffixes=frozenset())
    descriptor, name = tempfile.mkstemp(prefix=".yap_tmp_", suffix=".js", dir=script.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        try:
            os.link(temporary, script)
        except FileExistsError:
            _existing_gsap(script)
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def render_layer(
    composition_dir: Path,
    out: Path,
    *,
    frames: int,
    variables: Mapping[str, object],
    fps: int = 30,
    install: RendererInstall | None = None,
    checks: bool = True,
    workers: int | str = RENDER_WORKERS,
    browser_gpu: bool = True,
) -> LayerResult:
    """Render a box-sized layer and convert it to verified BT.709 with alpha.

    Best-effort rendering is disabled so page errors, timelines that never
    register, and missing or unready media fail without producing an output layer.
    """
    from yap_studio.render import layer_checks

    if not ((type(workers) is int and workers > 0)
            or (type(workers) is str and workers == "auto")):
        raise ValueError("workers must be a positive integer or 'auto'")
    if type(browser_gpu) is not bool:
        raise ValueError("browser_gpu must be a bool")
    check_result = None
    render_retries = 0
    with paths.operation():
        gsap_script = composition_dir / GSAP_SCRIPT
        composition = _composition(composition_dir, frames, fps)
        existing_gsap = _existing_gsap(gsap_script) if composition.uses_gsap else False
        _scan_composition_tree(composition_dir)
        if out.suffix.lower() != ".mov":
            raise ValidationError("out", "alpha clips require a .mov suffix")
        paths.validate_output(out, allowed_existing_suffixes=frozenset())
        renderer = install if install is not None else ensure_installed()
        gsap_source = gsap_file(renderer) if composition.uses_gsap and not existing_gsap else None
        created_gsap = False
        created_variables = False
        variables_file = composition_dir / "variables.json"
        raw = composition_dir / f".layer-{uuid4().hex}.mov"
        artifact_dir = composition_dir / ".hyperframes"
        had_artifact_dir = artifact_dir.exists() or artifact_dir.is_symlink()
        try:
            if gsap_source is not None:
                created_gsap = _copy_gsap(gsap_source, gsap_script)
            if checks:
                check_result = layer_checks.check_layer(
                    composition_dir, variables, install=renderer, fps=fps,
                    duration_s=composition.duration_s, workers=workers,
                    browser_gpu=browser_gpu,
                )
            paths.atomic_write_text(
                variables_file,
                json.dumps(dict(variables)) + "\n",
                allowed_existing_suffixes=frozenset(),
            )
            created_variables = True
            deadline = 120 + frames * composition.width * composition.height / 2e6

            def attempt() -> None:
                with paths.atomic_output(raw) as temporary:
                    _run_renderer(
                        renderer.executable,
                        [
                            "render",
                            str(composition_dir.absolute()),
                            "--format",
                            "mov",
                            "--variables-file",
                            str(variables_file.absolute()),
                            "--strict-variables",
                            "--no-best-effort",
                            "--quiet",
                            "--fps",
                            str(fps),
                            "--workers",
                            str(workers),
                            "--experimental-fast-capture=false",
                            *(["--no-browser-gpu"] if not browser_gpu else []),
                            "-o",
                            str(temporary.absolute()),
                        ],
                        timeout_s=deadline,
                    )

            retry = render_with_retry(attempt)
            if retry is not None:
                render_retries = 1
                if isinstance(check_result, dict):
                    check_result["render_retries"] = [retry]
            alpha.convert_to_bt709(raw, out, frames=frames, timeout_s=deadline)
        finally:
            raw.unlink(missing_ok=True)
            if created_variables:
                variables_file.unlink(missing_ok=True)
            if created_gsap:
                gsap_script.unlink(missing_ok=True)
            if not had_artifact_dir and artifact_dir.is_dir() and not artifact_dir.is_symlink():
                shutil.rmtree(artifact_dir, ignore_errors=True)
    return LayerResult(
        out, frames, composition.width, composition.height, composition.duration_s, check_result,
        render_retries,
    )
