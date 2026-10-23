"""Make a verified registry item use local renderer scripts and font faces."""

from __future__ import annotations

import hashlib
import html
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from yap_studio.errors import DependencyError, ValidationError, YapError
from yap_studio.hf_registry import catalog
from yap_studio.hf_registry.install import item_files
from yap_studio.render import fonts, hyperframes, user_fonts
from yap_studio.render.hyperframes import RendererInstall

_GOOGLE_HOSTS = {"fonts.googleapis.com", "fonts.gstatic.com"}
_GENERIC = {
    "serif",
    "sans-serif",
    "monospace",
    "cursive",
    "fantasy",
    "system-ui",
    "ui-sans-serif",
    "ui-serif",
    "ui-monospace",
    "ui-rounded",
    "emoji",
    "math",
    "fangsong",
    "-apple-system",
    "blinkmacsystemfont",
    "inherit",
    "initial",
    "unset",
    "revert",
    "revert-layer",
}
_FONT_DECLARATION = re.compile(
    r"""(?<![\w-])(--[A-Za-z0-9_-]+|font-family)\s*:\s*"""
    r"""((?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^;"'{}])+)""",
    re.IGNORECASE,
)
_CSS_COMMENT_OR_STRING = re.compile(
    r"""/\*.*?(?:\*/|$)|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*' """,
    re.DOTALL | re.VERBOSE,
)
_SCRIPT_FONT = re.compile(
    r"""(?:\bfontFamily|["']font-family["'])\s*:\s*(["'`])(.*?)\1""",
    re.DOTALL,
)
_IMPORT = re.compile(
    r"""@import\s+(?:url\(\s*(?P<url>"[^"]*"|'[^']*'|[^)]*)\s*\)"""
    r"""|(?P<quoted>"[^"]*"|'[^']*'))[^;{}]*(?:;|(?=[{}]|$))""",
    re.IGNORECASE,
)
_GSAP = re.compile(r"https://cdn\.jsdelivr\.net/npm/gsap@([^/]+)/dist/([^/?#]+)")


@dataclass(frozen=True)
class Localized:
    """The item copy and the fonts and scripts its host must provide."""

    main: str
    families: tuple[str, ...]
    plugins: tuple[str, ...]
    font_css: str


def host_head(localized: Localized) -> str:
    """Load pinned scripts and alias faces from the host composition root."""
    tags = [f'<script src="{hyperframes.GSAP_SCRIPT}"></script>']
    tags.extend(
        f'<script src="{html.escape(plugin, quote=True)}"></script>' for plugin in localized.plugins
    )
    if localized.font_css:
        tags.append(f"<style data-yap-fonts>\n{localized.font_css}\n</style>")
    return "\n".join(tags)


def _error(name: str, code: str, detail: str) -> YapError:
    return YapError(
        f'HyperFrames registry item "{name}": {detail}',
        error_type="validation_error",
        code=code,
        suggested_action="This item cannot render offline with the pinned renderer.",
    )


def _google_url(url: str) -> bool:
    return urlsplit(html.unescape(url)).hostname in _GOOGLE_HOSTS


def _url_families(url: str) -> list[str]:
    result: list[str] = []
    for value in parse_qs(urlsplit(html.unescape(url)).query).get("family", []):
        result.extend(family.split(":", 1)[0].strip() for family in value.split("|"))
    return result


def _font_stack(value: str) -> list[str]:
    """Split family lists without splitting quoted names or variable fallbacks."""
    parts: list[str] = []
    start, depth = 0, 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif quote:
            if char == quote:
                quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return parts


def _resolve_families(
    value: str,
    properties: Mapping[str, str],
    visiting: frozenset[str] = frozenset(),
) -> list[str]:
    names: list[str] = []
    for part in _font_stack(value):
        variable = re.fullmatch(r"var\(\s*(--[A-Za-z0-9_-]+)\s*(?:,(.*))?\)", part, re.I | re.S)
        if variable:
            name, fallback = variable.groups()
            # Include both declared values and fallbacks so either branch has local faces.
            if name not in visiting:
                names.extend(
                    _resolve_families(properties.get(name, ""), properties, visiting | {name})
                )
            if fallback:
                names.extend(_resolve_families(fallback, properties, visiting | {name}))
        else:
            family = part.strip("\"'`").strip()
            if (
                family
                and not family.lower().startswith("var(")
                and family.casefold() not in _GENERIC
            ):
                names.append(family)
    return names


def _families(source: str) -> list[str]:
    """Collect requested families, including referenced properties and all var() fallbacks."""
    # HyperFrames 0.8.181 does not request fonts from font: shorthand declarations.
    source = _CSS_COMMENT_OR_STRING.sub(
        lambda match: " " if match[0].startswith("/*") else match[0],
        source,
    )
    declarations = [
        (match[1], re.sub(r"\s*!important\s*$", "", match[2], flags=re.I).strip())
        for match in _FONT_DECLARATION.finditer(source)
    ]
    properties = {name: value for name, value in declarations if name.startswith("--")}
    values = [value for name, value in declarations if name.lower() == "font-family"]
    values.extend(match[2] for match in _SCRIPT_FONT.finditer(source))
    return [family for value in values for family in _resolve_families(value, properties)]


def _output_file(dest: Path, relative: str) -> Path:
    target = dest / relative
    if not target.resolve().is_relative_to(dest.resolve()):
        raise ValidationError("registry destination", f"escaping path {relative}")
    return target


def _plugin(name: str, filename: str, dest: Path, install: RendererInstall) -> None:
    expected = hyperframes.PINNED_GSAP_PLUGINS.get(filename)
    if expected is None:
        raise _error(name, "registry_item_gsap_plugin", f"unsupported GSAP plugin {filename}")
    source = install.prefix / "node_modules/gsap/dist" / filename
    try:
        intact = source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == expected
    except OSError:
        intact = False
    if not intact:
        raise DependencyError(
            f'HyperFrames registry item "{name}": '
            f"pinned GSAP plugin {filename} is missing or altered.",
            error_type="dependency_error",
            code="registry_item_gsap_plugin",
            suggested_action=(
                f"Delete the renderer folder {install.prefix} and run "
                "yap-studio renderer install again."
            ),
        )
    shutil.copyfile(source, _output_file(dest, filename))


class _ItemDurationParser(HTMLParser):
    """Edit duration values on the item root and its full-length child clips."""

    def __init__(self, source: str, root: str, duration: float) -> None:
        super().__init__()
        self.root = root
        self.duration = str(duration)
        self.edits: list[tuple[int, int]] = []
        self._parents: list[tuple[str, str | None]] = []
        self._html_duration: str | None = None
        self._lines = [0]
        for line in source.split("\n")[:-1]:
            self._lines.append(self._lines[-1] + len(line) + 1)
        self.feed(source)
        self.close()
        for start, end in sorted(self.edits, reverse=True):
            source = source[:start] + self.duration + source[end:]
        self.source = source

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        is_root = attributes.get("data-composition-id") == self.root
        if tag == "html":
            self._html_duration = attributes.get("data-composition-duration")
        parent_duration = self._parents[-1][1] if self._parents else None
        whole_clip = False
        if parent_duration is not None and "data-composition-id" not in attributes:
            try:
                whole_clip = (
                    float(attributes.get("data-start") or "0") == 0
                    and float(attributes.get("data-duration") or "nan") == float(parent_duration)
                )
            except ValueError:
                pass
        selected = {"data-duration"} if is_root or whole_clip else set()
        if tag == "html":
            selected.add("data-composition-duration")
        if selected:
            raw = self.get_starttag_text() or ""
            line, column = self.getpos()
            offset = self._lines[line - 1] + column
            # Tokenize attributes so quoted JSON and markup-like values stay untouched.
            for match in hyperframes._HTML_ATTRIBUTE.finditer(raw, 1):
                if match[1].lower() in selected and match.lastgroup is not None:
                    start, end = match.span(match.lastgroup)
                    self.edits.append((offset + start, offset + end))
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
                       "meta", "param", "source", "track", "wbr"}:
            authored = (attributes.get("data-duration") or self._html_duration
                        if is_root and tag != "html" else None)
            self._parents.append((tag, authored))

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self._parents) - 1, -1, -1):
            if self._parents[index][0] == tag:
                del self._parents[index:]
                break


class _ItemParser(HTMLParser):
    """Collect exact source edits without reserializing the upstream document."""

    def __init__(
        self,
        name: str,
        source: str,
        dest: Path,
        install: RendererInstall,
    ) -> None:
        super().__init__()
        self.name, self.source, self.dest, self.install = (
            name,
            source,
            dest,
            install,
        )
        self.edits: list[tuple[int, int, str]] = []
        self.families: list[str] = []
        self.styles: list[str] = []
        self.inline_styles: list[str] = []
        self.font_values: list[str] = []
        self.plugins: list[str] = []
        self.root: str | None = None
        self._script: int | None = None
        self._style = False
        self._lines = [0]
        for line in source.splitlines(keepends=True):
            self._lines.append(self._lines[-1] + len(line))

    def _offset(self) -> int:
        line, column = self.getpos()
        return self._lines[line - 1] + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        raw = self.get_starttag_text() or ""
        start = self._offset()
        attributes = dict(attrs)
        if self.root is None and attributes.get("data-composition-id"):
            self.root = attributes["data-composition-id"]
        if attributes.get("style"):
            self.inline_styles.append(attributes["style"])
        if attributes.get("data-font-family"):
            self.font_values.append(attributes["data-font-family"])
        if tag == "style":
            self._style = True
        url = attributes.get("href")
        if tag == "link" and url and _google_url(url):
            self.families.extend(_url_families(url))
            self.edits.append((start, start + len(raw), ""))
        src = attributes.get("src")
        match = _GSAP.fullmatch(src or "")
        if tag != "script" or match is None:
            return
        version, filename = match.groups()
        if version != hyperframes.GSAP_VERSION:
            raise _error(
                self.name,
                "registry_item_gsap_version",
                f"GSAP {version} differs from pinned {hyperframes.GSAP_VERSION}",
            )
        if filename != hyperframes.GSAP_SCRIPT:
            if filename not in self.plugins:
                _plugin(self.name, filename, self.dest, self.install)
                self.plugins.append(filename)
        self._script = start

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag == "script" and self._script is not None:
            end = self._offset() + len(self.get_starttag_text() or "")
            self.edits.append((self._script, end, ""))
            self._script = None
        self._style = False

    def handle_endtag(self, tag: str) -> None:
        if tag == "style":
            self._style = False
        if tag == "script" and self._script is not None:
            self.edits.append((self._script, self.source.index(">", self._offset()) + 1, ""))
            self._script = None

    def handle_data(self, data: str) -> None:
        if self._style:
            self.styles.append(data)

            def remove_import(match: re.Match[str]) -> str:
                url = (match["url"] or match["quoted"]).strip().strip("\"'")
                if not _google_url(url):
                    return match[0]
                self.families.extend(_url_families(url))
                return ""

            cleaned = _IMPORT.sub(remove_import, data)
            if cleaned != data:
                start = self._offset()
                self.edits.append((start, start + len(data), cleaned))
        else:
            self.font_values.extend(match[2] for match in _SCRIPT_FONT.finditer(data))


def _font_css(families: tuple[str, ...], directory: Path, target: str) -> str:
    aliases = {
        family: "Space Mono"
        if re.search(r"mono|code|courier|consolas|menlo", family, re.I)
        else target
        for family in families
    }
    for local in set(aliases.values()):
        if local not in fonts.BUNDLED and not user_fonts.find(local):
            raise ValidationError(
                "registry font", f"{local!r} has no bundled or creator font faces"
            )
    if not aliases:
        return ""
    for filename, _ in fonts.font_files(aliases.values()):
        _output_file(directory, f"fonts/{filename}")
    fonts.copy_fonts(aliases.values(), directory / "fonts")
    blocks: list[str] = []
    for alias, local in aliases.items():
        family = (
            fonts.BUNDLED[local].family if local in fonts.BUNDLED else f"Yap Studio User {local}"
        )
        blocks.append(
            fonts.font_face_css([local]).replace(
                f"font-family: {fonts._css_string(family)};",
                f"font-family: {fonts._css_string(alias)};",
            )
        )
    return "\n".join(blocks)


def localize(
    name: str,
    dest: Path,
    *,
    fonts_by_role: Mapping[str, str] | None = None,
    role: str = "cards",
    install: RendererInstall,
    duration_s: float | None = None,
) -> Localized:
    """Copy a verified item and prepare local scripts and fonts for its host."""
    entry = catalog.item(name)
    source_root = item_files(name)
    if dest.resolve().is_relative_to(source_root.resolve()):
        raise ValidationError("registry destination", "must be outside the installed item folder")
    if entry["status"] == "helper":
        source = (source_root / entry["main"]).read_text(encoding="utf-8")
        scripts = re.findall(r"<script\s*>(.*?)</script\s*>", source, re.I | re.S)
        if len(scripts) != 1:
            raise ValidationError("registry helper", "expected one inline script")
        registry = catalog.load_catalog()["registry"]
        notice = (f'// Modified by Yap Studio: local script from HyperFrames registry item '
                  f'"{name}" ({entry["upstream_path"]} at commit {registry["commit"]}), '
                  'Copyright 2026 HeyGen, Inc., Apache License 2.0.\n')
        main = str(Path(entry["main"]).with_suffix(".js"))
        target = _output_file(dest, main)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(notice + scripts[0].strip() + "\n", encoding="utf-8")
        try:
            hyperframes.scan_files(dest, dest.rglob("*"))
        except YapError as exc:
            raise _error(name, "registry_item_external", exc.message) from exc
        return Localized(main, (), (), "")
    for relative in entry["files"]:
        target = _output_file(dest, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / relative, target)
    main = dest / entry["main"]
    source = main.read_text(encoding="utf-8")
    parser = _ItemParser(name, source, dest, install)
    parser.feed(source)
    parser.close()
    # The compiler collects style blocks before inline styles, then data-font-family values.
    font_source = "\n;\n".join(
        [
            *parser.styles,
            *parser.inline_styles,
            *(f"font-family: {value};" for value in parser.font_values),
            *(f"font-family: {fonts._css_string(family)};" for family in parser.families),
        ]
    )
    families = tuple(dict.fromkeys(_families(font_source)))
    css = _font_css(families, dest, (fonts_by_role or {}).get(role) or "Geist")
    for start, end, replacement in sorted(parser.edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    if duration_s is not None:
        if parser.root is None:
            raise ValidationError("registry item", "composition root is missing")
        source = _ItemDurationParser(source, parser.root, duration_s).source
    registry = catalog.load_catalog()["registry"]
    notice = (
        "<!-- Modified by Yap Studio: GSAP from the pinned renderer, web fonts replaced with local "
        f'faces. HyperFrames registry item "{name}" ({entry["upstream_path"]} at commit '
        f"{registry['commit']}), Copyright 2026 HeyGen, Inc., Apache License 2.0. -->\n"
    )
    marker = re.match(r"<!-- hyperframes-registry-item: [^\n]+ -->\r?\n?", source)
    position = marker.end() if marker else 0
    source = source[:position] + notice + source[position:]
    main.write_text(source, encoding="utf-8")
    try:
        hyperframes.scan_files(dest, dest.rglob("*"))
    except YapError as exc:
        raise _error(name, "registry_item_external", exc.message) from exc
    return Localized(entry["main"], families, tuple(parser.plugins), css)
