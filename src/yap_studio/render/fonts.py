"""Resolve portable font families and prepare composition-local font assets."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

FONTS_DIR = Path(__file__).parents[1] / "fonts"


@dataclass(frozen=True)
class FontFace:
    """A bundled WOFF2 face with its CSS style and supported weights."""

    file: str
    style: str
    weight: str


@dataclass(frozen=True)
class BundledFont:
    """A style font, its renderer family, faces, license, and CSS fallback."""

    name: str
    family: str
    faces: tuple[FontFace, ...]
    license_file: str
    generic: str


BUNDLED: dict[str, BundledFont] = {
    name: BundledFont(
        name=name,
        family=f"Yap Studio {name}",
        faces=faces,
        license_file=license_file,
        generic=generic,
    )
    for name, faces, license_file, generic in (
        (
            "Instrument Serif",
            (
                FontFace("instrument-serif-regular.woff2", "normal", "400"),
                FontFace("instrument-serif-italic.woff2", "italic", "400"),
            ),
            "LICENSES/OFL-instrumentserif.txt",
            "serif",
        ),
        (
            "Gelasio",
            (
                FontFace("gelasio-variable.woff2", "normal", "400 700"),
                FontFace("gelasio-italic-variable.woff2", "italic", "400 700"),
            ),
            "LICENSES/OFL-gelasio.txt",
            "serif",
        ),
        (
            "Inter",
            (
                FontFace("inter-variable.woff2", "normal", "100 900"),
                FontFace("inter-italic-variable.woff2", "italic", "100 900"),
            ),
            "LICENSES/OFL-inter.txt",
            "sans-serif",
        ),
        (
            "Geist",
            (
                FontFace("geist-variable.woff2", "normal", "100 900"),
                FontFace("geist-italic-variable.woff2", "italic", "100 900"),
            ),
            "LICENSES/OFL-geist.txt",
            "sans-serif",
        ),
        (
            "Alata",
            (FontFace("alata-regular.woff2", "normal", "400"),),
            "LICENSES/OFL-alata.txt",
            "sans-serif",
        ),
        (
            "Archivo Black",
            (FontFace("archivo-black-regular.woff2", "normal", "400"),),
            "LICENSES/OFL-archivoblack.txt",
            "sans-serif",
        ),
        (
            "Anton SC",
            (FontFace("anton-sc-regular.woff2", "normal", "400"),),
            "LICENSES/OFL-antonsc.txt",
            "sans-serif",
        ),
        (
            "Titan One",
            (FontFace("titan-one-regular.woff2", "normal", "400"),),
            "LICENSES/OFL-titanone.txt",
            "sans-serif",
        ),
        (
            "Permanent Marker",
            (FontFace("permanent-marker-regular.woff2", "normal", "400"),),
            "LICENSES/Apache-2.0.txt",
            "cursive",
        ),
        (
            "Space Mono",
            (
                FontFace("space-mono-regular.woff2", "normal", "400"),
                FontFace("space-mono-italic.woff2", "italic", "400"),
            ),
            "LICENSES/OFL-spacemono.txt",
            "monospace",
        ),
    )
}


@dataclass(frozen=True)
class FontResolution:
    """The renderer family and availability of a requested style font."""

    name: str
    family: str
    bundled: bool
    installed: bool | None
    warning: str | None


def _font_directories() -> tuple[Path, ...]:
    home = Path.home()
    if sys.platform == "darwin":
        return (
            Path("/System/Library/Fonts"),
            Path("/Library/Fonts"),
            home / "Library" / "Fonts",
        )
    if sys.platform == "win32":
        directories = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"]
        local = os.environ.get("LOCALAPPDATA")
        if local:
            directories.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
        return tuple(directories)
    data = os.environ.get("XDG_DATA_HOME")
    return (
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        home / ".fonts",
        (Path(data) if data else home / ".local" / "share") / "fonts",
    )


def _is_font_installed(name: str) -> bool:
    command = shutil.which("fc-list")
    if command:
        try:
            result = subprocess.run(
                [command, ":", "family"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        else:
            if result.returncode == 0:
                return any(
                    family.strip().casefold() == name.casefold()
                    for line in result.stdout.splitlines()
                    for family in line.split(",")
                )
    stem = name.replace(" ", "").casefold()
    for directory in _font_directories():
        try:
            for file in directory.rglob("*"):
                if (
                    file.suffix.casefold() in {".ttf", ".otf", ".ttc", ".woff", ".woff2", ".dfont"}
                    and file.stem.replace(" ", "").casefold() == stem
                    and file.is_file()
                ):
                    return True
        except OSError:
            continue
    return False


def resolve_font(name: str) -> FontResolution:
    """Resolve a bundled family or warn when a system font cannot be found."""
    font = BUNDLED.get(name)
    if font is not None:
        return FontResolution(name, font.family, bundled=True, installed=None, warning=None)
    from yap_studio.render import user_fonts

    if user_fonts.find(name):
        return FontResolution(name, f"Yap Studio User {name}", bundled=False,
                              installed=True, warning=None)
    installed = _is_font_installed(name)
    warning = None
    if not installed:
        warning = (
            f'font "{name}" is not bundled with Yap Studio and was not found on this machine; '
            "the renderer will substitute another font"
        )
    return FontResolution(name, name, bundled=False, installed=installed, warning=warning)


def style_fonts(style: dict) -> list[str]:
    """Return distinct caption, hook, and graphics fonts in section order."""
    names = (style[section]["font"] for section in ("captions", "hook", "graphics"))
    return list(dict.fromkeys(names))


def _css_string(value: str) -> str:
    escaped = "".join(
        f"\\{ord(char):x} "
        if ord(char) < 32 or ord(char) == 127
        else "\\" + char
        if char in {'"', "\\"}
        else char
        for char in value
    )
    return f'"{escaped}"'


def font_face_css(names: Iterable[str], *, url_prefix: str = "fonts/") -> str:
    """Emit available faces once, in request order, using composition-relative URLs."""
    blocks: list[str] = []
    for name in dict.fromkeys(names):
        font = BUNDLED.get(name)
        if font is not None:
            faces = font.faces
            family = font.family
        else:
            from yap_studio.render import user_fonts

            faces = user_fonts.find(name)
            family = f"Yap Studio User {name}"
        for face in faces:
            extension = Path(face.file).suffix.lower().removeprefix(".")
            format_name = {"ttf": "truetype", "otf": "opentype"}.get(extension, extension)
            blocks.append(
                "@font-face {\n"
                f"  font-family: {_css_string(family)};\n"
                f'  src: url({_css_string(url_prefix + face.file)}) format("{format_name}");\n'
                f"  font-style: {face.style};\n"
                f"  font-weight: {face.weight};\n"
                "  font-display: block;\n"
                "}"
            )
    return "\n\n".join(blocks)


def font_stack(name: str) -> str:
    """Return a quoted renderer family followed by a valid CSS generic fallback."""
    font = BUNDLED.get(name)
    if font:
        family, generic = font.family, font.generic
    else:
        from yap_studio.render import user_fonts

        family = f"Yap Studio User {name}" if user_fonts.find(name) else name
        generic = "sans-serif"
    return f"{_css_string(family)}, {generic}"


def font_files(names: Iterable[str]) -> list[tuple[str, Path]]:
    """Return requested bundled or creator face names and source paths."""
    result: list[tuple[str, Path]] = []
    for name in dict.fromkeys(names):
        font = BUNDLED.get(name)
        if font is not None:
            result.extend((face.file, FONTS_DIR / face.file) for face in font.faces)
        else:
            from yap_studio.render import user_fonts

            result.extend((face.file, user_fonts.user_fonts_dir() / face.file)
                          for face in user_fonts.find(name))
    return result


def font_source(filename: str) -> Path:
    """Find the source of a face retained by a render snapshot."""
    if (FONTS_DIR / filename).is_file():
        return FONTS_DIR / filename
    from yap_studio.render import user_fonts

    return user_fonts.user_fonts_dir() / filename


def copy_fonts(names: Iterable[str], dest_dir: Path) -> list[Path]:
    """Copy requested faces into a caller-owned composition folder."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for filename, source in font_files(names):
        target = dest_dir / filename
        shutil.copyfile(source, target)
        copied.append(target)
    return copied
