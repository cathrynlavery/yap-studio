"""Bundled font resolution, composition assets, and distribution licenses."""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from yap_studio.render import fonts
from yap_studio.render.fonts import (
    BUNDLED,
    FONTS_DIR,
    copy_fonts,
    font_face_css,
    font_stack,
    resolve_font,
    style_fonts,
)
from yap_studio.style import SHIPPED_PRESETS, resolve

ROOT = Path(__file__).parents[1]

# HyperFrames 0.8.222 FONT_ALIAS_MAP keys, supplied in hf-font-aliases.txt.
RENDERER_FONT_ALIASES = {
    "andale mono",
    "archivo black",
    "arial",
    "arial black",
    "avenir",
    "avenir next",
    "bebas neue",
    "book antiqua",
    "calibri",
    "cambria",
    "candara",
    "consolas",
    "corbel",
    "courier",
    "courier new",
    "dejavu sans",
    "dejavu sans mono",
    "dejavu serif",
    "din alternate",
    "eb garamond",
    "futura",
    "garamond",
    "geneva",
    "georgia",
    "helvetica",
    "helvetica bold",
    "helvetica neue",
    "ibm plex mono",
    "inter",
    "jetbrains mono",
    "lato",
    "league gothic",
    "liberation mono",
    "liberation sans",
    "liberation serif",
    "lucida console",
    "lucida grande",
    "lucida sans",
    "lucida sans typewriter",
    "lucida sans unicode",
    "menlo",
    "monaco",
    "montserrat",
    "noto sans",
    "noto sans japanese",
    "noto sans jp",
    "nunito",
    "open sans",
    "optima",
    "oswald",
    "outfit",
    "palatino",
    "palatino linotype",
    "playfair display",
    "poppins",
    "roboto",
    "segoe ui",
    "sf mono",
    "sf pro",
    "sf pro display",
    "sf pro rounded",
    "sf pro text",
    "source code pro",
    "space mono",
    "tahoma",
    "times",
    "times new roman",
    "trebuchet ms",
    "verdana",
}


def test_registry_covers_all_bundled_families_and_files() -> None:
    assert set(BUNDLED) == {
        "Instrument Serif",
        "Gelasio",
        "Inter",
        "Geist",
        "Alata",
        "Archivo Black",
        "Anton SC",
        "Titan One",
        "Permanent Marker",
        "Space Mono",
    }
    files = [face.file for font in BUNDLED.values() for face in font.faces]
    assert len(files) == len(set(files)) == 15
    assert set(files) == {file.name for file in FONTS_DIR.iterdir() if file.is_file()}
    assert len({font.license_file for font in BUNDLED.values()}) == len(BUNDLED)
    assert BUNDLED["Permanent Marker"].license_file == "LICENSES/Apache-2.0.txt"


@pytest.mark.parametrize("name", BUNDLED)
def test_bundled_font_has_faces_license_and_unaliased_family(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_lookup(requested: str) -> bool:
        pytest.fail(f"bundled font triggered system lookup: {requested}")

    monkeypatch.setattr(fonts, "_is_font_installed", unexpected_lookup)
    font = BUNDLED[name]
    assert font.name == name
    assert font.family == f"Yap Studio {name}"
    assert font.family.lower() not in RENDERER_FONT_ALIASES
    assert font.generic in {"serif", "sans-serif", "monospace", "cursive", "fantasy", "system-ui"}
    assert font.faces
    for face in font.faces:
        assert (FONTS_DIR / face.file).read_bytes().startswith(b"wOF2")
        assert face.style in {"normal", "italic"}
    assert (ROOT / font.license_file).is_file()
    result = resolve_font(name)
    assert result.name == name
    assert result.family == font.family
    assert result.bundled is True
    assert result.installed is None
    assert result.warning is None


def test_presets_and_schema_defaults_only_use_bundled_fonts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    styles = [resolve(f"preset:{preset}") for preset in SHIPPED_PRESETS]
    styles.append(resolve(None))
    minimal = tmp_path / "minimal.json"
    minimal.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    styles.append(resolve(minimal))
    for style in styles:
        names = style_fonts(style.data)
        assert names
        assert set(names) <= BUNDLED.keys()


def test_style_fonts_preserves_section_order_and_removes_duplicates() -> None:
    style = {
        "graphics": {"font": "Inter"},
        "hook": {"font": "Gelasio"},
        "captions": {"font": "Inter"},
    }
    assert style_fonts(style) == ["Inter", "Gelasio"]
    style["graphics"]["font"] = "Space Mono"
    assert style_fonts(style) == ["Inter", "Gelasio", "Space Mono"]


@pytest.mark.parametrize("installed", [False, True])
def test_system_font_resolution_warns_only_when_missing(
    installed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fonts, "_is_font_installed", lambda name: installed)
    result = resolve_font("Unknown Font")
    assert result.name == result.family == "Unknown Font"
    assert result.bundled is False
    assert result.installed is installed
    if installed:
        assert result.warning is None
    else:
        assert result.warning is not None
        assert 'font "Unknown Font"' in result.warning
        assert "not bundled" in result.warning
        assert "not found on this machine" in result.warning
        assert "substitute" in result.warning


@pytest.mark.parametrize("name, installed", [("custom family", True), ("Custom", False)])
def test_installed_font_lookup_checks_exact_fontconfig_families(
    name: str, installed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fonts.shutil, "which", lambda command: "fc-list")

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command == ["fc-list", ":", "family"]
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["timeout"] == 10
        assert kwargs["capture_output"] is True
        output = "Custom Family,Alternate Family\nOther\n"
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(fonts.subprocess, "run", run)
    assert resolve_font(name).installed is installed


@pytest.mark.parametrize("failure", ["unavailable", "error", "timeout", "nonzero"])
def test_font_lookup_falls_back_to_standard_folder_scan(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        fonts.shutil, "which", lambda command: None if failure == "unavailable" else "fc-list"
    )

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 10)
        if failure == "error":
            raise OSError("font lookup unavailable")
        return subprocess.CompletedProcess(command, 1, "", "font lookup failed")

    monkeypatch.setattr(fonts.subprocess, "run", run)
    monkeypatch.setattr(fonts, "_font_directories", lambda: (tmp_path / "missing", tmp_path))
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "CustomFamily.OTF").write_bytes(b"installed font fixture")
    (nested / "NotAFont.txt").write_text("not a font", encoding="utf-8")
    assert resolve_font("custom family").installed is True
    assert resolve_font("Not A Font").warning is not None
    assert resolve_font("Absent").warning is not None


def test_font_face_css_only_emits_requested_bundled_faces() -> None:
    css = font_face_css(["Inter", "Nope"])
    assert css.count("@font-face {") == 2
    assert css.count('font-family: "Yap Studio Inter";') == 2
    assert 'src: url("fonts/inter-variable.woff2") format("woff2");' in css
    assert 'src: url("fonts/inter-italic-variable.woff2") format("woff2");' in css
    assert "font-style: normal;" in css
    assert "font-style: italic;" in css
    assert css.count("font-weight: 100 900;") == 2
    assert css.count("font-display: block;") == 2
    assert "Nope" not in css
    assert "Gelasio" not in css
    assert css == font_face_css(iter(["Inter", "Nope", "Inter"]))
    assert font_face_css(["Nope"]) == font_face_css([]) == ""


def test_font_face_css_preserves_request_order_and_custom_relative_prefix() -> None:
    css = font_face_css(["Gelasio", "Inter"], url_prefix="assets/fonts/")
    assert css.index("Yap Studio Gelasio") < css.index("Yap Studio Inter")
    assert 'url("assets/fonts/gelasio-variable.woff2")' in css
    assert 'url("assets/fonts/inter-variable.woff2")' in css
    assert css.count("font-weight: 400 700;") == 2


def test_geist_font_faces_and_stack() -> None:
    css = font_face_css(["Geist"])
    assert css.count("@font-face {") == 2
    assert css.count('font-family: "Yap Studio Geist";') == 2
    assert css.count("font-weight: 100 900;") == 2
    assert css.count("font-style: normal;") == 1
    assert css.count("font-style: italic;") == 1
    assert 'url("fonts/geist-variable.woff2")' in css
    assert 'url("fonts/geist-italic-variable.woff2")' in css
    assert font_stack("Geist") == '"Yap Studio Geist", sans-serif'


def test_geist_notice_and_license() -> None:
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "Geist: SIL Open Font License 1.1 (LICENSES/OFL-geist.txt)" in notice
    license_text = (ROOT / "LICENSES/OFL-geist.txt").read_text(encoding="utf-8")
    assert "SIL Open Font License, Version 1.1" in license_text
    assert "The Geist Project Authors" in license_text


@pytest.mark.parametrize(
    "name, stack",
    [
        ("Inter", '"Yap Studio Inter", sans-serif'),
        ("Instrument Serif", '"Yap Studio Instrument Serif", serif'),
        ("Space Mono", '"Yap Studio Space Mono", monospace'),
        ("Permanent Marker", '"Yap Studio Permanent Marker", cursive'),
        ("Unknown Font", '"Unknown Font", sans-serif'),
        ('A "quoted" font', '"A \\"quoted\\" font", sans-serif'),
        ("Font\nName", '"Font\\a Name", sans-serif'),
    ],
)
def test_font_stack_quotes_family_and_uses_css_generic(name: str, stack: str) -> None:
    assert font_stack(name) == stack


def test_copy_fonts_copies_only_needed_faces_byte_for_byte(tmp_path: Path) -> None:
    destination = tmp_path / "composition" / "fonts"
    copied = copy_fonts(iter(["Inter", "Nope", "Gelasio", "Inter"]), destination)
    expected = [face.file for name in ("Inter", "Gelasio") for face in BUNDLED[name].faces]
    assert copied == [destination / name for name in expected]
    assert {file.name for file in destination.iterdir()} == set(expected)
    for file in copied:
        assert file.read_bytes() == (FONTS_DIR / file.name).read_bytes()


def test_copy_fonts_creates_empty_folder_for_unbundled_fonts(tmp_path: Path) -> None:
    destination = tmp_path / "composition" / "fonts"
    assert copy_fonts(["Nope"], destination) == []
    assert destination.is_dir()
    assert list(destination.iterdir()) == []


def test_wheel_contains_every_font_and_its_license(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed; cannot verify wheel contents")
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        members = set(archive.namelist())
        license_roots = {
            member.split("/", 1)[0] + "/licenses/"
            for member in members
            if ".dist-info/licenses/" in member
        }
        assert len(license_roots) == 1
        license_root = license_roots.pop()
        for font in BUNDLED.values():
            for face in font.faces:
                member = f"yap_studio/fonts/{face.file}"
                assert member in members
                assert archive.read(member) == (FONTS_DIR / face.file).read_bytes()
            license_member = license_root + font.license_file
            assert license_member in members
            assert archive.read(license_member) == (ROOT / font.license_file).read_bytes()
