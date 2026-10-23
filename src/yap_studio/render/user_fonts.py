"""Import creator font faces into a private library outside shipped workspaces."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import tempfile
import time
import zlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from yap_studio.errors import ValidationError
from yap_studio.style.library import config_root

SCHEMA = "yap-studio/user-fonts@1"
MAX_BYTES = 20 * 1024 * 1024
SIGNATURES = {b"wOF2", b"wOFF", b"\x00\x01\x00\x00", b"OTTO", b"true"}
SUFFIXES = {".woff2", ".woff", ".ttf", ".otf"}
STORED_NAME = re.compile(
    r"user-[a-z0-9-]+-[0-9a-f]{8}-(normal|italic)-[0-9]+(-[0-9]+)?\.(woff2|woff|ttf|otf)"
)


class UserFontError(ValidationError):
    """Report an invalid creator font or a conflicting library face."""


@dataclass(frozen=True)
class UserFont:
    """One recorded face in the creator's font library."""

    family: str
    file: str
    style: str
    weight: str
    sha256: str


def user_fonts_dir() -> Path:
    """Return the creator's font library without creating it."""
    return config_root() / "fonts"


def _metadata() -> list[UserFont]:
    target = user_fonts_dir() / "fonts.json"
    if not target.exists():
        return []
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        if data["schema"] != SCHEMA or not isinstance(data["fonts"], list):
            raise ValueError("unsupported font library schema")
        result = [UserFont(**item) for item in data["fonts"]]
        if any(Path(item.file).name != item.file for item in result):
            raise ValueError("unsafe font file name")
        return result
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise UserFontError("brand fonts", f"cannot read font library: {exc}") from exc


def list_fonts() -> list[UserFont]:
    """List recorded creator faces in import order."""
    return _metadata()


def find(family: str) -> list[UserFont]:
    """Find all faces for a creator family, ignoring case."""
    return [font for font in _metadata() if font.family.casefold() == family.casefold()]


def _table(data: bytes, wanted: bytes) -> bytes | None:
    """Read an sfnt or WOFF table, if present."""
    signature = data[:4]
    if signature == b"wOFF":
        count = struct.unpack_from(">H", data, 12)[0]
        for index in range(count):
            offset = 44 + index * 20
            tag, start, length, size, _ = struct.unpack_from(">4sIIII", data, offset)
            if tag == wanted:
                table = data[start:start + length]
                if len(table) != length:
                    raise ValueError("incomplete font table")
                if length < size:
                    table = zlib.decompress(table)
                return table
    else:
        count = struct.unpack_from(">H", data, 4)[0]
        for index in range(count):
            offset = 12 + index * 16
            tag, _, start, length = struct.unpack_from(">4sIII", data, offset)
            if tag == wanted:
                table = data[start:start + length]
                if len(table) != length:
                    raise ValueError("incomplete font table")
                return table
    return None


def _name_table(data: bytes) -> tuple[str, str]:
    """Read family and subfamily from a TrueType, OpenType or WOFF name table."""
    table = _table(data, b"name")
    if table is None:
        raise ValueError("missing font name table")
    count, strings = struct.unpack_from(">HH", table, 2)
    names: dict[int, str] = {}
    for index in range(count):
        platform, _, _, name_id, length, offset = struct.unpack_from(
            ">HHHHHH", table, 6 + index * 12)
        if name_id not in {1, 2, 16, 17} or name_id in names:
            continue
        raw = table[strings + offset:strings + offset + length]
        if len(raw) != length:
            continue
        try:
            value = raw.decode("utf-16-be" if platform in {0, 3} else "mac_roman").strip()
        except UnicodeError:
            continue
        if value:
            names[name_id] = value
    family = names.get(16) or names.get(1)
    if not family:
        raise ValueError("missing font family name")
    return family, names.get(17) or names.get(2) or "Regular"


_WEIGHTS = (("extralight", 200), ("ultralight", 200), ("semibold", 600),
            ("demibold", 600), ("extrabold", 800), ("ultrabold", 800),
            ("regular", 400), ("medium", 500), ("normal", 400),
            ("thin", 100), ("light", 300), ("book", 400),
            ("bold", 700), ("black", 900), ("heavy", 900))


def _stem_weight(value: str) -> str:
    stem = value.casefold()
    if "variable" in stem or stem.endswith("vf"):
        return "100 900"
    for word, weight in sorted(_WEIGHTS, key=lambda item: -len(item[0])):
        if word in stem:
            return str(weight)
    return "400"


def _stem_family(stem: str) -> str:
    first = re.split(r"[-_]", stem, maxsplit=1)[0]
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", first).strip()


def _face_details(data: bytes, subfamily: str) -> tuple[str, str]:
    os2 = _table(data, b"OS/2")
    slant = any(word in subfamily.casefold() for word in ("italic", "oblique"))
    style = "italic" if slant else "normal"
    weight = _stem_weight(subfamily)
    if os2 is not None:
        if len(os2) >= 6:
            weight = str(max(1, min(1000, struct.unpack_from(">H", os2, 4)[0])))
        if len(os2) >= 64:
            style = "italic" if struct.unpack_from(">H", os2, 62)[0] & 1 else "normal"
    if _table(data, b"fvar") is not None:
        weight = "100 900"
    return style, weight


def _atomic_bytes(target: Path, data: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".font_", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _library_lock(library: Path, *, timeout_s: float = 30) -> Iterator[None]:
    """Hold the library lock until all face files and metadata are written."""
    import fcntl

    try:
        library.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            library / ".lock", os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600,
        )
    except OSError as exc:
        raise UserFontError("brand fonts", f"cannot lock font library: {exc}") from exc
    locked = False
    try:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise UserFontError(
                        "brand fonts", "another font import is in progress; try again",
                    ) from None
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
            except OSError as exc:
                raise UserFontError("brand fonts", f"cannot lock font library: {exc}") from exc
        yield
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _stored_name(name: str, style: str, weight: str, suffix: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")[:50] or "font"
    family_id = hashlib.sha256(name.casefold().encode()).hexdigest()[:8]
    return f"user-{slug}-{family_id}-{style}-{weight.replace(' ', '-')}{suffix}"


def add_font_files(
    files: Sequence[Path], *, family: str | None = None, replace: bool = False,
) -> list[UserFont]:
    """Validate and atomically copy font faces into the creator's library."""
    if not files:
        raise UserFontError("brand fonts", "provide at least one font file")
    if family is not None and (not family.strip() or len(family) > 100 or
                               any(ord(char) < 32 for char in family)):
        raise UserFontError("family", "use a font family name of 1 to 100 characters")
    library = user_fonts_dir()
    with _library_lock(library):
        return _add_font_files_locked(files, family=family, replace=replace, library=library)


def _add_font_files_locked(
    files: Sequence[Path], *, family: str | None, replace: bool, library: Path,
) -> list[UserFont]:
    existing = _metadata()
    staged: list[tuple[UserFont, bytes]] = []
    sources = []
    for given in map(Path, files):
        if given.is_dir():
            try:
                sources.extend(sorted((item for item in given.iterdir()
                                       if not item.name.startswith(".") and item.is_file()
                                       and item.suffix.lower() in SUFFIXES),
                                      key=lambda item: item.name))
            except OSError as exc:
                raise UserFontError("font folder", f"cannot read {given}: {exc}") from exc
        else:
            sources.append(given)
    for source in sources:
        if source.suffix.lower() not in SUFFIXES:
            raise UserFontError("font file", "use a WOFF2, WOFF, TTF or OTF file")
        try:
            if not source.is_file() or source.stat().st_size > MAX_BYTES:
                raise ValueError("file is missing or larger than 20 MB")
            data = source.read_bytes()
        except OSError as exc:
            raise UserFontError("font file", f"cannot read {source}: {exc}") from exc
        except ValueError as exc:
            raise UserFontError("font file", f"{source}: {exc}") from exc
        if len(data) > MAX_BYTES or data[:4] not in SIGNATURES:
            raise UserFontError("font file", f"{source}: invalid font signature or size")
        if data[:4] == b"wOF2":
            detected_family, subfamily = _stem_family(source.stem), source.stem
            style = "italic" if any(word in subfamily.casefold() for word in
                                      ("italic", "oblique")) else "normal"
            weight = _stem_weight(subfamily)
        else:
            try:
                detected_family, subfamily = _name_table(data)
                style, weight = _face_details(data, subfamily)
            except (ValueError, struct.error, zlib.error) as exc:
                raise UserFontError(
                    "font file", f"{source}: cannot read font names: {exc}") from exc
        name = family or detected_family
        from yap_studio.render.fonts import BUNDLED

        if any(name.casefold() == bundled.casefold() for bundled in BUNDLED):
            raise UserFontError(
                "brand fonts", f"{name} is a bundled font; choose another name with --family",
            )
        digest = hashlib.sha256(data).hexdigest()
        filename = _stored_name(name, style, weight, source.suffix.lower())
        if (not STORED_NAME.fullmatch(filename) or
                (library / filename).resolve().parent != library.resolve()):
            raise UserFontError("brand fonts", "unsafe font file name")
        face = UserFont(name, filename, style, weight, digest)
        previous = next(
            (item for item in [*existing, *(staged_face for staged_face, _ in staged)]
             if item.family.casefold() == name.casefold() and item.style == style
             and item.weight == weight), None)
        if previous and previous.sha256 != digest and not replace:
            raise UserFontError("brand fonts", f"{name} {style} already exists; use --replace")
        staged.append((face, data))
    try:
        library.mkdir(parents=True, exist_ok=True)
        updated = existing.copy()
        for face, data in staged:
            _atomic_bytes(library / face.file, data)
            updated = [item for item in updated if not (
                item.family.casefold() == face.family.casefold() and item.style == face.style
                and item.weight == face.weight)]
            updated.append(face)
        payload = json.dumps({"schema": SCHEMA, "fonts": [asdict(item) for item in updated]},
                             indent=2, ensure_ascii=False) + "\n"
        _atomic_bytes(library / "fonts.json", payload.encode("utf-8"))
    except OSError as exc:
        raise UserFontError("brand fonts", f"cannot save font library: {exc}") from exc
    return [face for face, _ in staged]
