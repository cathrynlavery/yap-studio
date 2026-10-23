"""A creator's saved styles and default selection."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yap_studio import paths
from yap_studio.errors import YapError
from yap_studio.style.loader import (
    PRESETS_DIR,
    SHIPPED_PRESETS,
    StyleError,
    _read,
    resolve,
    resolve_style_data,
)

_LIBRARY_ID = re.compile(r"[a-z0-9_-]{1,64}")
_CONFIG_SCHEMA = "yap-studio/config@1"


class StyleRefusal(YapError):
    """Refuse to replace a saved style without an explicit request."""

    def __init__(self, style_id: str) -> None:
        super().__init__(
            f"Style {style_id!r} already exists with different content.",
            error_type="style_refusal",
            code="style_exists",
            suggested_action=(
                "Choose another file name or pass --replace to update the saved style."
            ),
        )


@dataclass(frozen=True)
class LibraryEntry:
    """One saved style and whether it is the creator's default."""

    id: str
    name: str
    path: Path
    is_default: bool


@dataclass(frozen=True)
class SavedPick:
    """One saved gallery pick selected as the creator's default."""

    id: str
    name: str
    path: Path
    workspace_copy: Path | None
    revision: int
    is_default: bool


def config_root() -> Path:
    """Return the absolute config root, honoring YAP_STUDIO_CONFIG."""
    override = os.environ.get("YAP_STUDIO_CONFIG")
    if override:
        return Path(override).expanduser().absolute()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "yap-studio"
    # Use a data folder because paths refuses writes under the sensitive .config folder.
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser().absolute() if xdg else Path.home() / ".local" / "share"
    return base / "yap-studio"


def library_dir() -> Path:
    """Return the styles folder without creating it."""
    return config_root() / "styles"


def _config() -> dict[str, Any]:
    config = config_root() / "config.json"
    if not config.exists():
        return {"schema": _CONFIG_SCHEMA}
    data = _read(config)
    if data.get("schema") != _CONFIG_SCHEMA:
        raise StyleError("config.schema", f"must be {_CONFIG_SCHEMA!r}")
    return data


def get_default() -> str | None:
    """Return the saved default specification, if one has been selected."""
    spec = _config().get("default_style")
    if spec is not None and not isinstance(spec, str):
        raise StyleError("config.default_style", "expected a style id or preset:ID")
    return spec


def set_default(spec: str) -> None:
    """Validate and save a library id or shipped preset as the default."""
    if not spec.startswith("preset:") and (
        _LIBRARY_ID.fullmatch(spec) is None or not (library_dir() / f"{spec}.json").is_file()
    ):
        raise StyleError("style", "choose a saved library id or preset:ID")
    resolve(spec)
    config = _config()
    config["default_style"] = spec
    text = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    with paths.operation():
        try:
            paths.atomic_write_text(config_root() / "config.json", text)
        except OSError as exc:
            raise StyleError("style.default", f"cannot save default: {exc}") from exc


def _write_entry(target: Path, text: str) -> None:
    """Atomically replace only an explicitly selected library style."""
    with paths.operation():
        paths.validate_output_dir(target.parent)
        if not target.exists():
            paths.atomic_write_text(target, text)
            return
        if target.is_symlink() or not target.is_file():
            paths.validate_output(target)
        # Library files retain the original JSON and have no engine output marker.
        # Stage through the guarded writer, then replace the approved library entry.
        with tempfile.TemporaryDirectory(prefix=".yap_tmp_", dir=target.parent) as folder:
            staged = Path(folder) / target.name
            paths.atomic_write_text(staged, text)
            os.replace(staged, target)


def add(
    path: str | os.PathLike[str], *, make_default: bool = False, replace: bool = False
) -> LibraryEntry:
    """Validate and import original JSON, optionally selecting it as the default."""
    source = Path(path).expanduser()
    data = _read(source)
    resolved = resolve_style_data(data)
    style_id = re.sub(r"[^a-z0-9_-]+", "_", source.stem.lower()).strip("_-")[:64]
    if _LIBRARY_ID.fullmatch(style_id) is None:
        raise StyleError(
            "style.id", "file name must contain a letter, number, underscore or hyphen"
        )
    target = library_dir() / f"{style_id}.json"
    with _save_lock():
        identical = False
        if target.exists() or target.is_symlink():
            try:
                identical = _read(target) == data
            except StyleError:
                pass
            if not identical and not replace:
                raise StyleRefusal(style_id)
        if not identical:
            text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
            try:
                _write_entry(target, text)
            except OSError as exc:
                raise StyleError("style.library", f"cannot save style: {exc}") from exc
        if make_default:
            set_default(style_id)
        return LibraryEntry(style_id, resolved["name"], target, get_default() == style_id)


def _validate_workspace_copy(path: Path) -> None:
    """Allow replacement only when the copy matches a saved library revision."""
    with paths.operation():
        paths.validate_output_dir(path.parent)
        if path.is_symlink():
            paths.validate_output(path)
        if path.exists():
            resolve_style_data(_read(path))
            existing = path.read_bytes()
            if not any(
                entry.is_file() and not entry.is_symlink() and entry.read_bytes() == existing
                for folder in (library_dir(), library_dir() / ".replaced")
                for entry in folder.glob("*.json")
            ):
                raise StyleError(
                    "style.workspace",
                    "style/my-style.json is not a copy of a saved style; "
                    "move it or pass another --workspace",
                )


@contextmanager
def _save_lock() -> Iterator[None]:
    """Serialize saves across style pages and processes."""
    import fcntl

    folder = library_dir()
    try:
        with paths.operation():
            paths.validate_output_dir(folder)
            descriptor = os.open(
                folder / ".save.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600,
            )
    except OSError as exc:
        raise StyleError("style.library", f"cannot lock style library: {exc}") from exc
    locked = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            locked = True
        except OSError as exc:
            raise StyleError("style.library", f"cannot lock style library: {exc}") from exc
        yield
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def save_pick(
    data: dict, *, name: str | None = None, workspace: Path | None = None,
    replace: str | None = None,
) -> SavedPick:
    """Save and select a pick, optionally replacing an explicit library id."""
    resolved = resolve_style_data(data)
    saved = {key: value for key, value in data.items() if key not in {"schema", "internal"}}
    saved["name"] = name if name is not None else resolved["name"]
    resolve_style_data(saved)
    base = re.sub(r"[^a-z0-9_-]+", "_", saved["name"].lower()).strip("_-")[:64]
    if _LIBRARY_ID.fullmatch(base) is None:
        raise StyleError("style.name", "choose a name with a letter or number")
    with _save_lock():
        return _save_pick(saved, base=base, workspace=workspace, replace=replace)


def _save_pick(
    saved: dict, *, base: str, workspace: Path | None, replace: str | None,
) -> SavedPick:
    """Write a validated pick while holding the library lock."""
    text = json.dumps(saved, indent=2, ensure_ascii=False) + "\n"
    if replace is not None:
        target = library_dir() / f"{replace}.json"
        if (_LIBRARY_ID.fullmatch(replace) is None
                or target.is_symlink() or not target.is_file()):
            raise StyleError("style.replace", "choose a saved library id")
        style_id = replace
    workspace_copy = None
    if workspace is not None:
        workspace_copy = Path(workspace).expanduser() / "style" / "my-style.json"
        try:
            _validate_workspace_copy(workspace_copy)
        except OSError as exc:
            raise StyleError("style.workspace", f"cannot save style copy: {exc}") from exc
    revision = 1
    if replace is not None:
        try:
            previous = target.read_bytes()
            if previous != text.encode("utf-8"):
                digest = hashlib.sha256(previous).hexdigest()
                revision_path = library_dir() / ".replaced" / f"{digest}.json"
                _write_entry(revision_path, previous.decode("utf-8"))
            _write_entry(target, text)
        except OSError as exc:
            raise StyleError("style.library", f"cannot save style: {exc}") from exc
    while replace is None:
        suffix = f"-{revision}" if revision > 1 else ""
        style_id = f"{base[:64 - len(suffix)]}{suffix}"
        target = library_dir() / f"{style_id}.json"
        if target.is_symlink():
            revision += 1
            continue
        if target.exists():
            try:
                if _read(target) == saved:
                    break
            except StyleError:
                pass
            revision += 1
            continue
        try:
            with paths.operation():
                paths.validate_output_dir(target.parent)
                with tempfile.TemporaryDirectory(prefix=".yap_tmp_", dir=target.parent) as folder:
                    staged = Path(folder) / target.name
                    paths.atomic_write_text(staged, text)
                    _publish_new(staged, target)
            break
        except FileExistsError:
            continue
        except OSError as exc:
            raise StyleError("style.library", f"cannot save style: {exc}") from exc
    if workspace_copy is not None:
        try:
            if replace is None:
                _validate_workspace_copy(workspace_copy)
            with paths.operation():
                if workspace_copy.exists():
                    with tempfile.TemporaryDirectory(
                        prefix=".yap_tmp_", dir=workspace_copy.parent
                    ) as folder:
                        staged = Path(folder) / workspace_copy.name
                        paths.atomic_write_text(staged, text)
                        os.replace(staged, workspace_copy)
                else:
                    paths.atomic_write_text(workspace_copy, text)
        except OSError as exc:
            if replace is not None:
                try:
                    _write_entry(target, previous.decode("utf-8"))
                except OSError as restore_error:
                    raise StyleError(
                        "style.library", f"style {style_id} could not be restored: {restore_error}",
                    ) from restore_error
                raise StyleError(
                    "style.workspace",
                    f"style {style_id} was not updated because the copy could not be written: "
                    f"{exc}",
                ) from exc
            raise StyleError(
                "style.workspace",
                f"style was saved to the library as {style_id} but not made the default "
                f"because the copy could not be written: {exc}",
            ) from exc
    set_default(style_id)
    return SavedPick(style_id, saved["name"], target, workspace_copy, revision, True)


def current_default_data() -> dict:
    """Return the resolved style selected for new projects."""
    return resolve().data


def create(name: str, *, preset: str = "clean_minimal", out: str | None = None) -> Path:
    """Create a portable style file from a shipped preset without replacing a file."""
    if preset not in SHIPPED_PRESETS:
        raise StyleError("style.preset", f"choose one of {', '.join(SHIPPED_PRESETS)}")
    data = _read(PRESETS_DIR / f"{preset}.json")
    data["name"] = name
    resolve_style_data(data)
    style_id = re.sub(r"[^a-z0-9_-]+", "_", name.lower()).strip("_-")[:64]
    if out is None and _LIBRARY_ID.fullmatch(style_id) is None:
        raise StyleError("style.name", "choose a name with a letter or number")
    target = Path(out).expanduser() if out is not None else Path(f"{style_id}.json")
    exists = StyleError("style.new", f"{target} already exists; choose another path")
    if target.exists() or target.is_symlink():
        raise exists
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    with paths.operation():
        paths.validate_output_dir(target.parent)
        paths.validate_output(target)
        # Stage through the guarded writer, then link the stage into place.
        # A link never replaces a file another writer published after the check.
        with tempfile.TemporaryDirectory(prefix=".yap_tmp_", dir=target.parent) as folder:
            staged = Path(folder) / target.name
            paths.atomic_write_text(staged, text)
            try:
                _publish_new(staged, target)
            except FileExistsError as exc:
                raise exists from exc
            except OSError as exc:
                raise StyleError("style.new", f"cannot save style: {exc}") from exc
    return target


def _publish_new(staged: Path, target: Path) -> None:
    """Publish a whole staged file without replacing a file that already exists."""
    try:
        os.link(staged, target)
        return
    except FileExistsError:
        raise
    except OSError:
        pass  # Some drives, such as exFAT, have no hard links.
    _rename_no_replace(staged, target)


def _rename_no_replace(source: Path, target: Path) -> None:
    """Rename atomically, raising FileExistsError rather than replacing the target."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        call = getattr(libc, "renamex_np", None)
        args: tuple = (os.fsencode(source), os.fsencode(target), ctypes.c_uint(0x4))  # EXCL
    elif sys.platform.startswith("linux"):
        call = getattr(libc, "renameat2", None)
        here = ctypes.c_int(-100)  # AT_FDCWD
        args = (here, os.fsencode(source), here, os.fsencode(target), ctypes.c_uint(1))
    else:
        call = None
    if call is None:
        raise OSError(errno.ENOTSUP, "this system cannot add a file without risking another")
    if call(*args) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), os.fspath(target))


def list_entries() -> list[LibraryEntry]:
    """List saved styles in id order, marking the selected library default."""
    default = get_default()
    entries = []
    for path in sorted(library_dir().glob("*.json")):
        if _LIBRARY_ID.fullmatch(path.stem) is None:
            continue
        data = resolve_style_data(_read(path))
        entries.append(LibraryEntry(path.stem, data["name"], path, path.stem == default))
    return entries


def find_by_name(name: str) -> LibraryEntry | None:
    """Find the newest saved style with a trimmed, case-insensitive name."""
    matches = [entry for entry in list_entries()
               if entry.name.strip().lower() == name.strip().lower()]
    return min(matches, key=lambda entry: (-entry.path.stat().st_mtime_ns, entry.id),
               default=None)
