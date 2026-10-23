"""Validate paths and publish complete output files atomically."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import sys
import unicodedata
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Literal

from yap_studio.errors import InputError, OutputPathError

_INPUT_REALPATHS: ContextVar[frozenset[str] | None] = ContextVar("input_realpaths", default=None)
_OUTPUT_REALPATHS: ContextVar[frozenset[str] | None] = ContextVar("output_realpaths", default=None)
_CREATED_REALPATHS: ContextVar[frozenset[str] | None] = ContextVar(
    "created_realpaths", default=None
)
_NUL_REASON = "the path contains a NUL character"
_MACOS_SYSTEM_FOLDERS = (
    "/System", "/Library", "/bin", "/sbin", "/usr", "/dev", "/etc", "/var/db", "/var/log",
    "/var/root", "/private/etc", "/private/var/db", "/private/var/log", "/private/var/root",
)
_LINUX_SYSTEM_FOLDERS = (
    "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32", "/usr", "/etc", "/boot",
    "/dev", "/proc", "/sys", "/var/lib", "/var/log", "/root",
)
_SENSITIVE_HOME_FOLDERS = frozenset({
    ".ssh", ".gnupg", ".aws", ".azure", ".oci", ".docker", ".kube", ".config",
})
_DEFAULT_EXISTING_SUFFIXES = frozenset({".json"})


_MAX_REPORT_BYTES = 64 * 1024 * 1024
_ROOT_OUTPUTS = {"fillers": frozenset({"removals.json"})}
_ARTIFACT_WAV_NAME = re.compile(
    r"^(?:(detected-)?cut-\d+(-(render|plan|left|right))?"
    r"|declared-(?:render|join-\d+-lead-\d+))\.wav$"
)


@contextmanager
def operation() -> Iterator[None]:
    """Scope path records to one operation and restore any enclosing records."""
    handles = (_INPUT_REALPATHS, _OUTPUT_REALPATHS, _CREATED_REALPATHS)
    tokens = [(handle, handle.set(frozenset())) for handle in handles]
    try:
        yield
    finally:
        for handle, token in reversed(tokens):
            handle.reset(token)


def _path_text(path: str | os.PathLike[str]) -> str:
    return os.fspath(path)


def _record(which: ContextVar[frozenset[str] | None], realpath: str) -> None:
    records = which.get()
    if records is not None:
        which.set(records | {realpath})


def _case_insensitive_volume(path: str) -> bool:
    """Ask the nearest existing parent folder's volume how it compares names."""
    if sys.platform != "darwin":
        return False
    setting = os.pathconf_names.get("PC_CASE_SENSITIVE", 11)
    try:
        folder = Path(path).parent
        while not folder.is_dir():
            parent = folder.parent
            if parent == folder:
                return False
            folder = parent
        return os.pathconf(folder, setting) == 0
    except (OSError, ValueError):
        return False


def _record_contains(
    which: ContextVar[frozenset[str] | None], realpath: str, *, checked_path: str | None = None,
) -> bool:
    records = which.get()
    if not records:
        return False
    if realpath in records:
        return True
    try:
        target_stat = os.stat(checked_path if checked_path is not None else realpath)
    except (FileNotFoundError, NotADirectoryError):
        if sys.platform != "darwin":
            return False
        compared = unicodedata.normalize("NFD", realpath)
        names = (unicodedata.normalize("NFD", record) for record in records)
        if _case_insensitive_volume(realpath):
            compared = compared.casefold()
            return any(record.casefold() == compared for record in names)
        return compared in names
    except OSError:
        return False
    for recorded in records:
        try:
            recorded_stat = os.stat(recorded)
        except OSError:
            continue
        if (target_stat.st_dev, target_stat.st_ino) == (
            recorded_stat.st_dev, recorded_stat.st_ino,
        ):
            return True
    return False


def protect_input(path: str | os.PathLike[str]) -> None:
    """Reserve an input path against writes without requiring the file to exist."""
    given = _path_text(path)
    if "\x00" in given:
        raise InputError(given, _NUL_REASON)
    _record(_INPUT_REALPATHS, os.path.realpath(given))


def validate_input(path: str | os.PathLike[str]) -> Path:
    """Resolve and record an existing readable regular input file."""
    given = _path_text(path)
    if "\x00" in given:
        raise InputError(given, _NUL_REASON)
    resolved = os.path.realpath(given)
    if not os.path.isfile(resolved):
        raise InputError(given)
    try:
        with open(resolved, "rb"):
            pass
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise InputError(given, f"cannot open it for reading ({detail})") from exc
    if not _record_contains(_OUTPUT_REALPATHS, resolved):
        _record(_INPUT_REALPATHS, resolved)
    return Path(resolved)


def _is_within(path: str, directory: str) -> bool:
    return path == directory or path.startswith(directory.rstrip(os.sep) + os.sep)


def _write_path_text(given: str) -> str:
    return os.fspath(Path(given))


def _guard_write(path: str | os.PathLike[str]) -> tuple[str, str]:
    """Refuse unsafe spellings, input aliases and protected locations."""
    given = _path_text(path)
    if "\x00" in given:
        raise OutputPathError("Output path contains a NUL character.")
    if ".." in re.split(r"[/\\]", given):
        raise OutputPathError(
            "Output path uses '..' to step up a folder; give the path without it."
        )
    checked = _write_path_text(given)
    resolved = os.path.realpath(checked)
    if _record_contains(_INPUT_REALPATHS, resolved, checked_path=checked):
        raise OutputPathError("Output path is an input file of this command.")
    if os.path.islink(checked):
        raise OutputPathError(
            "Output path is a symlink; give the real file path.", code="unsafe_path"
        )
    insensitive = _case_insensitive_volume(resolved)
    macos = sys.platform == "darwin"
    compared = unicodedata.normalize("NFD", resolved) if macos else resolved
    if insensitive:
        compared = compared.casefold()
    folders = _MACOS_SYSTEM_FOLDERS if macos else _LINUX_SYSTEM_FOLDERS
    if macos:
        folders = tuple(unicodedata.normalize("NFD", folder) for folder in folders)
    if any(
        _is_within(compared, folder.casefold() if insensitive else folder) for folder in folders
    ):
        raise OutputPathError("Output path is inside a system folder.", code="unsafe_path")
    home = os.path.realpath(os.path.expanduser("~"))
    if macos:
        home = unicodedata.normalize("NFD", home)
    if insensitive:
        home = home.casefold()
    if _is_within(compared, home):
        relative = os.path.relpath(compared, home)
        if relative.split(os.sep)[0] in _SENSITIVE_HOME_FOLDERS:
            raise OutputPathError(
                "Output path is inside a home folder that holds keys or settings.",
                code="unsafe_path",
            )
    return given, resolved


def _is_yap_json(
    path: str, existing_json: Literal["yap", "verify-report"]
) -> bool:
    try:
        if os.path.getsize(path) > _MAX_REPORT_BYTES:
            return False
        with open(path, "rb") as stream:
            payload = stream.read(_MAX_REPORT_BYTES + 1)
        if len(payload) > _MAX_REPORT_BYTES:
            return False
        report = json.loads(payload)
    except (OSError, UnicodeError, ValueError, RecursionError):
        return False
    if not isinstance(report, dict):
        return False
    schema = report.get("schema")
    if existing_json == "yap":
        return isinstance(schema, str) and schema.startswith("yap-studio/")
    required = {"media", "summary", "checks", "findings"}
    return (
        isinstance(schema, str)
        and schema.startswith("yap-studio/verify-report@")
        and required <= report.keys()
    )


def _is_yap_wav(path: str) -> bool:
    if _ARTIFACT_WAV_NAME.fullmatch(Path(path).name) is None:
        return False
    try:
        with open(path, "rb") as stream:
            header = stream.read(12)
    except OSError:
        return False
    return len(header) == 12 and header[:4] == b"RIFF" and header[8:] == b"WAVE"


def _is_yap_output(
    path: str,
    suffix: str,
    existing_json: Literal["yap", "verify-report"],
) -> bool:
    if suffix == ".json":
        return _is_yap_json(path, existing_json)
    if suffix == ".wav":
        return _is_yap_wav(path)
    return False


def _owned_location(target: str, receipt_path: Path, stage_receipt: bool) -> bool:
    """Permit a receipt's folder tree and its fixed project-root exception."""
    return _is_within(target, str(receipt_path.parent)) or (
        stage_receipt
        and Path(target).parent == receipt_path.parent.parent
        and Path(target).name in _ROOT_OUTPUTS.get(receipt_path.parent.name, frozenset())
    )


def _is_owned_output(path: str, owner_receipts: Sequence[str | os.PathLike[str]]) -> bool:
    """Require a receipt in the file's folder tree with its current content hash."""
    digest: str | None = None
    for receipt_path in owner_receipts:
        receipt = Path(os.path.realpath(receipt_path))
        try:
            if receipt.stat().st_size > _MAX_REPORT_BYTES:
                continue
            data = json.loads(receipt.read_bytes())
            if not isinstance(data, dict):
                continue
            schema = data.get("schema")
            if not isinstance(schema, str) or not schema.startswith("yap-studio/"):
                continue
            stage_receipt = schema in {
                "yap-studio/stage-receipt@1", "yap-studio/stage-pending@1",
            }

            if not _owned_location(os.path.realpath(path), receipt, stage_receipt):
                continue
            outputs = data.get("outputs")
            if not isinstance(outputs, list):
                continue
            for output in outputs:
                if not isinstance(output, dict) or not isinstance(output.get("path"), str):
                    continue
                candidate = Path(output["path"])
                if not candidate.is_absolute():
                    # Project-stage receipts name outputs relative to the project.
                    root = (
                        receipt.parent.parent
                        if schema in {"yap-studio/stage-receipt@1", "yap-studio/stage-pending@1"}
                        else receipt.parent
                    )
                    candidate = root / candidate
                if not _owned_location(os.path.realpath(candidate), receipt, stage_receipt):
                    continue
                try:
                    same_file = os.path.samefile(candidate, path)
                except (OSError, ValueError):
                    continue
                if not same_file:
                    continue
                if digest is None:
                    hasher = hashlib.sha256()
                    with open(path, "rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            hasher.update(chunk)
                    digest = hasher.hexdigest()
                if output.get("sha256") == digest:
                    return True
                if receipt.name == "receipt.pending.json" and schema in {
                    "yap-studio/stage-receipt@1", "yap-studio/stage-pending@1",
                } and digest in (output.get("old_sha256"), output.get("new_sha256")):
                    return True
        except (OSError, UnicodeError, ValueError, RecursionError):
            continue
    return False


def _checked_output(
    path: str | os.PathLike[str],
    allowed_existing_suffixes: frozenset[str] = _DEFAULT_EXISTING_SUFFIXES,
    *,
    existing_json: Literal["yap", "verify-report"] = "yap",
    owner_receipts: Sequence[str | os.PathLike[str]] = (),
    created_temporary: bool = False,
) -> tuple[str, str]:
    given, resolved = _guard_write(path)
    checked = _write_path_text(given)
    if created_temporary:
        if not os.path.isfile(checked):
            raise OutputPathError(
                "Temporary output changed type during the write.", code="unsafe_path"
            )
    elif os.path.isdir(checked):
        raise OutputPathError("Output path is an existing folder.", code="unsafe_path")
    elif os.path.exists(checked):
        if not os.path.isfile(checked):
            raise OutputPathError("Output path is not a regular file.", code="unsafe_path")
        owned = _is_owned_output(checked, owner_receipts) if owner_receipts else False
        if owner_receipts and not owned:
            raise OutputPathError(
                "Existing file is not listed in an owner receipt matching its current hash; "
                "choose a new path.", code="unsafe_path",
            )
        suffix = Path(checked).suffix.lower()
        if not owned and suffix not in allowed_existing_suffixes:
            raise OutputPathError(
                "Existing file type is not one this command replaces; choose a new path.",
                code="unsafe_path",
            )
        if (not owned and not _record_contains(_CREATED_REALPATHS, resolved, checked_path=checked)
                and not _is_yap_output(checked, suffix, existing_json)):
            raise OutputPathError(
                "Existing file is not a yap-studio output; choose a new path.", code="unsafe_path"
            )
    _record(_OUTPUT_REALPATHS, resolved)
    return given, resolved


def _validate_output(
    path: str | os.PathLike[str],
    allowed_existing_suffixes: frozenset[str] = _DEFAULT_EXISTING_SUFFIXES,
    *,
    existing_json: Literal["yap", "verify-report"] = "yap",
    owner_receipts: Sequence[str | os.PathLike[str]] = (),
    created_temporary: bool = False,
) -> Path:
    given, _ = _checked_output(
        path, allowed_existing_suffixes, existing_json=existing_json,
        owner_receipts=owner_receipts, created_temporary=created_temporary,
    )
    return Path(given)


def validate_output(
    path: str | os.PathLike[str],
    allowed_existing_suffixes: frozenset[str] = _DEFAULT_EXISTING_SUFFIXES,
    *,
    existing_json: Literal["yap", "verify-report"] = "yap",
    owner_receipts: Sequence[str | os.PathLike[str]] = (),
) -> Path:
    """Validate an output, permitting receipt-owned files with matching hashes."""
    return _validate_output(
        path, allowed_existing_suffixes, existing_json=existing_json, owner_receipts=owner_receipts
    )


def validate_output_dir(path: str | os.PathLike[str]) -> Path:
    """Validate, create, and record an output directory."""
    given, resolved = _guard_write(path)
    if os.path.exists(given) and not os.path.isdir(given):
        raise OutputPathError("Output directory is not a directory.", code="unsafe_path")
    Path(given).mkdir(parents=True, exist_ok=True)
    _record(_OUTPUT_REALPATHS, resolved)
    return Path(given)


def _discard_temporary(temporary: Path) -> None:
    if temporary.is_dir() and not temporary.is_symlink():
        shutil.rmtree(temporary)
    else:
        temporary.unlink(missing_ok=True)


@contextmanager
def atomic_output(
    path: str | os.PathLike[str],
    allowed_existing_suffixes: frozenset[str] = _DEFAULT_EXISTING_SUFFIXES,
    *,
    existing_json: Literal["yap", "verify-report"] = "yap",
    owner_receipts: Sequence[str | os.PathLike[str]] = (),
) -> Iterator[Path]:
    """Yield a temporary file and atomically replace a rechecked target."""
    owner_receipts = tuple(os.path.abspath(receipt) for receipt in owner_receipts)
    given, _ = _checked_output(
        path, allowed_existing_suffixes, existing_json=existing_json, owner_receipts=owner_receipts
    )
    target = Path(os.path.abspath(_write_path_text(given)))
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    while True:
        temporary = target.parent / f".yap_tmp_{secrets.token_hex(12)}{target.suffix}"
        try:
            descriptor = os.open(temporary, flags, 0o600)
        except FileExistsError:
            continue
        break
    try:
        os.close(descriptor)
        temporary_realpath = os.path.realpath(temporary)
        _record(_OUTPUT_REALPATHS, temporary_realpath)
        _record(_CREATED_REALPATHS, temporary_realpath)
        yield temporary
        _validate_output(temporary, created_temporary=True)
        _, resolved = _checked_output(
            target, allowed_existing_suffixes, existing_json=existing_json,
            owner_receipts=owner_receipts,
        )
        os.replace(temporary, resolved)
        _record(_CREATED_REALPATHS, resolved)
    finally:
        _discard_temporary(temporary)


def atomic_write_text(
    path: str | os.PathLike[str],
    text: str,
    allowed_existing_suffixes: frozenset[str] = _DEFAULT_EXISTING_SUFFIXES,
    *,
    existing_json: Literal["yap", "verify-report"] = "yap",
) -> Path:
    """Write UTF-8 text through an atomic output file."""
    with atomic_output(
        path, allowed_existing_suffixes, existing_json=existing_json
    ) as temporary:
        temporary.write_text(text, encoding="utf-8")
    return Path(path)
