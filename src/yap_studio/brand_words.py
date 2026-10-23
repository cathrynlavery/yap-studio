"""Creator vocabulary stored outside projects."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from yap_studio.errors import ValidationError
from yap_studio.style.library import config_root

SCHEMA = "yap-studio/brand-words@1"


def _target() -> Path:
    return config_root() / "brand" / "words.json"


def _validated(words: Sequence[str]) -> list[str]:
    unique: dict[str, str] = {}
    for word in words:
        if (not isinstance(word, str) or not 1 <= len(word) <= 60
                or not word.strip() or any(char in word for char in "\r\n")):
            raise ValidationError("brand words",
                                  "use entries of 1 to 60 characters without newlines")
        unique.pop(word.casefold(), None)
        unique[word.casefold()] = word
    if len(unique) > 500:
        raise ValidationError("brand words", "use at most 500 entries")
    return list(unique.values())


def list_words() -> list[str]:
    """Read the creator's spelling list in import order."""
    target = _target()
    if not target.exists():
        return []
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or data.get("schema") != SCHEMA
                or not isinstance(data.get("words"), list)):
            raise ValueError("unsupported vocabulary schema")
        words = _validated(data["words"])
        if words != data["words"]:
            raise ValueError("duplicate vocabulary entries")
        return words
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise ValidationError("brand words", f"cannot read vocabulary: {exc}") from exc


@contextmanager
def _lock() -> Iterator[None]:
    import fcntl

    folder = _target().parent
    try:
        folder.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(folder / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        raise ValidationError("brand words", f"cannot lock vocabulary: {exc}") from exc
    locked = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            locked = True
        except OSError as exc:
            raise ValidationError("brand words", f"cannot lock vocabulary: {exc}") from exc
        yield
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _write(words: Sequence[str]) -> list[str]:
    validated = _validated(words)
    target = _target()
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=".words_", dir=target.parent)
    except OSError as exc:
        raise ValidationError("brand words", f"cannot write vocabulary: {exc}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"schema": SCHEMA, "words": validated}, stream, indent=2,
                      ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, target)
    except OSError as exc:
        raise ValidationError("brand words", f"cannot write vocabulary: {exc}") from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return validated


def add_words(words: Sequence[str]) -> list[str]:
    """Add entries, keeping the newest case-insensitive spelling."""
    with _lock():
        return _write([*list_words(), *words])


def remove_words(words: Sequence[str]) -> list[str]:
    """Remove entries by spelling without regard to case."""
    with _lock():
        unwanted = {word.casefold() for word in _validated(words)}
        return _write([word for word in list_words() if word.casefold() not in unwanted])


def set_words(words: Sequence[str]) -> list[str]:
    """Replace the whole vocabulary after validation."""
    with _lock():
        return _write(words)


def read_words_file(file: Path) -> list[str]:
    """Read one entry per line, ignoring comments and blank lines."""
    try:
        return [line.strip() for line in file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    except (OSError, UnicodeError) as exc:
        raise ValidationError("brand words", f"cannot read {file}: {exc}") from exc


def set_words_file(file: Path) -> list[str]:
    """Replace spellings with entries from a text file."""
    return set_words(read_words_file(file))
