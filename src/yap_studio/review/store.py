"""Persistent reviews and media hashes in the application data folder."""

from __future__ import annotations

import json
import os
import re
import secrets
import sys
import threading
import time
import unicodedata
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

from yap_studio import paths, receipt
from yap_studio.errors import InputError, OutputPathError, ValidationError, YapError
from yap_studio.project.lock import pid_alive
from yap_studio.review.export import comment_from_dict, comment_to_dict
from yap_studio.review.media import file_identity, stat_identity
from yap_studio.review.model import (
    Comment,
    MediaFacts,
    conform_comment,
    normalize_input,
    sorted_comments,
)
from yap_studio.review.rounds import (
    INTERRUPTED_ERROR,
    ROUNDS_SCHEMA,
    Round,
)
from yap_studio.review.rounds import (
    round_from_dict as _round_from_dict,
)
from yap_studio.review.rounds import (
    round_to_dict as _round_to_dict,
)
from yap_studio.review.series import (
    REPLY_STATUSES,
    Reply,
    Series,
    SeriesEntry,
    SeriesVersion,
    Thread,
    build_threads,
    series_from_dict,
    series_to_dict,
    validate_evidence,
    validate_reply_text,
)

_INVALID_REASON = "not a valid review file; move it aside to start a new review"
_INVALID_SERIES_REASON = "not a valid series file; move it aside to start a new series"
_INVALID_ROUNDS_REASON = "not a valid rounds file; move it aside to start new rounds"
_REVIEW_SCHEMA = "yap-studio/review@2"
_LEGACY_REVIEW_SCHEMA = "yap-studio/review@1"
_INDEX_SCHEMA = "yap-studio/media-index@1"
_SETTINGS_SCHEMA = "yap-studio/review-settings@1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMENT_ID = re.compile(r"c([1-9][0-9]*)\Z")
_MISSING = object()
_TRUST_FILE_METADATA = os.name == "posix"
_HASH_READ_ERROR = (
    "the file disappeared or could not be read while it was being hashed; "
    "try again once the export has finished"
)


def data_dir(
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    home: Path | None = None,
) -> Path:
    """Return the application data directory for the current platform."""
    env = os.environ if environ is None else environ
    system = sys.platform if platform is None else platform
    user_home = Path.home() if home is None else home
    override = env.get("YAP_STUDIO_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if system == "darwin":
        return user_home / "Library" / "Application Support" / "yap-studio"
    if system == "win32":
        return Path(env["APPDATA"]) / "yap-studio" if env.get("APPDATA") else (
            user_home / "AppData" / "Roaming" / "yap-studio"
        )
    xdg_home = env.get("XDG_DATA_HOME")
    if xdg_home and Path(xdg_home).is_absolute():
        return Path(xdg_home) / "yap-studio"
    return user_home / ".local" / "share" / "yap-studio"


class UnknownCommentError(ValidationError):
    """Report a comment id that is absent from the selected review."""

    def __init__(self, comment_id: str) -> None:
        super().__init__("comment_id", f"unknown comment: {comment_id}")


class UnknownRoundError(ValidationError):
    """Report a round absent from the selected series."""

    def __init__(self, round_id: str) -> None:
        super().__init__("round_id", f"unknown round: {round_id}")


class RoundConflictError(YapError):
    """Report a round or comment that cannot be changed in its current state."""

    def __init__(self, message: str, code: str = "round_conflict") -> None:
        super().__init__(message, error_type=code, code=code)


class CommentSentError(RoundConflictError):
    """Keep notes immutable once their send begins."""

    def __init__(self, round_id: str) -> None:
        super().__init__(
            f"This note was sent in round {round_id} and can no longer be changed. "
            "Add a new note instead.", "comment_sent",
        )


class MediaHash(NamedTuple):
    """Hash and file version verified during a media read."""

    sha256: str
    previous_sha256: str | None
    identity: tuple[int, int, int, int, int]


def round_from_dict(data: object) -> Round:
    """Extend legacy round validation to retain a written batch during send or failure."""
    if isinstance(data, dict) and data.get("status") in ("sending", "failed"):
        batch = data.get("batch")
        if batch is not None:
            if not isinstance(batch, str) or re.fullmatch(r"b[1-9][0-9]*", batch) is None:
                raise ValueError("invalid batch id")
            return replace(_round_from_dict({**data, "batch": None}), batch=batch)
    return _round_from_dict(data)


def round_to_dict(round: Round) -> dict:
    """Serialize resumable rounds through the original strict record validator."""
    if round.status in ("sending", "failed") and round.batch is not None:
        value = {**_round_to_dict(replace(round, batch=None)), "batch": round.batch}
        round_from_dict(value)
        return value
    return _round_to_dict(round)


class ReviewStore:
    """Store reviews, file hashes, and reviewer settings with atomic writes."""

    def __init__(self, root: Path, now: Callable[[], datetime] | None = None) -> None:
        self.root = Path(root)
        self._now = now if now is not None else lambda: datetime.now(UTC)
        self._lock = threading.Lock()

    def _timestamp(self, *, precise: bool = False) -> str:
        value = self._now()
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ" if precise else "%Y-%m-%dT%H:%M:%SZ")

    def _review_path(self, sha256: str) -> Path:
        if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
            raise ValidationError("sha256", "must be a lowercase SHA-256 digest")
        return self.root / "reviews" / f"{sha256}.json"

    @staticmethod
    def _read(path: Path) -> object:
        if not path.exists():
            return _MISSING
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise InputError(str(path), _INVALID_REASON) from exc

    def _storage_error(self, exc: OSError) -> OutputPathError:
        error = OutputPathError(
            f"Cannot write review data in {self.root}: {exc.strerror or exc}",
            code="review_storage",
        )
        error.suggested_action = "Set YAP_STUDIO_DATA_DIR to a writable folder."
        return error

    def _write(self, path: Path, document: object) -> None:
        payload = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
        try:
            paths.atomic_write_text(path, payload)
        except OSError as exc:
            raise self._storage_error(exc) from exc

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        """Serialize review transactions across threads and processes."""
        with self._lock:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                fd = os.open(self.root / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
            except OSError as exc:
                raise self._storage_error(exc) from exc
            locked = False
            try:
                if os.name == "nt":
                    import msvcrt

                    deadline = time.monotonic() + 30
                    while True:
                        try:
                            os.lseek(fd, 0, os.SEEK_SET)
                            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                            break
                        except OSError as exc:
                            if time.monotonic() >= deadline:
                                raise OutputPathError(
                                    "Another review session holds the lock.", code="review_locked"
                                ) from exc
                            time.sleep(0.05)
                else:
                    import fcntl

                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX)
                    except OSError as exc:
                        raise self._storage_error(exc) from exc
                locked = True
                yield
            finally:
                try:
                    if locked:
                        if os.name == "nt":
                            os.lseek(fd, 0, os.SEEK_SET)
                            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                        else:
                            fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)

    def _index(self) -> dict:
        path = self.root / "media-index.json"
        data = self._read(path)
        if data is _MISSING:
            return {}
        if (
            not isinstance(data, dict)
            or set(data) != {"schema", "entries"}
            or data["schema"] != _INDEX_SCHEMA
            or not isinstance(data["entries"], dict)
        ):
            raise InputError(str(path), _INVALID_REASON)
        entries = data["entries"]
        for key, entry in entries.items():
            if (
                not isinstance(key, str)
                or not isinstance(entry, dict)
                or set(entry) != {
                    "sha256", "size", "mtime_ns", "ctime_ns", "ino", "dev", "previous_sha256"
                }
                or not isinstance(entry["sha256"], str)
                or _SHA256.fullmatch(entry["sha256"]) is None
                or type(entry["size"]) is not int
                or entry["size"] < 0
                or type(entry["mtime_ns"]) is not int
                or type(entry["ctime_ns"]) is not int
                or type(entry["ino"]) is not int
                or type(entry["dev"]) is not int
                or (entry["previous_sha256"] is not None and (
                    not isinstance(entry["previous_sha256"], str)
                    or _SHA256.fullmatch(entry["previous_sha256"]) is None
                ))
            ):
                raise InputError(str(path), _INVALID_REASON)
        return entries

    def _review(self, sha256: str) -> dict:
        path = self._review_path(sha256)
        data = self._read(path)
        if data is _MISSING:
            return {
                "schema": _REVIEW_SCHEMA,
                "media": {},
                "next_id": 1,
                "comments": [],
            }
        try:
            if (
                not isinstance(data, dict)
                or set(data) != {"schema", "media", "next_id", "comments"}
                or data["schema"] not in (_LEGACY_REVIEW_SCHEMA, _REVIEW_SCHEMA)
                or not isinstance(data.get("media"), dict)
                or type(data.get("next_id")) is not int
                or data["next_id"] < 1
                or not isinstance(data.get("comments"), list)
            ):
                raise ValueError("invalid review document")
            self._parse_media(data["media"], sha256)
            legacy = data["schema"] == _LEGACY_REVIEW_SCHEMA
            comments = [comment_from_dict(item, legacy=legacy) for item in data["comments"]]
            ids = []
            for comment in comments:
                match = _COMMENT_ID.fullmatch(comment.id)
                if match is None:
                    raise ValueError("invalid comment id")
                ids.append(int(match.group(1)))
            if len(ids) != len(set(ids)) or any(number >= data["next_id"] for number in ids):
                raise ValueError("invalid comment sequence")
        except (TypeError, ValueError, ZeroDivisionError, KeyError, AttributeError) as exc:
            raise InputError(str(path), _INVALID_REASON) from exc
        return {
            **data,
            "schema": _REVIEW_SCHEMA,
            "comments": [comment_to_dict(comment) for comment in comments],
        }

    @staticmethod
    def _parse_media(media: dict, sha256: str) -> MediaFacts | None:
        if not media:
            return None
        if set(media) != {"name", "sha256", "duration_s", "fps", "cfr"}:
            raise ValueError("invalid media facts")
        if not isinstance(media["name"], str) or media["sha256"] != sha256:
            raise ValueError("invalid media identity")
        duration = media["duration_s"]
        if duration is not None and (
            isinstance(duration, bool) or not isinstance(duration, (int, float))
        ):
            raise ValueError("invalid duration")
        fps = Fraction(media["fps"]) if media["fps"] is not None else None
        if not isinstance(media["cfr"], bool) or (media["cfr"] and fps is None):
            raise ValueError("invalid frame rate")
        return MediaFacts(media["name"], sha256, duration, fps, media["cfr"])

    def _has_comments(self, sha256: str | None) -> bool:
        return bool(sha256 and self._review(sha256)["comments"])

    def hash_media(self, path: str | os.PathLike[str]) -> MediaHash:
        """Hash changed media and retain the last reviewed version at this path."""
        with self._write_lock():
            real = paths.validate_input(path)
            index = self._index()
            key = str(real)
            old = index.get(key)
            try:
                stat = real.stat()
            except OSError as exc:
                raise InputError(str(path), _HASH_READ_ERROR) from exc
            identity = stat_identity(stat)
            if (
                _TRUST_FILE_METADATA
                and old
                and old["size"] == stat.st_size
                and old["mtime_ns"] == stat.st_mtime_ns
                and old["ctime_ns"] == stat.st_ctime_ns
                and old["ino"] == stat.st_ino
                and old["dev"] == stat.st_dev
            ):
                sha256 = old["sha256"]
                previous = old["previous_sha256"]
            else:
                for attempt in range(3):
                    try:
                        sha256 = receipt.sha256_file(real)
                    except OSError as exc:
                        raise InputError(str(path), _HASH_READ_ERROR) from exc
                    after = file_identity(real)
                    if after is None:
                        raise InputError(str(path), _HASH_READ_ERROR)
                    if after == identity:
                        break
                    if attempt < 2:
                        try:
                            stat = real.stat()
                        except OSError as exc:
                            raise InputError(str(path), _HASH_READ_ERROR) from exc
                        identity = stat_identity(stat)
                else:
                    raise InputError(
                        str(path),
                        "the file kept changing while it was being hashed; "
                        "wait for the export to finish and try again",
                    )
                previous = old["previous_sha256"] if old else None
                if old and old["sha256"] != sha256 and self._has_comments(old["sha256"]):
                    previous = old["sha256"]
                if previous == sha256:
                    previous = None
                index[key] = {
                    "sha256": sha256,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "ctime_ns": stat.st_ctime_ns,
                    "ino": stat.st_ino,
                    "dev": stat.st_dev,
                    "previous_sha256": previous,
                }
                self._write(
                    self.root / "media-index.json",
                    {"schema": _INDEX_SCHEMA, "entries": index},
                )
            return MediaHash(sha256, previous if self._has_comments(previous) else None, identity)

    def comments(self, sha256: str) -> list[Comment]:
        """Read comments in timeline order."""
        with self._lock:
            records = self._review(sha256)["comments"]
            return sorted_comments([comment_from_dict(item) for item in records])

    def review_facts(self, sha256: str) -> MediaFacts | None:
        """Return media facts saved with a review, if any."""
        with self._lock:
            return self._parse_media(self._review(sha256)["media"], sha256)

    def add_comment(self, sha256: str, facts: MediaFacts, data: dict) -> Comment:
        """Validate and save a new comment with a never-reused id."""
        with self._write_lock():
            review = self._review(sha256)
            fields = normalize_input(data, facts)
            timestamp = self._timestamp(
                precise=(fields["request"] or {}).get("type") == "caption_fix")
            comment = Comment(
                f"c{review['next_id']}", **fields, created_at=timestamp, updated_at=timestamp
            )
            review["next_id"] += 1
            review["media"] = self._media(facts)
            review["comments"].append(comment_to_dict(comment))
            self._write(self._review_path(sha256), review)
            return comment

    def update_comment(
        self, sha256: str, facts: MediaFacts, comment_id: str, data: dict
    ) -> Comment:
        """Patch an existing comment while retaining its id and creation time."""
        with self._write_lock():
            review = self._review(sha256)
            for position, item in enumerate(review["comments"]):
                if item["id"] == comment_id:
                    self._check_comment_lock(sha256, comment_id)
                    current = conform_comment(comment_from_dict(item), facts)
                    fields = normalize_input(data, facts, partial=True, current=current)
                    timestamp = self._timestamp()
                    if (datetime.fromisoformat(timestamp)
                            <= datetime.fromisoformat(current.updated_at)):
                        timestamp = (datetime.fromisoformat(current.updated_at)
                                     + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    updated = Comment(
                        comment_id,
                        **fields,
                        created_at=current.created_at,
                        updated_at=timestamp,
                    )
                    review["comments"][position] = comment_to_dict(updated)
                    review["media"] = self._media(facts)
                    self._write(self._review_path(sha256), review)
                    return updated
            raise UnknownCommentError(comment_id)

    def delete_comment(self, sha256: str, comment_id: str) -> None:
        """Remove a comment without resetting the next id."""
        with self._write_lock():
            review = self._review(sha256)
            for position, item in enumerate(review["comments"]):
                if item["id"] == comment_id:
                    self._check_comment_lock(sha256, comment_id)
                    del review["comments"][position]
                    self._write(self._review_path(sha256), review)
                    return
            raise UnknownCommentError(comment_id)

    def _series_path(self, series_id: str) -> Path:
        if not isinstance(series_id, str) or _SHA256.fullmatch(series_id) is None:
            raise ValidationError("series_id", "must be a lowercase SHA-256 digest")
        return self.root / "series" / f"{series_id}.json"

    def _series_document(self, series_id: str) -> tuple[Series, dict]:
        path = self._series_path(series_id)
        try:
            data = self._read(path)
        except InputError as exc:
            raise InputError(str(path), _INVALID_SERIES_REASON) from exc
        if data is _MISSING:
            raise InputError(str(path), "series does not exist")
        try:
            series = series_from_dict(data)
            if series.id != series_id:
                raise ValueError("series id must match filename")
        except (TypeError, ValueError, KeyError, AttributeError, OverflowError) as exc:
            raise InputError(str(path), _INVALID_SERIES_REASON) from exc
        return series, data

    def series_and_paths_containing(
        self, sha256: str,
    ) -> tuple[Series | None, dict[str, list[Path]]]:
        """Find a saved series and indexed paths for its versions."""
        with self._write_lock():
            found = None
            for path in sorted((self.root / "series").glob("*.json")):
                if _SHA256.fullmatch(path.stem) is None:
                    raise InputError(str(path), _INVALID_SERIES_REASON)
                series, _ = self._series_document(path.stem)
                if any(item.sha256 == sha256 for item in series.versions):
                    found = series
            if found is None:
                return None, {}
            indexed: dict[str, list[Path]] = {}
            for path, item in self._index().items():
                indexed.setdefault(item["sha256"], []).append(Path(path))
            return found, indexed

    def open_series(self, entries: Sequence[SeriesEntry]) -> Series:
        """Create or extend the single series whose history matches these entries."""
        entries = tuple(entries)
        if not entries:
            raise ValidationError("entries", "must contain at least one video")
        with self._write_lock():
            timestamp = self._timestamp()
            versions = []
            seen: dict[str, str] = {}
            for number, entry in enumerate(entries, 1):
                if not isinstance(entry, SeriesEntry):
                    raise ValidationError("entries", "must contain SeriesEntry records")
                self._review_path(entry.sha256)
                if entry.sha256 in seen:
                    raise ValidationError(
                        "entries", f"{seen[entry.sha256]} and {entry.name} have identical contents"
                    )
                seen[entry.sha256] = entry.name
                versions.append(SeriesVersion(number, entry.sha256, entry.name, entry.version_id,
                                              entry.project, timestamp))
            proposed = Series(entries[0].sha256, entries[-1].name, tuple(versions), ())
            try:
                series_from_dict(series_to_dict(proposed))
            except (TypeError, ValueError) as exc:
                raise ValidationError("entries", str(exc)) from exc
            matches = []
            for path in sorted((self.root / "series").glob("*.json")):
                if _SHA256.fullmatch(path.stem) is None:
                    raise InputError(str(path), _INVALID_SERIES_REASON)
                existing, document = self._series_document(path.stem)
                if any(item.sha256 in seen for item in existing.versions):
                    matches.append((existing, document))
            if len(matches) > 1:
                raise ValidationError("entries", "videos belong to more than one series")
            if not matches:
                self._write(self._series_path(proposed.id), series_to_dict(proposed))
                return proposed
            existing, document = matches[0]
            if len(entries) < len(existing.versions):
                missing = next(item for item in existing.versions if item.sha256 not in seen)
                raise ValidationError(
                    "entries", f"missing version {missing.number}: {missing.name}; "
                    f"expected order: {', '.join(item.name for item in existing.versions)}; "
                    f"given: {', '.join(item.name for item in entries)}"
                )
            for saved, entry in zip(existing.versions, entries, strict=False):
                if saved.sha256 != entry.sha256:
                    raise ValidationError(
                        "entries", "expected order: "
                        f"{', '.join(v.name for v in existing.versions)}; "
                        f"given: {', '.join(v.name for v in entries)}"
                    )
                for field in ("version_id", "project"):
                    stored, supplied = getattr(saved, field), getattr(entry, field)
                    if stored is not None and supplied is not None and stored != supplied:
                        raise ValidationError("entries", f"{saved.name}: {field} does not match")
            extra = tuple(versions[len(existing.versions):])
            if not extra:
                return existing
            extended = Series(existing.id, entries[-1].name, existing.versions + extra,
                              existing.replies)
            self._write(self._series_path(existing.id),
                        series_to_dict(extended, next_reply_id=document["next_reply_id"]))
            return extended

    def series(self, series_id: str) -> Series:
        """Read a strict series record without modifying stored files."""
        with self._lock:
            return self._series_document(series_id)[0]

    def _link_document(self, series_id: str) -> tuple[Path, dict | None]:
        self._series_path(series_id)
        path = self.root / "links" / f"{series_id}.json"
        data = self._read(path)
        if data is _MISSING:
            return path, None
        if (not isinstance(data, dict) or set(data) != {"schema", "series", "token", "port"}
                or data["schema"] != "yap-studio/review-link@1" or data["series"] != series_id
                or not isinstance(data["token"], str)
                or re.fullmatch(r"[A-Za-z0-9_-]{20,}", data["token"]) is None
                or (data["port"] is not None and (
                    type(data["port"]) is not int or not 1 <= data["port"] <= 65535
                ))):
            raise InputError(str(path), "not a valid review link; "
                             "move it aside to start a new link")
        return path, data

    def series_link(self, series_id: str) -> dict:
        """Create or read the series token and preferred port under the store lock."""
        with self._write_lock():
            self._series_document(series_id)
            path, data = self._link_document(series_id)
            if data is None:
                data = {"schema": "yap-studio/review-link@1", "series": series_id,
                        "token": secrets.token_urlsafe(24), "port": None}
                self._write(path, data)
            try:
                path.chmod(0o600)
            except OSError as exc:
                raise self._storage_error(exc) from exc
            return dict(data)

    def save_link_port(self, series_id: str, port: int) -> None:
        """Persist the bound port while preserving the series's token."""
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValidationError("port", "must be from 1 to 65535")
        with self._write_lock():
            path, data = self._link_document(series_id)
            if data is None:
                self._series_document(series_id)
                data = {"schema": "yap-studio/review-link@1", "series": series_id,
                        "token": secrets.token_urlsafe(24), "port": None}
            self._write(path, {**data, "port": port})
            try:
                path.chmod(0o600)
            except OSError as exc:
                raise self._storage_error(exc) from exc

    def _rounds_path(self, series_id: str) -> Path:
        self._series_path(series_id)
        return self.root / "rounds" / f"{series_id}.json"

    def _rounds_document(self, series_id: str) -> tuple[list[Round], dict]:
        path = self._rounds_path(series_id)
        try:
            data = self._read(path)
            if data is _MISSING:
                return [], {"schema": ROUNDS_SCHEMA, "series": series_id, "next_id": 1,
                            "rounds": []}
            if (not isinstance(data, dict)
                    or set(data) != {"schema", "series", "next_id", "rounds"}
                    or data["schema"] != ROUNDS_SCHEMA or data["series"] != series_id
                    or type(data["next_id"]) is not int or data["next_id"] < 1
                    or not isinstance(data["rounds"], list)):
                raise ValueError("invalid rounds document")
            rounds = [round_from_dict(item) for item in data["rounds"]]
            ids, locked, locked_answers = set(), set(), set()
            for round in rounds:
                if round.id in ids or int(round.id[1:]) >= data["next_id"]:
                    raise ValueError("invalid round id sequence")
                ids.add(round.id)
                if round.status in ("sending", "sent"):
                    for ref in round.comments:
                        key = (round.version, ref["id"])
                        if key in locked:
                            raise ValueError("comment locked by more than one round")
                        locked.add(key)
                    for answer in round.answers:
                        key = (answer["batch"], answer["comment"]["version"],
                               answer["comment"]["id"])
                        if key in locked_answers:
                            raise ValueError("answer locked by more than one round")
                        locked_answers.add(key)
            return rounds, data
        except (InputError, TypeError, ValueError, KeyError, AttributeError, OverflowError) as exc:
            raise InputError(str(path), _INVALID_ROUNDS_REASON) from exc

    def _save_rounds(self, series_id: str, rounds: Sequence[Round], document: dict) -> None:
        self._write(self._rounds_path(series_id),
                    {**document, "rounds": [round_to_dict(item) for item in rounds]})

    @staticmethod
    def _find_round(rounds: Sequence[Round], round_id: str) -> Round:
        for round in rounds:
            if round.id == round_id:
                return round
        raise UnknownRoundError(round_id)

    def _check_comment_lock(self, sha256: str, comment_id: str) -> None:
        for path in sorted((self.root / "series").glob("*.json")):
            series, _ = self._series_document(path.stem)
            version = next((item.number for item in series.versions if item.sha256 == sha256), None)
            if version is None:
                continue
            for round in self._rounds_document(series.id)[0]:
                if (round.version == version and round.status in ("sending", "sent")
                        and any(ref["id"] == comment_id for ref in round.comments)):
                    raise CommentSentError(round.id)

    def rounds(self, series_id: str) -> list[Round]:
        """Read strict rounds while holding the shared transaction lock."""
        with self._write_lock():
            return self._rounds_document(series_id)[0]

    def start_round(
        self, series_id: str, version: int, comments: Sequence[Comment],
        answers: Sequence[dict] = (),
        *, owner: dict | None = None,
    ) -> Round:
        """Capture current note revisions and reserve a never-reused round id."""
        comments = tuple(comments)
        answers = tuple(answers)
        if not comments and not answers:
            raise ValidationError("comments", "no draft notes to send and no answers")
        with self._write_lock():
            series, _ = self._series_document(series_id)
            if type(version) is not int or not 1 <= version <= len(series.versions):
                raise ValidationError("version", "must name a version in the series")
            rounds, document = self._rounds_document(series_id)
            sha256 = series.versions[version - 1].sha256
            saved = {item["id"]: item for item in self._review(sha256)["comments"]}
            refs, ids = [], set()
            for comment in comments:
                if not isinstance(comment, Comment) or comment.id in ids:
                    raise ValidationError("comments", "must contain distinct comment records")
                ids.add(comment.id)
                if comment.id not in saved:
                    raise UnknownCommentError(comment.id)
                self._check_comment_lock(sha256, comment.id)
                if comment_to_dict(comment) != saved[comment.id]:
                    raise RoundConflictError(
                        "These notes changed. Reload the review and send again.",
                    )
                refs.append({"id": comment.id, "revision": comment.updated_at})
            timestamp = self._timestamp()
            round = Round(f"n{document['next_id']}", version, tuple(refs), "sending", None,
                          None, None, 1, timestamp, timestamp, answers, owner)
            try:
                round = round_from_dict(round_to_dict(round))
            except (TypeError, ValueError) as exc:
                raise ValidationError("answers", str(exc)) from exc
            if any(answer in other.answers for answer in round.answers for other in rounds
                   if other.status in ("sending", "sent")):
                raise RoundConflictError("These answers already belong to another round.")
            document["next_id"] += 1
            self._save_rounds(series_id, [*rounds, round], document)
            return round

    def update_round(
        self, series_id: str, round_id: str, *, status: str | None = None,
        step: object = _MISSING, batch: object = _MISSING, error: object = _MISSING,
        instance: str | None = None,
    ) -> Round:
        """Atomically replace send progress or a terminal result after strict validation."""
        with self._write_lock():
            rounds, document = self._rounds_document(series_id)
            current = self._find_round(rounds, round_id)
            if current.owner is not None and current.owner["instance"] != instance:
                raise RoundConflictError("Only this round's owner can update it.")
            fields = {key: value for key, value in (("step", step), ("batch", batch),
                                                   ("error", error)) if value is not _MISSING}
            if status is not None:
                fields["status"] = status
            fields["updated_at"] = max(self._timestamp(), current.updated_at)
            try:
                updated = round_from_dict(round_to_dict(replace(current, **fields)))
            except (TypeError, ValueError) as exc:
                raise ValidationError("round", str(exc)) from exc
            if current.status != "sending" or updated.status not in ("sending", "sent", "failed"):
                raise RoundConflictError("Only a sending round can be updated.")
            self._save_rounds(
                series_id, [updated if item.id == round_id else item for item in rounds], document,
            )
            return updated

    def retry_round(
        self, series_id: str, round_id: str, *, owner: dict | None = None,
    ) -> Round:
        """Retry a failed round while preserving its id and exact comment revisions."""
        with self._write_lock():
            rounds, document = self._rounds_document(series_id)
            current = self._find_round(rounds, round_id)
            if current.status != "failed":
                raise RoundConflictError("Only a failed round can be retried.")
            for round in rounds:
                if round.status in ("sending", "sent"):
                    overlap = (round.version == current.version and
                               {ref["id"] for ref in round.comments}
                               & {ref["id"] for ref in current.comments})
                    if overlap or any(answer in round.answers for answer in current.answers):
                        raise RoundConflictError(
                            "These notes already belong to another sent round.",
                        )
            updated = replace(current, status="sending", step=None, error=None,
                              owner=owner,
                              attempts=current.attempts + 1,
                              updated_at=max(self._timestamp(), current.updated_at))
            self._save_rounds(
                series_id, [updated if item.id == round_id else item for item in rounds], document,
            )
            return updated

    def recover_rounds(self, series_id: str, *, instance: str | None = None) -> int:
        """Fail interrupted sends once when a series starts serving again."""
        with self._write_lock():
            rounds, document = self._rounds_document(series_id)
            interrupted = {item.id for item in rounds if item.status == "sending" and (
                item.owner is None or not pid_alive(item.owner["pid"])
                or item.owner["instance"] == instance
            )}
            count = len(interrupted)
            if count:
                recovered = [replace(item, status="failed", step=None,
                                     error=dict(INTERRUPTED_ERROR),
                                     updated_at=max(self._timestamp(), item.updated_at))
                             if item.id in interrupted else item for item in rounds]
                self._save_rounds(series_id, recovered, document)
            return count

    def add_reply(
        self,
        series_id: str,
        *,
        comment_version: int,
        comment_id: str,
        version: int,
        status: str,
        text: str,
        evidence: dict | None = None,
    ) -> Reply:
        """Add or replace a later-version reply while retaining stable reply ids."""
        if status not in REPLY_STATUSES:
            raise ValidationError("status", f"must be one of {', '.join(REPLY_STATUSES)}")
        clean_text = validate_reply_text(text)
        clean_evidence = validate_evidence(evidence)
        with self._write_lock():
            series, document = self._series_document(series_id)
            if (type(comment_version) is not int or type(version) is not int
                    or not 1 <= comment_version < version <= len(series.versions)):
                raise ValidationError("version", "must follow comment_version within the series")
            sha = series.versions[comment_version - 1].sha256
            if not any(item["id"] == comment_id for item in self._review(sha)["comments"]):
                raise UnknownCommentError(comment_id)
            position = next((i for i, item in enumerate(series.replies)
                             if (item.comment_version, item.comment_id, item.version)
                             == (comment_version, comment_id, version)), None)
            current = series.replies[position] if position is not None else None
            timestamp = self._timestamp()
            reply = Reply(current.id if current else f"r{document['next_reply_id']}",
                          comment_version, comment_id, version, status, clean_text, clean_evidence,
                          current.created_at if current else timestamp, timestamp)
            replies = list(series.replies)
            if position is None:
                replies.append(reply)
                document["next_reply_id"] += 1
            else:
                replies[position] = reply
            updated = Series(series.id, series.name, series.versions, tuple(replies))
            self._write(self._series_path(series_id),
                        series_to_dict(updated, next_reply_id=document["next_reply_id"]))
            return reply

    def threads(self, series_id: str, as_of: int) -> list[Thread]:
        """Read prior comments and their replies as of a selected series version."""
        with self._lock:
            series = self._series_document(series_id)[0]
            if type(as_of) is not int or not 1 <= as_of <= len(series.versions):
                raise ValidationError("as_of", "must name a version in the series")
            comments = {
                item.number: [comment_from_dict(raw)
                              for raw in self._review(item.sha256)["comments"]]
                for item in series.versions if item.number < as_of
            }
            return build_threads(series, comments, as_of)

    @staticmethod
    def _media(facts: MediaFacts) -> dict:
        return {
            "name": facts.name,
            "sha256": facts.sha256,
            "duration_s": facts.duration_s,
            "fps": f"{facts.fps.numerator}/{facts.fps.denominator}" if facts.fps else None,
            "cfr": facts.cfr,
        }

    def reviewer(self) -> str | None:
        """Read the saved reviewer name, if present."""
        with self._lock:
            return self._reviewer_unlocked()

    def set_reviewer(self, name: str | None) -> None:
        """Save a short printable reviewer name, or clear it."""
        if name is not None and not isinstance(name, str):
            raise ValidationError("reviewer", "must be a string or null")
        clean = name.strip() if name is not None else None
        if clean == "":
            clean = None
        if clean is not None and (
            len(clean) > 100 or any(unicodedata.category(char) == "Cc" for char in clean)
        ):
            raise ValidationError(
                "reviewer", "must be 1 to 100 characters with no control characters"
            )
        with self._write_lock():
            self._reviewer_unlocked()
            self._write(
                self.root / "settings.json",
                {"schema": _SETTINGS_SCHEMA, "reviewer": clean},
            )

    def _reviewer_unlocked(self) -> str | None:
        path = self.root / "settings.json"
        data = self._read(path)
        if data is _MISSING:
            return None
        if (
            not isinstance(data, dict)
            or set(data) != {"schema", "reviewer"}
            or data["schema"] != _SETTINGS_SCHEMA
            or (
                data["reviewer"] is not None and not isinstance(data["reviewer"], str)
            )
        ):
            raise InputError(str(path), _INVALID_REASON)
        return data["reviewer"]
