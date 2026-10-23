"""A local catalog of facts about a creator's own media."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from yap_studio import ffmpeg, hdr, receipt
from yap_studio import paths as guarded_paths
from yap_studio.errors import InputError, OutputPathError, ValidationError, YapError
from yap_studio.media.captured import capture_time
from yap_studio.review.media import file_identity, stat_identity
from yap_studio.style.library import config_root

CATALOG_SCHEMA = "yap-studio/media-catalog@1"
_BATCH_FILES = 25
_INVALID = "not a valid media catalog; move it aside to start a new one"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ITEM_ID = re.compile(r"m([1-9][0-9]*)\Z")
_DECISION_ID = re.compile(r"d([1-9][0-9]*)\Z")
_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
_FPS = re.compile(r"[1-9][0-9]*/[1-9][0-9]*\Z")
_VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".mkv", ".webm", ".avi", ".mts", ".m2ts", ".3gp"}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff"}
_MEDIA_TYPES = {"clip", "photo", "screenshot", "screen_recording"}
_ITEM_KEYS = {
    "id", "sha256", "bytes", "kind", "media_type", "media_type_by", "format", "codec",
    "width", "height", "rotation", "orientation", "duration_s", "fps", "variable_frame_rate",
    "color_transfer", "hdr", "has_audio", "camera", "provenance", "locations", "added_at",
    "description", "decisions", "next_decision", "sheet", "speech",
}
_CAMERA_KEYS = {"make", "model", "software", "lens", "captured_at", "container_time"}
_INDEX_KEYS = {"sha256", "size", "mtime_ns", "ctime_ns", "ino", "dev"}
_WRITE_LOCK = threading.Lock()
T = TypeVar("T")


class UnknownMediaError(YapError):
    """Report an id absent from the creator's media catalog."""

    def __init__(self, item_id: str) -> None:
        super().__init__(
            f"Unknown media id: {item_id}",
            error_type="input_error",
            code="unknown_media",
            suggested_action="Run yap-studio media list to see catalog ids.",
        )


def catalog_path() -> Path:
    """Return the creator's catalog path without creating it."""
    return config_root() / "assets" / "catalog.json"


def _keys(value: object, expected: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("invalid object fields")
    return value


def _string(value: object, *, nullable: bool = False) -> None:
    if not isinstance(value, str) and not (nullable and value is None):
        raise ValueError("invalid string")


def _integer(value: object, minimum: int | None = None) -> None:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError("invalid integer")


def _number(value: object, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        raise ValueError("invalid finite number")


def _enum(value: object, choices: set) -> None:
    if value not in choices:
        raise ValueError("invalid enum")


def _boolean(value: object, *, nullable: bool = False) -> None:
    if type(value) is not bool and not (nullable and value is None):
        raise ValueError("invalid boolean")


def _digest(value: object) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError("invalid hash")


def _timestamp(value: object, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or _TIMESTAMP.fullmatch(value) is None:
        raise ValueError("invalid UTC timestamp")
    datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def _absolute(value: object) -> None:
    _string(value)
    if not Path(value).is_absolute() or "\x00" in value:
        raise ValueError("invalid absolute path")


def _sequence(value: object) -> list:
    if not isinstance(value, list):
        raise ValueError("invalid list")
    return value


def _id_number(value: object, pattern: re.Pattern, next_id: int) -> int:
    if not isinstance(value, str) or (match := pattern.fullmatch(value)) is None:
        raise ValueError("invalid id")
    number = int(match.group(1))
    if number >= next_id:
        raise ValueError("invalid id sequence")
    return number


def _validate_enrichment(item: dict) -> None:
    description = item["description"]
    if description is not None:
        _keys(description, {"text", "tags", "by", "at", "revision"})
        _string(description["text"], nullable=True)
        for tag in _sequence(description["tags"]):
            _string(tag)
        _string(description["by"])
        _timestamp(description["at"])
        _integer(description["revision"], 1)

    _integer(item["next_decision"], 1)
    decision_ids = set()
    for decision in _sequence(item["decisions"]):
        _keys(decision, {
            "id", "verdict", "start_s", "end_s", "reason", "by", "at",
            "revoked_at", "revoked_by",
        })
        number = _id_number(decision["id"], _DECISION_ID, item["next_decision"])
        if number in decision_ids:
            raise ValueError("duplicate decision id")
        decision_ids.add(number)
        _enum(decision["verdict"], {"approved", "rejected"})
        _number(decision["start_s"], nullable=True)
        _number(decision["end_s"], nullable=True)
        _string(decision["reason"])
        _string(decision["by"])
        _timestamp(decision["at"])
        _timestamp(decision["revoked_at"], nullable=True)
        _string(decision["revoked_by"], nullable=True)

    sheet = item["sheet"]
    if sheet is not None:
        _keys(sheet, {"path", "sha256", "width", "height", "frames", "colour_converted", "at"})
        _string(sheet["path"])
        _digest(sheet["sha256"])
        _integer(sheet["width"], 1)
        _integer(sheet["height"], 1)
        _boolean(sheet["colour_converted"], nullable=True)
        _timestamp(sheet["at"])
        for frame in _sequence(sheet["frames"]):
            _keys(frame, {"index", "time_s", "column", "row"})
            for key in ("index", "column", "row"):
                _integer(frame[key], 0)
            _number(frame["time_s"], nullable=True)

    speech = item["speech"]
    if speech is not None:
        _keys(speech, {"status", "engine", "model", "words", "text", "path", "message", "at"})
        _enum(speech["status"], {
            "words", "no_speech", "no_audio", "no_engine", "skipped", "failed",
        })
        for key in ("engine", "model", "path", "message"):
            _string(speech[key], nullable=True)
        _integer(speech["words"], 0)
        _string(speech["text"])
        _timestamp(speech["at"])


def _validate_item(item: object) -> dict:
    optional = {"stock"} if isinstance(item, dict) and "stock" in item else set()
    if isinstance(item, dict) and "generated" in item:
        optional.add("generated")
    item = _keys(item, _ITEM_KEYS | optional)
    if "stock" in item:
        from yap_studio.stock.metadata import validate_stock

        validate_stock(item["stock"])
    if "generated" in item:
        from yap_studio.pictures.metadata import validate_generated

        validate_generated(item["generated"])
    _digest(item["sha256"])
    _integer(item["bytes"], 0)
    _enum(item["kind"], {"video", "image"})
    _enum(item["media_type"], _MEDIA_TYPES)
    _enum(item["media_type_by"], {"rule", "agent"})
    for key in ("format", "codec", "color_transfer"):
        _string(item[key], nullable=True)
    _integer(item["width"], 1)
    _integer(item["height"], 1)
    _integer(item["rotation"])
    _enum(item["rotation"], {0, 90, 180, 270})
    _enum(item["orientation"], {"vertical", "horizontal", "square"})
    _boolean(item["variable_frame_rate"])
    _boolean(item["has_audio"])
    _enum(item["hdr"], {"hlg", "pq", None})
    if item["kind"] == "image":
        if item["duration_s"] is not None or item["fps"] is not None:
            raise ValueError("invalid still timing")
        if item["variable_frame_rate"]:
            raise ValueError("invalid still frame rate")
    else:
        _number(item["duration_s"])
        if not isinstance(item["fps"], str) or _FPS.fullmatch(item["fps"]) is None:
            raise ValueError("invalid frame rate")
    camera = item["camera"]
    if camera is not None:
        _keys(camera, _CAMERA_KEYS)
        for value in camera.values():
            _string(value, nullable=True)
            if value is not None and (not value or value != value.strip() or len(value) > 200):
                raise ValueError("invalid camera tag")
        if all(value is None for value in camera.values()):
            raise ValueError("empty camera metadata")
    provenance = _keys(item["provenance"], {"kind", "source", "license", "url"})
    _enum(provenance["kind"], {"own", "licensed", "generated", None})
    for key in ("source", "license", "url"):
        _string(provenance[key], nullable=True)
    for location in _sequence(item["locations"]):
        _keys(location, {"path", "added_at", "seen_at"})
        _absolute(location["path"])
        _timestamp(location["added_at"])
        _timestamp(location["seen_at"])
    _timestamp(item["added_at"])
    _validate_enrichment(item)
    return item


def validate_catalog(data: object, path: Path) -> dict:
    """Validate the complete catalog, including fields reserved for agent updates."""
    try:
        document = _keys(data, {"schema", "next_id", "items", "index"})
        if document["schema"] != CATALOG_SCHEMA:
            raise ValueError("unknown catalog schema")
        _integer(document["next_id"], 1)
        ids, hashes = set(), set()
        for item in _sequence(document["items"]):
            item = _validate_item(item)
            number = _id_number(item["id"], _ITEM_ID, document["next_id"])
            if number in ids or item["sha256"] in hashes:
                raise ValueError("duplicate media identity")
            ids.add(number)
            hashes.add(item["sha256"])
        if not isinstance(document["index"], dict):
            raise ValueError("invalid media index")
        for key, entry in document["index"].items():
            _absolute(key)
            _keys(entry, _INDEX_KEYS)
            _digest(entry["sha256"])
            if entry["sha256"] not in hashes:
                raise ValueError("unknown indexed hash")
            for field in ("size", "mtime_ns", "ctime_ns", "ino", "dev"):
                _integer(entry[field], 0 if field == "size" else None)
        return document
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        raise InputError(str(path), _INVALID) from exc


def load_catalog(path: Path | None = None) -> dict:
    """Read a validated catalog, or an empty document if the file is missing."""
    target = catalog_path() if path is None else Path(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"schema": CATALOG_SCHEMA, "next_id": 1, "items": [], "index": {}}
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise InputError(str(target), _INVALID) from exc
    return validate_catalog(data, target)


def _storage_error(root: Path, exc: OSError) -> OutputPathError:
    error = OutputPathError(
        f"Cannot write media catalog in {root}: {exc.strerror or exc}", code="media_storage"
    )
    error.suggested_action = "Set YAP_STUDIO_CONFIG to a writable folder."
    return error


@contextmanager
def _write_lock(root: Path) -> Iterator[None]:
    with _WRITE_LOCK:
        try:
            guarded_paths.validate_output_dir(root)
            lock_path = root / ".lock"
            if lock_path.is_symlink():
                raise OutputPathError(
                    "Media catalog lock must not be a symlink.", code="unsafe_path"
                )
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        except OSError as exc:
            raise _storage_error(root, exc) from exc
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
                                "Another media catalog session holds the lock.", code="media_locked"
                            ) from exc
                        time.sleep(0.05)
            else:
                import fcntl

                try:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                except OSError as exc:
                    raise _storage_error(root, exc) from exc
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


def update_catalog(mutate: Callable[[dict], T]) -> T:
    """Apply a mutation under the write lock and atomically save a validated document."""
    target = catalog_path()
    with guarded_paths.operation(), _write_lock(target.parent):
        document = load_catalog(target)
        result = mutate(document)
        validate_catalog(document, target)
        try:
            guarded_paths.atomic_write_text(
                target, json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            )
        except OSError as exc:
            raise _storage_error(target.parent, exc) from exc
        return result


def list_items(*, media_type: str | None = None) -> list[dict]:
    """Return items in numeric id order, optionally filtered by media type."""
    if media_type is not None and media_type not in _MEDIA_TYPES:
        raise ValidationError("media_type", "choose clip, photo, screenshot, or screen_recording")
    items = load_catalog()["items"]
    return sorted(
        (item for item in items if media_type is None or item["media_type"] == media_type),
        key=lambda item: int(item["id"][1:]),
    )


def get_item(item_id: str) -> dict:
    """Return the full stored item, raising UnknownMediaError for an unknown id."""
    for item in load_catalog()["items"]:
        if item["id"] == item_id:
            return item
    raise UnknownMediaError(item_id)


def _scan(
    arguments: Sequence[str | os.PathLike[str]],
) -> tuple[list[Path], list[dict], list[str]]:
    roots = []
    for argument in arguments:
        root = Path(argument).expanduser()
        if not root.exists():
            raise InputError(os.fspath(argument), "file or folder not found")
        roots.append(root)
    if not roots:
        raise ValidationError("paths", "provide at least one file or folder")
    config = config_root().resolve()
    candidates, skipped = set(), []
    unreadable = set()

    def cannot_list(exc: OSError) -> None:
        folder = str(Path(exc.filename).absolute())
        skipped.append({"path": folder, "reason": f"cannot list folder: {exc.strerror or exc}"})
        unreadable.add(folder)

    def consider(path: Path, *, walking: bool) -> None:
        if path.name.startswith(".") or path.suffix.lower() not in (
            _VIDEO_EXTENSIONS | _IMAGE_EXTENSIONS
        ) or path.resolve().is_relative_to(config):
            return
        if walking and path.is_symlink():
            skipped.append({"path": str(path.absolute()), "reason": "symbolic link"})
        else:
            candidates.add(path.resolve())

    for root in roots:
        if root.name.startswith(".") or root.resolve().is_relative_to(config):
            continue
        if root.is_dir():
            for directory, folders, files in os.walk(root, followlinks=False, onerror=cannot_list):
                folders[:] = sorted(
                    name for name in folders
                    if not name.startswith(".")
                    and not (Path(directory) / name).resolve().is_relative_to(config)
                )
                for name in sorted(files):
                    consider(Path(directory) / name, walking=True)
        else:
            consider(root, walking=False)
    return sorted(candidates), skipped, sorted(unreadable)


def _hash_media(path: Path, entry: dict | None) -> tuple[str, tuple[int, int, int, int, int]]:
    identity = stat_identity(path.stat())
    if os.name == "posix" and entry and identity == (
        entry["dev"], entry["ino"], entry["size"], entry["mtime_ns"], entry["ctime_ns"],
    ):
        return entry["sha256"], identity
    for attempt in range(3):
        digest = receipt.sha256_file(path)
        after = file_identity(path)
        if after is None:
            raise InputError(str(path), "file disappeared while being hashed; try again")
        if after == identity:
            return digest, identity
        if attempt < 2:
            identity = stat_identity(path.stat())
    raise InputError(str(path), "file kept changing while being hashed; try again once it is ready")


def _probe_json(path: Path, options: list[str]) -> dict:
    result = ffmpeg.run(
        ["ffprobe", "-v", "error", *options, "-of", "json", str(path)],
        timeout_s=ffmpeg.PROBE_TIMEOUT_S,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError("invalid probe output")
    return payload


def _tile_grid(payload: dict) -> tuple[int, int, int] | None:
    """Read upright HEIF dimensions and rotation from a Tile Grid stream group."""
    for group in payload.get("stream_groups", []):
        if group.get("type") != "Tile Grid" or not group.get("components"):
            continue
        component = group["components"][0]
        width, height = component.get("width"), component.get("height")
        if type(width) is not int or type(height) is not int or min(width, height) <= 0:
            continue
        side_data = [
            entry for entry in (
                component.get("side_data_list", []) + group.get("side_data_list", [])
            ) if entry.get("side_data_type") == "Display Matrix"
        ]
        rotation = ffmpeg._rotation({"side_data_list": side_data})
        if rotation in {90, 270}:
            width, height = height, width
        return width, height, rotation
    return None


def heif_grid(path: Path, *, commands: list[tuple[str, ...]] | None = None,
              strict: bool = False) -> tuple[int, int, int] | None:
    """Return a HEIF tile grid's upright size, or None for a single image."""
    if path.suffix.lower() not in {".heic", ".heif"}:
        return None
    try:
        if commands is not None:
            commands.append((ffmpeg.find_tool("ffprobe"), "-hide_banner", "-v", "error",
                             "-show_stream_groups", "-of", "json", str(path)))
        return _tile_grid(_probe_json(path, ["-show_stream_groups"]))
    except (YapError, OSError, ValueError, TypeError):
        if strict:
            raise
        return None


def _camera(path: Path, info: ffmpeg.MediaInfo, *, image: bool,
            sha256: str | None = None) -> dict | None:
    sources = [info.format_tags or {}]
    try:
        payload = _probe_json(path, ["-select_streams", "v:0", "-show_entries", "stream_tags"])
        streams = payload.get("streams", [])
        if streams:
            sources.append(streams[0].get("tags", {}))
    except (YapError, OSError, ValueError, TypeError):
        pass
    if image:
        try:
            payload = _probe_json(path, [
                "-select_streams", "v:0", "-show_frames", "-read_intervals", "%+#1",
                "-show_entries", "frame_tags",
            ])
            frames = payload.get("frames", [])
            if frames:
                sources.append(frames[0].get("tags", {}))
        except (YapError, OSError, ValueError, TypeError):
            pass
    aliases = {
        "make": ("com.apple.quicktime.make", "make", "Make", "com.android.manufacturer"),
        "model": ("com.apple.quicktime.model", "model", "Model", "com.android.model"),
        "software": ("com.apple.quicktime.software", "software", "Software"),
        "lens": ("com.apple.quicktime.camera.lens_model", "ExifIFD/LensModel", "LensModel"),
        "captured_at": (
            "com.apple.quicktime.creationdate", "ExifIFD/DateTimeOriginal", "DateTimeOriginal",
        ),
        "container_time": ("creation_time",),
    }
    camera = dict.fromkeys(aliases)
    for source in sources:
        safe = {
            key: value for key, value in source.items()
            if "location" not in key.lower() and "gps" not in key.lower()
        }
        for field, keys in aliases.items():
            for key in keys:
                value = safe.get(key)
                if camera[field] is None and isinstance(value, str) and value.strip():
                    camera[field] = value.strip()[:200].strip()
                    break
    if image and path.suffix.lower() in {".heic", ".heif"} and camera["captured_at"] is None:
        captured = capture_time(path, sha256=sha256)
        if captured is not None:
            camera["captured_at"] = captured.strftime("%Y:%m:%d %H:%M:%S")
    return camera if any(value is not None for value in camera.values()) else None


def _new_item(path: Path, digest: str, size: int, item_id: str, timestamp: str) -> dict:
    info = ffmpeg.probe(path, count_frames=False)
    if not info.has_video:
        raise InputError(str(path), "no video stream")
    image = path.suffix.lower() in _IMAGE_EXTENSIONS
    width, height, rotation = info.width, info.height, info.rotation
    grid = heif_grid(path)
    if grid is not None:
        width, height, rotation = grid
    if not width or not height:
        raise InputError(str(path), "no display dimensions")
    duration = info.video_duration_s or info.duration_s
    if not image and (
        duration is None or not math.isfinite(duration) or duration < 0
        or info.fps is None or info.fps <= 0
    ):
        raise InputError(str(path), "no usable duration or frame rate")
    camera = _camera(path, info, image=image, sha256=digest)
    camera_known = camera is not None and bool(camera["make"] or camera["model"])
    name = re.sub(r"[-_.]+", " ", path.name.lower())
    if image:
        screenshot = not camera_known and (
            path.suffix.lower() == ".png"
            or any(marker in name for marker in ("screenshot", "screen shot", "screen capture"))
        )
        media_type = "screenshot" if screenshot else "photo"
    else:
        screen = not camera_known and any(marker in name for marker in (
            "screen recording", "screen record", "screenrecording", "screencast", "screen capture",
        ))
        media_type = "screen_recording" if screen else "clip"
    orientation = "vertical" if height > width else "horizontal" if width > height else "square"
    return {
        "id": item_id, "sha256": digest, "bytes": size, "kind": "image" if image else "video",
        "media_type": media_type, "media_type_by": "rule", "format": info.format_name,
        "codec": info.video_codec, "width": width, "height": height, "rotation": rotation,
        "orientation": orientation, "duration_s": None if image else float(duration),
        "fps": None if image else f"{info.fps.numerator}/{info.fps.denominator}",
        "variable_frame_rate": False if image else info.variable_frame_rate,
        "color_transfer": info.color_transfer, "hdr": hdr.hdr_kind(info),
        "has_audio": info.has_audio, "camera": camera,
        "provenance": {"kind": None, "source": None, "license": None, "url": None},
        "locations": [], "added_at": timestamp, "description": None, "decisions": [],
        "next_decision": 1, "sheet": None, "speech": None,
    }


def _add_batch(candidates: Sequence[Path], timestamp: str) -> dict:
    skipped = []

    def mutate(document: dict) -> dict:
        by_hash = {item["sha256"]: item for item in document["items"]}
        added, located, unchanged = set(), set(), set()
        for candidate in candidates:
            try:
                real = guarded_paths.validate_input(candidate)
                key = str(real)
                old = document["index"].get(key)
                digest, identity = _hash_media(real, old)
                item = by_hash.get(digest)
                new = item is None
                if new:
                    item = _new_item(
                        real, digest, identity[2], f"m{document['next_id']}", timestamp
                    )
                    if file_identity(real) != identity:
                        raise InputError(key, "file changed while being probed; try again")
            except InputError as exc:
                skipped.append({"path": str(candidate), "reason": exc.reason})
                continue
            except (YapError, OSError, ValueError) as exc:
                reason = (
                    "could not probe media" if isinstance(exc, YapError) else "could not read media"
                )
                skipped.append({"path": str(candidate), "reason": reason})
                continue
            if new:
                document["next_id"] += 1
                document["items"].append(item)
                by_hash[digest] = item
                added.add(item["id"])
            if old and old["sha256"] != digest:
                previous = by_hash[old["sha256"]]
                previous["locations"] = [
                    location for location in previous["locations"] if location["path"] != key
                ]
            location = next((loc for loc in item["locations"] if loc["path"] == key), None)
            if location is not None:
                location["seen_at"] = timestamp
                unchanged.add(item["id"])
            else:
                item["locations"].append({"path": key, "added_at": timestamp, "seen_at": timestamp})
                if not new:
                    located.add(item["id"])
            dev, ino, size, mtime_ns, ctime_ns = identity
            document["index"][key] = {
                "sha256": digest, "size": size, "mtime_ns": mtime_ns, "ctime_ns": ctime_ns,
                "ino": ino, "dev": dev,
            }
        def order(item_id: str) -> int:
            return int(item_id[1:])

        unique_skips = {(entry["path"], entry["reason"]) for entry in skipped}
        return {
            "catalog": str(catalog_path()), "added": sorted(added, key=order),
            "located": sorted(located, key=order), "unchanged": sorted(unchanged, key=order),
            "skipped": [{"path": path, "reason": reason} for path, reason in sorted(unique_skips)],
            "unreadable": [],
            "items": len(document["items"]),
        }

    return update_catalog(mutate)


def add_paths(
    paths: Sequence[str | os.PathLike[str]], *, now: Callable[[], datetime] | None = None,
) -> dict:
    """Scan local media and commit facts in batches without changing the originals."""
    # Preflight every argument before even creating the storage folder or lock.
    candidates, skipped, unreadable_folders = _scan(paths)
    instant = now() if now is not None else datetime.now(UTC)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=UTC)
    timestamp = instant.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    ids: dict[str, set[str]] = {"added": set(), "located": set(), "unchanged": set()}
    unique_skips = {(entry["path"], entry["reason"]) for entry in skipped}
    unreadable = set(unreadable_folders)
    # An empty scan still validates and saves the catalog, as before.
    for offset in range(0, max(1, len(candidates)), _BATCH_FILES):
        result = _add_batch(candidates[offset:offset + _BATCH_FILES], timestamp)
        for category, values in ids.items():
            values.update(result[category])
        unique_skips.update((entry["path"], entry["reason"]) for entry in result["skipped"])
        unreadable.update(result["unreadable"])
    return {
        "catalog": str(catalog_path()),
        **{category: sorted(values, key=lambda item_id: int(item_id[1:]))
           for category, values in ids.items()},
        "skipped": [{"path": path, "reason": reason} for path, reason in sorted(unique_skips)],
        "unreadable": sorted(unreadable),
        "items": len(load_catalog()["items"]),
    }


def add_stock_item(path: Path, *, stock: dict, description_text: str, now=None) -> dict:
    """Add licensed stock bytes and their location under the catalog write lock."""
    from yap_studio.stock.metadata import timestamp, validate_stock

    try:
        validate_stock(stock)
    except (ValueError, TypeError) as exc:
        raise ValidationError("stock", str(exc)) from None
    at = timestamp(now)

    def mutate(document: dict) -> dict:
        real = guarded_paths.validate_input(path)
        key = str(real)
        digest, identity = _hash_media(real, document["index"].get(key))
        item = next((item for item in document["items"] if item["sha256"] == digest), None)
        creator_item = item is not None and "stock" not in item
        if item is None:
            item = _new_item(real, digest, identity[2], f"m{document['next_id']}", at)
            document["next_id"] += 1
            document["items"].append(item)
        if file_identity(real) != identity:
            raise InputError(key, "file changed while being probed; try again")
        old = document["index"].get(key)
        if old and old["sha256"] != digest:
            for previous in document["items"]:
                if previous["sha256"] == old["sha256"]:
                    previous["locations"] = [loc for loc in previous["locations"]
                                             if loc["path"] != key]
        if not creator_item:
            item["stock"] = dict(stock)
            label = "Pexels" if stock["provider"] == "pexels" else stock["provider"]
            item["provenance"] = {"kind": "licensed", "source": label,
                                  "license": stock["license"], "url": stock["page_url"]}
            if item["description"] is None:
                item["description"] = {"text": description_text, "tags": ["stock_clip"],
                                       "by": "stock", "at": at, "revision": 1}
        location = next((loc for loc in item["locations"] if loc["path"] == key), None)
        if location is None:
            item["locations"].append({"path": key, "added_at": at, "seen_at": at})
        else:
            location["seen_at"] = at
        dev, ino, size, mtime_ns, ctime_ns = identity
        document["index"][key] = {"sha256": digest, "size": size, "mtime_ns": mtime_ns,
                                  "ctime_ns": ctime_ns, "ino": ino, "dev": dev}
        return item

    return update_catalog(mutate)


def add_generated_item(path: Path, *, generated: dict, description_text: str, now=None) -> dict:
    """Add generated picture bytes and their location under the catalog write lock."""
    from yap_studio.pictures.metadata import validate_generated
    from yap_studio.stock.metadata import timestamp

    try:
        validate_generated(generated)
    except (ValueError, TypeError) as exc:
        raise ValidationError("generated", str(exc)) from None
    at = timestamp(now)

    def mutate(document: dict) -> dict:
        real = guarded_paths.validate_input(path)
        key = str(real)
        digest, identity = _hash_media(real, document["index"].get(key))
        item = next((item for item in document["items"] if item["sha256"] == digest), None)
        creator_item = item is not None and "generated" not in item and "stock" not in item
        if item is None:
            item = _new_item(real, digest, identity[2], f"m{document['next_id']}", at)
            if item["kind"] == "image":
                item["media_type"] = "photo"
            document["next_id"] += 1
            document["items"].append(item)
        if file_identity(real) != identity:
            raise InputError(key, "file changed while being probed; try again")
        old = document["index"].get(key)
        if old and old["sha256"] != digest:
            for previous in document["items"]:
                if previous["sha256"] == old["sha256"]:
                    previous["locations"] = [loc for loc in previous["locations"]
                                             if loc["path"] != key]
        if not creator_item:
            item["generated"] = dict(generated)
            item["provenance"] = {"kind": "generated", "source": "WaveSpeed",
                                  "license": "WaveSpeed Terms of Service",
                                  "url": "https://wavespeed.ai/static/terms"}
            if item["description"] is None:
                item["description"] = {"text": description_text, "tags": ["generated_picture"],
                                       "by": "generated", "at": at, "revision": 1}
        location = next((loc for loc in item["locations"] if loc["path"] == key), None)
        if location is None:
            item["locations"].append({"path": key, "added_at": at, "seen_at": at})
        else:
            location["seen_at"] = at
        dev, ino, size, mtime_ns, ctime_ns = identity
        document["index"][key] = {"sha256": digest, "size": size, "mtime_ns": mtime_ns,
                                  "ctime_ns": ctime_ns, "ino": ino, "dev": dev}
        return item

    return update_catalog(mutate)
