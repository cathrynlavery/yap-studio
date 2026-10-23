"""Content-addressed capture dates and small visual fingerprints."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from yap_studio import cache, ffmpeg, receipt
from yap_studio.errors import YapError

_STILLS = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".heic", ".heif", ".webp"}


def _read_cache(digest: str) -> tuple[Path, dict]:
    target = cache.cache_dir("media-facts") / f"{digest}.json"
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("sha256") == digest:
            return target, data
    except (OSError, ValueError, TypeError):
        pass
    return target, {"sha256": digest}


def _save_cache(target: Path, data: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=f".{target.stem}-", suffix=".tmp",
                                         delete=False) as output:
            temporary = Path(output.name)
            json.dump(data, output, separators=(",", ":"))
        os.replace(temporary, target)
    except OSError:
        pass  # Derived facts can always be recomputed.
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _tag_time(path: Path) -> datetime | None:
    def probe(options: list[str]) -> dict:
        result = ffmpeg.run(["ffprobe", "-v", "error", *options, "-of", "json", str(path)],
                            timeout_s=ffmpeg.PROBE_TIMEOUT_S)
        return json.loads(result.stdout)

    metadata = probe(["-show_format", "-show_streams"])
    sources = [metadata.get("format", {}).get("tags", {})]
    sources.extend(stream.get("tags", {}) for stream in metadata.get("streams", []))
    if path.suffix.lower() in _STILLS - {".heic", ".heif"}:
        frames = probe(["-select_streams", "v:0", "-show_frames", "-read_intervals",
                        "%+#1", "-show_entries", "frame_tags"]).get("frames", [])
        sources.extend(frame.get("tags", {}) for frame in frames[:1])
    for tags in sources:
        value = tags.get("com.apple.quicktime.creationdate")
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value).replace(tzinfo=None)
            except ValueError:
                pass
    for tags in sources:
        for key in ("ExifIFD/DateTimeOriginal", "DateTimeOriginal"):
            value = tags.get(key)
            if isinstance(value, str):
                try:
                    return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    pass
    for tags in sources:
        value = tags.get("creation_time")
        if isinstance(value, str):
            try:
                instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return instant.astimezone().replace(tzinfo=None)
            except ValueError:
                pass
    return None


def _uint(data: bytes, offset: int, size: int) -> int:
    if size < 0 or offset < 0 or offset + size > len(data):
        raise ValueError("metadata outside file")
    return int.from_bytes(data[offset:offset + size], "big")


def _boxes(data: bytes, start: int, end: int):
    position = start
    while position + 8 <= end:
        size = _uint(data, position, 4)
        kind = data[position + 4:position + 8]
        header = 8
        if size == 1:
            size = _uint(data, position + 8, 8)
            header = 16
        elif size == 0:
            size = end - position
        if size < header or position + size > end:
            raise ValueError("invalid box size")
        yield kind, position + header, position + size
        position += size
    if position != end:
        raise ValueError("trailing box bytes")


def _tiff_time(payload: bytes) -> datetime | None:
    start = 4 + _uint(payload, 0, 4)
    if start + 8 > len(payload):
        raise ValueError("missing TIFF header")
    order = payload[start:start + 2]
    if order not in {b"II", b"MM"}:
        raise ValueError("invalid TIFF byte order")
    endian = "little" if order == b"II" else "big"

    def number(offset: int, size: int) -> int:
        absolute = start + offset
        if absolute < start or absolute + size > len(payload):
            raise ValueError("TIFF field outside item")
        return int.from_bytes(payload[absolute:absolute + size], endian)

    if number(2, 2) != 42:
        raise ValueError("invalid TIFF marker")

    def tags(offset: int) -> dict[int, str | int]:
        count = number(offset, 2)
        if count > (len(payload) - start) // 12:
            raise ValueError("invalid TIFF count")
        values: dict[int, str | int] = {}
        for index in range(count):
            entry = offset + 2 + index * 12
            tag, kind, length = number(entry, 2), number(entry + 2, 2), number(entry + 4, 4)
            if tag == 0x8769 and kind == 4 and length == 1:
                values[tag] = number(entry + 8, 4)
            if tag in {0x9003, 0x0132} and kind == 2 and 0 < length <= 200:
                location = entry + 8 if length <= 4 else number(entry + 8, 4)
                text = payload[start + location:start + location + length]
                if len(text) != length:
                    raise ValueError("TIFF string outside item")
                values[tag] = text.rstrip(b"\x00").decode("ascii")
        return values

    root = tags(number(4, 4))
    exif = tags(root[0x8769]) if isinstance(root.get(0x8769), int) else {}
    for value in (exif.get(0x9003), root.get(0x0132)):
        if isinstance(value, str):
            try:
                return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass
    return None


def _heic_time(path: Path, *, strict: bool = False) -> datetime | None:
    try:
        data = path.read_bytes()
        metadata = next((start, end) for kind, start, end in _boxes(data, 0, len(data))
                        if kind == b"meta")
        start, end = metadata
        item_id = None
        extent = None
        idat = None
        for kind, first, last in _boxes(data, start + 4, end):
            if kind == b"iinf":
                version = data[first]
                cursor = first + 4
                count_size = 2 if version == 0 else 4
                count = _uint(data, cursor, count_size)
                cursor += count_size
                for entry_kind, entry_start, entry_end in _boxes(data, cursor, last):
                    if entry_kind != b"infe" or data[entry_start] not in {2, 3}:
                        continue
                    id_size = 2 if data[entry_start] == 2 else 4
                    if entry_start + 4 + id_size + 6 > entry_end:
                        raise ValueError("short Exif item entry")
                    identity = _uint(data, entry_start + 4, id_size)
                    item_type = data[entry_start + 4 + id_size + 2:entry_start + 4 + id_size + 6]
                    if item_type == b"Exif":
                        item_id = identity
                if count == 0:
                    item_id = None
            elif kind == b"iloc":
                extent = (first, last)
            elif kind == b"idat":
                idat = (first, last)
        if item_id is None or extent is None:
            return None
        first, last = extent
        version = data[first]
        if version not in {0, 1, 2}:
            return None
        sizes = _uint(data, first + 4, 1)
        offset_size, length_size = sizes >> 4, sizes & 15
        sizes = _uint(data, first + 5, 1)
        base_size, index_size = sizes >> 4, (sizes & 15) if version in {1, 2} else 0
        if max(offset_size, length_size, base_size, index_size) > 8:
            return None
        cursor = first + 6
        count_size = 4 if version == 2 else 2
        count = _uint(data, cursor, count_size)
        cursor += count_size
        for _ in range(count):
            if cursor + count_size > last:
                raise ValueError("iloc item outside box")
            identity = _uint(data, cursor, count_size)
            cursor += count_size
            method = 0
            if version in {1, 2}:
                method = _uint(data, cursor, 2) & 15
                cursor += 2
            cursor += 2  # data_reference_index
            base = _uint(data, cursor, base_size)
            cursor += base_size
            extents = _uint(data, cursor, 2)
            cursor += 2
            parts = []
            for _ in range(extents):
                if cursor + index_size + offset_size + length_size > last:
                    raise ValueError("iloc extent outside box")
                cursor += index_size
                offset = _uint(data, cursor, offset_size)
                cursor += offset_size
                length = _uint(data, cursor, length_size)
                cursor += length_size
                if identity == item_id and method in {0, 1}:
                    origin, limit = (0, len(data)) if method == 0 else idat or (0, 0)
                    position = origin + base + offset
                    if position + length > limit:
                        raise ValueError("Exif extent outside file")
                    parts.append(data[position:position + length])
            if cursor > last:
                raise ValueError("iloc outside box")
            if identity == item_id and parts:
                return _tiff_time(b"".join(parts))
    except (OSError, ValueError, StopIteration, IndexError, UnicodeError, OverflowError):
        if strict:
            raise
    return None


def capture_time(path: Path, *, sha256: str | None = None) -> datetime | None:
    """Return the local wall-clock capture time, or None when unavailable."""
    path = Path(path)
    try:
        digest = sha256 or receipt.sha256_file(path)
        target, facts = _read_cache(digest)
        if "captured_at" in facts:
            value = facts["captured_at"]
            if value is None and facts.get("capture_probe_ok") is True:
                return None
            if isinstance(value, str):
                try:
                    cached = datetime.fromisoformat(value)
                    if cached.tzinfo is None:
                        return cached
                except ValueError:
                    pass
        if path.suffix.lower() in {".heic", ".heif"}:
            exif_failed = False
            try:
                value = _heic_time(path, strict=True)
            except (OSError, ValueError, StopIteration, IndexError, UnicodeError, OverflowError):
                value = None
                exif_failed = True
            if value is None:
                value = _tag_time(path)
                if exif_failed:
                    return value
        else:
            value = _tag_time(path)
        facts["captured_at"] = value.isoformat() if value else None
        facts["capture_probe_ok"] = True
        _save_cache(target, facts)
        return value
    except (YapError, OSError, ValueError, TypeError, KeyError, AttributeError,
            StopIteration, IndexError, UnicodeError, OverflowError):
        return None


def fingerprint(path: Path, *, sha256: str, time_s: float = 0,
                crop: tuple[int, int, int, int] | None = None) -> bytes | None:
    """Return an 8x8 RGB thumbnail of a still or a clip's selected source frame."""
    try:
        target, facts = _read_cache(sha256)
        key = str(round(time_s * 1000))
        crop_filter = ""
        if crop is not None:
            x, y, width, height = crop
            key += f":crop={x},{y},{width},{height}"
            crop_filter = f"crop={width}:{height}:{x}:{y}:exact=1,"
        thumbs = facts.setdefault("thumbnails", {})
        if not isinstance(thumbs, dict):
            thumbs = facts["thumbnails"] = {}
        if key in thumbs:
            try:
                pixels = bytes.fromhex(thumbs[key])
                if len(pixels) == 192:
                    return pixels
            except (ValueError, TypeError):
                pass
        seek = [] if Path(path).suffix.lower() in _STILLS else ["-ss", f"{time_s:.3f}"]
        filters = ["-vf", f"{crop_filter}scale=8:8:flags=area,format=rgb24"]
        if Path(path).suffix.lower() in {".heic", ".heif"}:
            from yap_studio.media import catalog

            grid = catalog.heif_grid(Path(path), strict=True)
            if grid is not None:
                filters = ["-filter_complex", f"[0:g:0]{crop_filter}scale=8:8:flags=area,"
                           "format=rgb24[out]",
                           "-map", "[out]"]
        result = ffmpeg.run(["ffmpeg", "-v", "error", *seek, "-i", str(path), *filters,
                             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            timeout_s=ffmpeg.DEFAULT_TIMEOUT_S, binary=True)
        pixels = result.stdout if len(result.stdout) == 192 else None
        if pixels is None:
            return None
        thumbs[key] = pixels.hex()
        _save_cache(target, facts)
        return pixels
    except (YapError, OSError, ValueError, TypeError, KeyError, AttributeError):
        return None
