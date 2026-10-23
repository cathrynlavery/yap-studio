"""Capture dates from synthetic media and bounded HEIF metadata."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import synth_broll as synth

from yap_studio.errors import ProcessingError
from yap_studio.media import captured, catalog
from yap_studio.receipt import sha256_file


def box(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data) + 8) + kind + data


def tiff_date(order: str) -> bytes:
    prefix = "<" if order == "II" else ">"
    stamp = b"2023:06:17 18:27:59\x00"
    ifd0 = struct.pack(prefix + "H", 1)
    ifd0 += struct.pack(prefix + "HHII", 0x8769, 4, 1, 26)
    ifd0 += struct.pack(prefix + "I", 0)
    exif = struct.pack(prefix + "H", 1)
    exif += struct.pack(prefix + "HHII", 0x9003, 2, len(stamp), 44)
    exif += struct.pack(prefix + "I", 0)
    return order.encode() + struct.pack(prefix + "HI", 42, 8) + ifd0 + exif + stamp


def tiff_ifd0_date(order: str) -> bytes:
    prefix = "<" if order == "II" else ">"
    stamp = b"2023:06:17 18:27:59\x00"
    ifd0 = struct.pack(prefix + "H", 1)
    ifd0 += struct.pack(prefix + "HHII", 0x0132, 2, len(stamp), 26)
    return order.encode() + struct.pack(prefix + "HI", 42, 8) + ifd0 + struct.pack(
        prefix + "I", 0) + stamp


def heic(path: Path, order: str, version: int = 1, *, tiff: bytes | None = None,
         header_offset: int = 0, base_index: bool = False) -> Path:
    payload = struct.pack(">I", header_offset) + bytes(header_offset) + (tiff or tiff_date(order))
    infe = box(b"infe", bytes([2, 0, 0, 0]) + struct.pack(">HH", 1, 0) + b"Exif\x00")
    iinf = box(b"iinf", bytes(4) + struct.pack(">H", 1) + infe)
    hdlr = box(b"hdlr", bytes(8) + b"pict" + bytes(12))
    ftyp = box(b"ftyp", b"heic\x00\x00\x00\x00heic")
    # iloc uses 4-byte extent offset and length, no base offset or index.
    def metadata(offset: int) -> bytes:
        entry = struct.pack(">I" if version == 2 else ">H", 1)
        if version in {1, 2}:
            entry += b"\x00\x00"  # construction_method 0
        entry += struct.pack(">H", 0)  # data_reference_index
        if base_index:
            entry += struct.pack(">I", max(0, offset - 4))
        entry += struct.pack(">H", 1)
        if base_index:
            entry += struct.pack(">I", 1)  # extent_index
        entry += struct.pack(">II", 4 if base_index else offset, len(payload))
        count = struct.pack(">H", 1) if version < 2 else struct.pack(">I", 1)
        iloc = box(b"iloc", bytes([version, 0, 0, 0, 0x44, 0x44 if base_index else 0])
                   + count + entry)
        return box(b"meta", bytes(4) + hdlr + iinf + iloc)
    offset = len(ftyp) + len(metadata(0)) + 8
    path.write_bytes(ftyp + metadata(offset) + box(b"mdat", payload))
    return path


def heic_without_exif(path: Path) -> Path:
    path.write_bytes(box(b"ftyp", b"heic\x00\x00\x00\x00heic") + box(
        b"meta", bytes(4) + box(b"hdlr", bytes(8) + b"pict" + bytes(12))
        + box(b"iinf", bytes(4) + b"\x00\x00")))
    return path


@pytest.fixture(autouse=True)
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YAP_STUDIO_CACHE", str(tmp_path / "cache"))


@pytest.mark.parametrize("tags,expected", [
    ({"com.apple.quicktime.creationdate": "2023-05-29T13:34:10-0500"},
     datetime(2023, 5, 29, 13, 34, 10)),
    ({"creation_time": "2023-05-29T18:34:10Z"}, None),
])
def test_video_times(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                     tags: dict, expected: datetime | None) -> None:
    path = tmp_path / "shot.mov"
    path.write_bytes(b"synthetic video identity")
    monkeypatch.setattr(captured.ffmpeg, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout=json.dumps({"format": {"tags": tags}, "streams": []})))
    result = captured.capture_time(path)
    if expected is None:
        assert result == datetime.fromisoformat("2023-05-29T18:34:10+00:00").astimezone().replace(
            tzinfo=None)
    else:
        assert result == expected


def test_jpeg_exif(tmp_path: Path) -> None:
    path = synth.jpeg(tmp_path / "dated.jpg")
    data = path.read_bytes()
    exif = b"Exif\x00\x00" + tiff_date("II")
    path.write_bytes(data[:2] + b"\xff\xe1" + struct.pack(">H", len(exif) + 2) + exif + data[2:])
    assert captured.capture_time(path) == datetime(2023, 6, 17, 18, 27, 59)


@pytest.mark.parametrize("order,version", [("II", 0), ("MM", 1), ("II", 2)])
def test_heic_exif(tmp_path: Path, order: str, version: int) -> None:
    assert captured.capture_time(heic(tmp_path / "dated.heic", order, version)) == datetime(
        2023, 6, 17, 18, 27, 59)


def test_heic_ifd0_fallback_and_extent_fields(tmp_path: Path) -> None:
    path = heic(tmp_path / "ifd0.heic", "MM", 1, tiff=tiff_ifd0_date("MM"),
                header_offset=4, base_index=True)
    assert captured.capture_time(path) == datetime(2023, 6, 17, 18, 27, 59)


@pytest.mark.parametrize(("tag", "expected"), [
    ({"com.apple.quicktime.creationdate": "2023-05-29T13:34:10-0500"},
     datetime(2023, 5, 29, 13, 34, 10)),
    ({"creation_time": "2023-05-29T18:34:10Z"},
     datetime.fromisoformat("2023-05-29T18:34:10+00:00").astimezone().replace(tzinfo=None)),
])
def test_heic_without_exif_uses_container_creation_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tag: dict[str, str], expected: datetime,
) -> None:
    path = heic_without_exif(tmp_path / "undated.heic")
    assert captured._heic_time(path) is None

    def probe(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout=json.dumps({"format": {"tags": tag}, "streams": []}))

    monkeypatch.setattr(captured.ffmpeg, "run", probe)
    assert captured.capture_time(path) == expected


@pytest.mark.parametrize("tag,expected", [
    ("2023-05-29T13:34:10-0500", datetime(2023, 5, 29, 13, 34, 10)),
    (None, None),
])
def test_heic_damaged_exif_uses_container_date_without_caching_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tag: str | None,
    expected: datetime | None,
) -> None:
    path = heic(tmp_path / "damaged.heic", "II", tiff=b"II")
    with pytest.raises(ValueError, match="missing TIFF header"):
        captured._heic_time(path, strict=True)
    tags = {"com.apple.quicktime.creationdate": tag} if tag else {}
    monkeypatch.setattr(captured.ffmpeg, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout=json.dumps({"format": {"tags": tags}, "streams": []})))

    assert captured.capture_time(path) == expected
    entries = list((tmp_path / "cache" / "media-facts").glob("*.json"))
    assert not entries or "captured_at" not in json.loads(entries[0].read_text())


def test_heic_exif_read_failure_retries_before_caching_container_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = heic_without_exif(tmp_path / "retry.heic")
    exif_date = datetime(2023, 6, 17, 18, 27, 59)
    container_date = datetime(2023, 6, 20, 18, 27, 59)
    calls = 0

    def exif_time(*args: object, **kwargs: object) -> datetime:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary read failure")
        return exif_date

    monkeypatch.setattr(captured, "_heic_time", exif_time)
    monkeypatch.setattr(captured, "_tag_time", lambda path: container_date)

    assert captured.capture_time(path) == container_date
    entries = list((tmp_path / "cache" / "media-facts").glob("*.json"))
    assert not entries or "captured_at" not in json.loads(entries[0].read_text())

    assert captured.capture_time(path) == exif_date
    assert calls == 2


def test_heic_missing_date_is_cached_only_after_successful_tag_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = heic_without_exif(tmp_path / "unknown.heic")

    def fail(*args: object, **kwargs: object) -> object:
        raise ProcessingError("ffprobe", -1, "timed out", code="ffmpeg_timeout")

    monkeypatch.setattr(captured.ffmpeg, "run", fail)
    assert captured.capture_time(path) is None
    entries = list((tmp_path / "cache" / "media-facts").glob("*.json"))
    assert not entries or "captured_at" not in json.loads(entries[0].read_text())
    monkeypatch.setattr(captured.ffmpeg, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout=json.dumps({"format": {"tags": {}}, "streams": []})))
    assert captured.capture_time(path) is None
    entry = next((tmp_path / "cache" / "media-facts").glob("*.json"))
    saved = json.loads(entry.read_text())
    assert saved["captured_at"] is None and saved["capture_probe_ok"] is True


def test_garbage_and_truncated_heic(tmp_path: Path) -> None:
    path = heic(tmp_path / "bad.heic", "II")
    data = path.read_bytes()
    for payload in (b"garbage", data[:25], data[:-10]):
        path.write_bytes(payload)
        assert captured.capture_time(path) is None


def test_cache_hit_and_corrupt_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = heic(tmp_path / "cached.heic", "MM")
    assert captured.capture_time(path) == datetime(2023, 6, 17, 18, 27, 59)
    monkeypatch.setattr(captured, "_heic_time", lambda path, **kwargs: (_ for _ in ()).throw(
        AssertionError("cache missed")))
    assert captured.capture_time(path) == datetime(2023, 6, 17, 18, 27, 59)
    entry = next((tmp_path / "cache" / "media-facts").glob("*.json"))
    entry.write_text("broken", encoding="utf-8")
    monkeypatch.setattr(captured, "_heic_time", lambda path, **kwargs: datetime(
        2023, 6, 17, 18, 27, 59))
    assert captured.capture_time(path) == datetime(2023, 6, 17, 18, 27, 59)


def test_catalog_camera_uses_heic_exif_when_probe_has_no_date(tmp_path: Path) -> None:
    path = heic(tmp_path / "catalog.heic", "II")
    camera = catalog._camera(path, SimpleNamespace(format_tags={}), image=True)
    assert camera is not None
    assert camera["captured_at"] == "2023:06:17 18:27:59"


def test_tiled_heic_fingerprint_uses_full_picture(tmp_path: Path) -> None:
    if shutil.which("sips") is None:
        pytest.skip("sips is needed to generate a tiled HEIC still")
    png = tmp_path / "quadrants.png"
    heic = tmp_path / "quadrants.heic"
    synth.synth._run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        "color=red:s=1280x720,"
        "drawbox=x=640:y=0:w=640:h=360:c=green:t=fill,"
        "drawbox=x=0:y=360:w=640:h=360:c=blue:t=fill,"
        "drawbox=x=640:y=360:w=640:h=360:c=yellow:t=fill",
        "-frames:v", "1", str(png),
    ])
    subprocess.run(["sips", "-s", "format", "heic", str(png), "--out", str(heic)],
                   check=True, capture_output=True)
    assert catalog.heif_grid(heic) is not None

    def thumbnail(path: Path, *, grid: bool = False) -> bytes:
        filters = ["-filter_complex", "[0:g:0]scale=8:8:flags=area,format=rgb24[out]",
                   "-map", "[out]"] if grid else ["-map", "0:v:0", "-vf",
                                                  "scale=8:8:flags=area,format=rgb24"]
        return captured.ffmpeg.run([
            "ffmpeg", "-v", "error", "-i", str(path), *filters,
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ], timeout_s=60, binary=True).stdout

    expected = thumbnail(png)
    flat_tile = thumbnail(heic)
    full = captured.fingerprint(heic, sha256="a" * 64)
    assert full is not None and len(full) == 192

    def distance(value: bytes) -> float:
        return sum(abs(a - b) for a, b in zip(value, expected, strict=True)) / 192

    assert distance(flat_tile) > 24
    assert distance(full) < 10
    crop = (640, 0, 640, 360)
    expected_crop = captured.fingerprint(png, sha256="d" * 64, crop=crop)
    grid_crop = captured.fingerprint(heic, sha256="a" * 64, crop=crop)
    assert expected_crop is not None and grid_crop is not None
    assert sum(abs(a - b) for a, b in zip(grid_crop, expected_crop, strict=True)) / 192 < 10


def test_fingerprint_caches_distinct_crops_of_same_frame(tmp_path: Path) -> None:
    path = tmp_path / "split.png"
    synth.synth._run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        "color=c=red:s=80x80,drawbox=x=40:y=0:w=40:h=80:c=blue:t=fill",
        "-frames:v", "1", str(path),
    ])
    digest = sha256_file(path)
    uncropped = captured.fingerprint(path, sha256=digest)
    left = captured.fingerprint(path, sha256=digest, crop=(0, 0, 40, 80))
    right = captured.fingerprint(path, sha256=digest, crop=(40, 0, 40, 80))
    assert uncropped is not None and left is not None and right is not None and left != right

    entry = captured.cache.cache_dir("media-facts") / f"{digest}.json"
    thumbs = json.loads(entry.read_text())["thumbnails"]
    assert thumbs["0"] == uncropped.hex()
    assert len(thumbs) == 3
    assert set(thumbs.values()) == {uncropped.hex(), left.hex(), right.hex()}


def test_probe_failure_does_not_cache_unknown_capture_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "retry.mov"
    path.write_bytes(b"synthetic video identity")
    real_run = captured.ffmpeg.run

    def fail(*args: object, **kwargs: object) -> object:
        raise ProcessingError("ffprobe", -1, "timed out", code="ffmpeg_timeout")

    monkeypatch.setattr(captured.ffmpeg, "run", fail)
    assert captured.capture_time(path) is None
    entries = list((tmp_path / "cache" / "media-facts").glob("*.json"))
    assert not entries or "captured_at" not in json.loads(entries[0].read_text())
    monkeypatch.setattr(captured.ffmpeg, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout=json.dumps({"format": {"tags": {
            "com.apple.quicktime.creationdate": "2023-05-29T13:34:10-0500"}}, "streams": []})))
    assert captured.capture_time(path) == datetime(2023, 5, 29, 13, 34, 10)
    monkeypatch.setattr(captured.ffmpeg, "run", real_run)


def test_old_unknown_cache_entry_is_reprobed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "old.mov"
    path.write_bytes(b"synthetic video identity")
    digest = sha256_file(path)
    entry = captured.cache.cache_dir("media-facts") / f"{digest}.json"
    entry.write_text(json.dumps({"sha256": digest, "captured_at": None}), encoding="utf-8")
    monkeypatch.setattr(captured.ffmpeg, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout=json.dumps({"format": {"tags": {
            "com.apple.quicktime.creationdate": "2023-05-29T13:34:10-0500"}}, "streams": []})))
    assert captured.capture_time(path) == datetime(2023, 5, 29, 13, 34, 10)


@pytest.mark.parametrize("failure", ["error", "empty"])
def test_decode_failure_does_not_cache_missing_thumbnail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    path = tmp_path / "retry.png"
    path.write_bytes(b"synthetic image identity")
    pixels = bytes(range(192))
    def bad_run(*args: object, **kwargs: object) -> object:
        if failure == "error":
            raise ProcessingError("ffmpeg", -1, "timed out", code="ffmpeg_timeout")
        return SimpleNamespace(stdout=b"")

    monkeypatch.setattr(captured.ffmpeg, "run", bad_run)
    assert captured.fingerprint(path, sha256="b" * 64) is None
    entries = list((tmp_path / "cache" / "media-facts").glob("*.json"))
    assert not entries or "0" not in json.loads(entries[0].read_text()).get("thumbnails", {})
    monkeypatch.setattr(captured.ffmpeg, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout=pixels))
    assert captured.fingerprint(path, sha256="b" * 64) == pixels


def test_heic_grid_probe_failure_does_not_cache_tile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "retry.heic"
    path.write_bytes(b"synthetic image identity")

    def fail(*args: object, **kwargs: object) -> object:
        raise ProcessingError("ffprobe", -1, "timed out", code="ffmpeg_timeout")

    monkeypatch.setattr(catalog, "_probe_json", fail)
    assert captured.fingerprint(path, sha256="c" * 64) is None
    entries = list((tmp_path / "cache" / "media-facts").glob("*.json"))
    assert not entries or "0" not in json.loads(entries[0].read_text()).get("thumbnails", {})
    monkeypatch.setattr(catalog, "_probe_json", lambda *args, **kwargs: {"stream_groups": []})
    monkeypatch.setattr(captured.ffmpeg, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout=bytes(range(192))))
    assert captured.fingerprint(path, sha256="c" * 64) == bytes(range(192))
