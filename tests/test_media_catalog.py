"""Media facts, catalog identity, strict storage, and command line behavior."""

from __future__ import annotations

import copy
import errno
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import synth_broll as synth

from yap_studio import cli, ffmpeg, paths, receipt
from yap_studio.errors import InputError, OutputPathError, ValidationError
from yap_studio.media import (
    CATALOG_SCHEMA,
    UnknownMediaError,
    add_paths,
    catalog,
    catalog_path,
    get_item,
    list_items,
    load_catalog,
    update_catalog,
    validate_catalog,
)
from yap_studio.media import commands as media_commands

NOW = datetime(2026, 10, 5, 12, 34, 56, tzinfo=UTC)
STAMP = "2026-10-05T12:34:56Z"


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))


@pytest.fixture(scope="session")
def sample_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synth.clip(tmp_path_factory.mktemp("broll") / "clip.mov")


@pytest.fixture
def clip(tmp_path: Path, sample_clip: Path) -> Path:
    return Path(shutil.copyfile(sample_clip, tmp_path / "clip.mov"))


def _add(*sources: Path, now: datetime = NOW) -> dict:
    return add_paths(sources, now=lambda: now)


def test_missing_catalog_reads_do_not_create_storage() -> None:
    assert load_catalog() == {"schema": CATALOG_SCHEMA, "next_id": 1, "items": [], "index": {}}
    assert list_items() == []
    assert not catalog_path().parent.exists()
    with pytest.raises(UnknownMediaError) as caught:
        get_item("m1")
    assert caught.value.code == "unknown_media"
    assert caught.value.suggested_action == "Run yap-studio media list to see catalog ids."


def test_recursive_scan_is_deterministic_and_preserves_originals(
    tmp_path: Path, sample_clip: Path,
) -> None:
    media = tmp_path / "media"
    nested = media / "nested"
    nested.mkdir(parents=True)
    valid = Path(shutil.copyfile(sample_clip, nested / "sample.MOV"))
    image = synth.png(media / "still.PNG")
    shutil.copyfile(sample_clip, media / ".hidden.mov")
    shutil.copyfile(sample_clip, media / "._sample.mov")
    hidden = media / ".hidden"
    hidden.mkdir()
    shutil.copyfile(sample_clip, hidden / "clip.mov")
    (media / "notes.txt").write_text("ignored")
    corrupt = media / "broken.mov"
    corrupt.write_bytes(b"not media")
    link = media / "linked.mov"
    link.symlink_to(valid)
    (media / "linked-folder").symlink_to(nested, target_is_directory=True)
    originals = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in media.rglob("*") if path.is_file()
    }
    listing = sorted(str(path.relative_to(media)) for path in media.rglob("*"))
    result = _add(media)
    assert result["unreadable"] == []
    assert result["added"] == ["m1", "m2"]
    assert result["located"] == result["unchanged"] == []
    assert result["items"] == 2
    assert result["skipped"] == [
        {"path": str(corrupt), "reason": "ffmpeg cannot decode it as audio or video"},
        {"path": str(link), "reason": "symbolic link"},
    ]
    assert [Path(item["locations"][0]["path"]) for item in list_items()] == [valid, image]
    for path, (content, mtime_ns) in originals.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == (content, mtime_ns)
    assert sorted(str(path.relative_to(media)) for path in media.rglob("*")) == listing
    assert catalog_path() == tmp_path / "config" / "assets" / "catalog.json"
    assert result["catalog"] == str(catalog_path())
    assert {path.name for path in tmp_path.iterdir()} == {"media", "config"}
    assert {str(path.relative_to(tmp_path / "config")) for path in (
        tmp_path / "config"
    ).rglob("*")} == {"assets", "assets/.lock", "assets/catalog.json"}


@pytest.mark.parametrize("as_json", [False, True])
def test_unreadable_nested_folder_keeps_readable_sibling_and_cli_warns(
    tmp_path: Path, sample_clip: Path, capsys: pytest.CaptureFixture[str], as_json: bool,
) -> None:
    if os.name == "nt" or os.geteuid() == 0:
        pytest.skip("folder permissions do not prevent listing on Windows or as root")
    archive = tmp_path / "archive"
    blocked, readable = archive / "blocked", archive / "readable"
    blocked.mkdir(parents=True)
    readable.mkdir()
    shutil.copyfile(sample_clip, blocked / "hidden.mov")
    source = Path(shutil.copyfile(sample_clip, readable / "visible.mov"))
    original_mode = blocked.stat().st_mode
    blocked.chmod(0)
    try:
        result = _add(archive)
        assert result["added"] == ["m1"]
        assert cli.main(["media", "add", str(archive), *(["--json"] if as_json else [])]) == 1
        output = capsys.readouterr()
        assert result["unreadable"] == [str(blocked)]
        assert result["skipped"] == [{
            "path": str(blocked), "reason": f"cannot list folder: {os.strerror(errno.EACCES)}",
        }]
        assert get_item("m1")["locations"][0]["path"] == str(source)
        assert output.err == (
            "warning: could not list 1 folder(s); media inside was not cataloged\n"
        )
        if as_json:
            assert json.loads(output.out)["unreadable"] == [str(blocked)]
        else:
            assert f"skipped: {blocked}: cannot list folder:" in output.out
        assert len(list_items()) == 1
    finally:
        blocked.chmod(original_mode)


@pytest.mark.parametrize("as_json", [False, True])
def test_scan_onerror_reports_sorted_folders_across_batches(
    tmp_path: Path, clip: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], as_json: bool,
) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    sources = [Path(shutil.copyfile(clip, archive / "clip.mov")), synth.png(archive / "still.png")]
    blocked = [archive / "z-blocked", archive / "a-blocked"]

    def walk(root: Path, *, followlinks: bool = False, onerror=None):
        if onerror is not None:
            for folder in blocked:
                onerror(OSError(errno.EACCES, "permission denied", str(folder)))
        yield str(root), [], [source.name for source in sources]

    monkeypatch.setattr(catalog.os, "walk", walk)
    monkeypatch.setattr(catalog, "_BATCH_FILES", 1)
    result = _add(archive, archive)
    assert result["added"] == ["m1", "m2"]
    assert cli.main(["media", "add", str(archive), *(["--json"] if as_json else [])]) == 1
    output = capsys.readouterr()
    assert result["unreadable"] == sorted(str(folder) for folder in blocked)
    assert result["skipped"] == [
        {"path": str(folder), "reason": "cannot list folder: permission denied"}
        for folder in sorted(blocked)
    ]
    assert output.err == "warning: could not list 2 folder(s); media inside was not cataloged\n"
    if as_json:
        assert json.loads(output.out)["unreadable"] == result["unreadable"]
    assert len(list_items()) == 2


def test_clip_facts_and_initial_agent_fields(clip: Path) -> None:
    _add(clip)
    item = get_item("m1")
    assert item["sha256"] == receipt.sha256_file(clip)
    assert item["bytes"] == clip.stat().st_size
    assert item["kind"] == "video"
    assert item["media_type"] == "clip"
    assert item["media_type_by"] == "rule"
    assert (item["width"], item["height"], item["rotation"]) == (160, 90, 0)
    assert item["duration_s"] == pytest.approx(2.0)
    assert item["fps"] == "30/1"
    assert item["variable_frame_rate"] is False
    assert item["has_audio"] is True
    assert item["codec"] == "h264"
    assert "mov" in item["format"]
    assert item["orientation"] == "horizontal"
    assert item["hdr"] is None
    assert item["description"] is item["sheet"] is item["speech"] is None
    assert item["decisions"] == [] and item["next_decision"] == 1
    assert item["provenance"] == {"kind": None, "source": None, "license": None, "url": None}
    assert item["added_at"] == STAMP
    assert item["locations"] == [{"path": str(clip), "added_at": STAMP, "seen_at": STAMP}]
    entry = load_catalog()["index"][str(clip)]
    stat = clip.stat()
    assert entry == {
        "sha256": item["sha256"], "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns, "ino": stat.st_ino, "dev": stat.st_dev,
    }


def test_vertical_clip(tmp_path: Path) -> None:
    _add(synth.clip(tmp_path / "vertical.mov", size="90x160", audio=False))
    item = get_item("m1")
    assert (item["width"], item["height"], item["orientation"]) == (90, 160, "vertical")
    assert item["has_audio"] is False


def test_rotated_clip(tmp_path: Path, clip: Path) -> None:
    if not synth.display_rotation_supported():
        pytest.skip("display rotation is unsupported by this encoder")
    _add(synth.rotated(clip, tmp_path / "rotated.mov"))
    item = get_item("m1")
    assert (item["width"], item["height"], item["orientation"]) == (90, 160, "vertical")
    assert item["rotation"] in {90, 270}


def test_hlg_clip(tmp_path: Path) -> None:
    _add(synth.hlg(tmp_path / "hlg.mp4"))
    item = get_item("m1")
    assert item["hdr"] == "hlg"
    assert item["color_transfer"] == "arib-std-b67"
    assert item["codec"] == "hevc"


def test_stills_and_type_filter(tmp_path: Path) -> None:
    _add(synth.png(tmp_path / "still.png"), synth.jpeg(tmp_path / "still.jpg"))
    items = list_items()
    assert [item["media_type"] for item in items] == ["photo", "screenshot"]
    for item in items:
        assert item["kind"] == "image"
        assert item["duration_s"] is item["fps"] is None
        assert item["variable_frame_rate"] is False
        assert item["has_audio"] is False
        assert (item["width"], item["height"], item["orientation"]) == (80, 80, "square")
        assert item["camera"] is None or item["camera"]["make"] is None
    assert list_items(media_type="photo") == [items[0]]
    assert list_items(media_type="clip") == []
    with pytest.raises(ValidationError):
        list_items(media_type="unknown")


@pytest.mark.parametrize("filename", [
    "Screen Recording.mov", "demo-live-screen-recording-720.mp4", "screen_recording.mov",
    "Screen.Recording.mov", "screen--__..recording.mov",
])
def test_camera_tags_are_recorded_and_geographic_tags_dropped(
    tmp_path: Path, filename: str,
) -> None:
    _add(synth.clip(tmp_path / filename, camera=True))
    item = get_item("m1")
    assert item["camera"] == {
        "make": "Example Camera", "model": "Pocket Model", "software": None,
        "lens": None, "captured_at": None, "container_time": None,
    }
    assert item["media_type"] == "clip"
    text = catalog_path().read_text()
    assert "quicktime.location" not in text.lower()
    assert "gps" not in text.lower()
    assert "+01.0000+002.0000/" not in text
    assert set(item["camera"]) == catalog._CAMERA_KEYS


@pytest.mark.parametrize("filename", [
    "Screen Recording.mov", "demo-live-screen-recording-720.mp4", "screen_recording.mov",
    "Screen.Recording.mov", "screen--__..recording.mov",
])
def test_screen_recording_filename(tmp_path: Path, clip: Path, filename: str) -> None:
    recording = tmp_path / filename
    clip.rename(recording)
    _add(recording)
    assert get_item("m1")["media_type"] == "screen_recording"


@pytest.mark.parametrize("filename", [
    "Screen Shot.jpg", "screen-shot.jpg", "screen_shot.jpg", "Screen.Shot.jpg",
    "screen--__..shot.jpeg",
])
@pytest.mark.parametrize("camera", [False, True])
def test_screenshot_filename_separators_preserve_camera_photos(
    tmp_path: Path, filename: str, camera: bool,
) -> None:
    _add(synth.jpeg(tmp_path / filename, camera=camera))
    item = get_item("m1")
    assert item["media_type"] == ("photo" if camera else "screenshot")
    if camera:
        assert item["camera"]["make"] == "Example Camera"
        assert item["camera"]["model"] == "Pocket Model"


def test_audio_only_supported_file_is_skipped(tmp_path: Path, clip: Path) -> None:
    audio = synth.audio_only(tmp_path / "audio.mov")
    result = _add(audio, clip)
    assert result["added"] == ["m1"] and result["items"] == 1
    assert result["skipped"] == [{"path": str(audio), "reason": "no video stream"}]
    assert str(audio) not in load_catalog()["index"]


def test_unchanged_readd_uses_index_without_hashing_or_probing(
    clip: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "posix":
        pytest.skip("file metadata is trusted only on POSIX")
    _add(clip)
    before = get_item("m1")

    def fail(*args: object, **kwargs: object) -> None:
        pytest.fail("unchanged media was read again")

    monkeypatch.setattr(receipt, "sha256_file", fail)
    monkeypatch.setattr(ffmpeg, "probe", fail)
    later = NOW + timedelta(hours=1)
    result = _add(clip, now=later)
    assert result["added"] == result["located"] == []
    assert result["unchanged"] == ["m1"]
    after = get_item("m1")
    before["locations"][0]["seen_at"] = "2026-10-05T13:34:56Z"
    assert after == before


def test_moved_file_keeps_id_and_location_history(tmp_path: Path, clip: Path) -> None:
    _add(clip)
    moved = clip.rename(tmp_path / "moved.mov")
    result = _add(moved, now=NOW + timedelta(hours=1))
    assert result["added"] == [] and result["located"] == ["m1"]
    assert result["items"] == 1
    assert [loc["path"] for loc in get_item("m1")["locations"]] == [str(clip), str(moved)]
    assert get_item("m1")["locations"][0]["seen_at"] == STAMP
    assert not clip.exists()


def test_duplicate_copies_share_one_item_and_results_are_deduplicated(
    tmp_path: Path, clip: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_probe = ffmpeg.probe
    calls = []

    def record(path: Path, *, count_frames: bool = True) -> ffmpeg.MediaInfo:
        calls.append(path)
        assert count_frames is False
        return original_probe(path, count_frames=count_frames)

    monkeypatch.setattr(ffmpeg, "probe", record)
    duplicate = Path(shutil.copyfile(clip, tmp_path / "duplicate.mov"))
    result = _add(clip, duplicate, clip, duplicate)
    assert result["added"] == result["located"] == ["m1"]
    assert result["items"] == 1 and result["unchanged"] == []
    assert len(calls) == 1
    assert [location["path"] for location in get_item("m1")["locations"]] == [
        str(clip), str(duplicate),
    ]
    assert _add(clip, duplicate, clip)["unchanged"] == ["m1"]


def test_changed_content_transfers_location_and_never_reuses_ids(clip: Path) -> None:
    _add(clip)
    first_digest = get_item("m1")["sha256"]
    synth.clip(clip, colour="red")
    assert _add(clip)["added"] == ["m2"]
    assert get_item("m1")["locations"] == []
    assert get_item("m2")["locations"][0]["path"] == str(clip)
    assert get_item("m2")["sha256"] != first_digest
    assert load_catalog()["index"][str(clip)]["sha256"] == get_item("m2")["sha256"]
    update_catalog(lambda doc: doc["items"].pop(0))
    synth.clip(clip, colour="green")
    assert _add(clip)["added"] == ["m3"]
    assert [item["id"] for item in list_items()] == ["m2", "m3"]


def test_missing_argument_preflight_does_not_write(tmp_path: Path, clip: Path) -> None:
    missing = tmp_path / "missing.mov"
    with pytest.raises(InputError) as caught:
        _add(clip, missing)
    assert caught.value.path == str(missing)
    assert caught.value.reason == "file or folder not found"
    assert not catalog_path().parent.exists()
    _add(clip)
    before = catalog_path().read_bytes()
    with pytest.raises(InputError):
        _add(clip, missing)
    assert catalog_path().read_bytes() == before


def test_config_tree_and_unsupported_files_are_ignored(tmp_path: Path, clip: Path) -> None:
    own = tmp_path / "config"
    own.mkdir()
    shutil.copyfile(clip, own / "private.mov")
    (tmp_path / "notes.txt").write_text("ignored")
    result = _add(tmp_path)
    assert result["added"] == ["m1"] and result["skipped"] == []
    assert [loc["path"] for loc in get_item("m1")["locations"]] == [str(clip)]
    assert _add(own / "private.mov")["unchanged"] == []


def test_explicit_symlink_is_resolved(tmp_path: Path, clip: Path) -> None:
    link = tmp_path / "explicit.mov"
    link.symlink_to(clip)
    assert _add(link)["skipped"] == []
    assert get_item("m1")["locations"][0]["path"] == str(clip.resolve())


@pytest.mark.parametrize("payload", ["{broken", "[]", "null", '{"schema":"unknown"}'])
def test_corrupt_catalog_reports_recovery(tmp_path: Path, payload: str) -> None:
    target = catalog_path()
    target.parent.mkdir(parents=True)
    target.write_text(payload)
    with pytest.raises(InputError) as caught:
        load_catalog()
    assert caught.value.path == str(target)
    assert caught.value.reason == catalog._INVALID


@pytest.fixture
def document(clip: Path) -> dict:
    _add(clip)
    return load_catalog()


@pytest.mark.parametrize("defect", [
    "schema", "extra", "duplicate_id", "duplicate_sha", "bad_sha", "boolean_integer",
    "bad_index", "relative_path", "bad_timestamp", "bad_enum", "nonfinite", "bad_fps",
    "future_id", "camera_extra", "provenance_extra", "location_extra", "index_extra",
])
def test_catalog_validation_rejects_invalid_documents(document: dict, defect: str) -> None:
    item = document["items"][0]
    if defect == "schema":
        document["schema"] = "yap-studio/media-catalog@2"
    elif defect == "extra":
        document["extra"] = 1
    elif defect == "duplicate_id":
        duplicate = copy.deepcopy(item)
        duplicate["sha256"] = "a" * 64
        document["items"].append(duplicate)
    elif defect == "duplicate_sha":
        duplicate = copy.deepcopy(item)
        duplicate["id"] = "m2"
        document["next_id"] = 3
        document["items"].append(duplicate)
    elif defect == "bad_sha":
        item["sha256"] = "A" * 64
    elif defect == "boolean_integer":
        item["width"] = True
    elif defect == "bad_index":
        next(iter(document["index"].values()))["sha256"] = "b" * 64
    elif defect == "relative_path":
        item["locations"][0]["path"] = "relative.mov"
    elif defect == "bad_timestamp":
        item["added_at"] = "2026-02-30T00:00:00Z"
    elif defect == "bad_enum":
        item["media_type"] = "unknown"
    elif defect == "nonfinite":
        item["duration_s"] = float("nan")
    elif defect == "bad_fps":
        item["fps"] = "30/0"
    elif defect == "future_id":
        item["id"] = "m2"
    elif defect == "camera_extra":
        item["camera"] = dict.fromkeys(catalog._CAMERA_KEYS)
        item["camera"].update(make="Camera", gps="not allowed")
    elif defect == "provenance_extra":
        item["provenance"]["extra"] = True
    elif defect == "location_extra":
        item["locations"][0]["exists"] = True
    else:
        next(iter(document["index"].values()))["extra"] = 1
    target = catalog_path()
    target.write_text(json.dumps(document))
    with pytest.raises(InputError, match="not a valid media catalog"):
        load_catalog()


def _enrich(item: dict) -> None:
    item["media_type_by"] = "agent"
    item["description"] = {
        "text": "A useful view", "tags": ["desk"], "by": "agent", "at": STAMP, "revision": 1,
    }
    item["decisions"] = [{
        "id": "d1", "verdict": "approved", "start_s": 0.0, "end_s": 1.0,
        "reason": "Useful", "by": "agent", "at": STAMP, "revoked_at": None, "revoked_by": None,
    }]
    item["next_decision"] = 2
    item["sheet"] = {
        "path": "sheet.jpg", "sha256": "b" * 64, "width": 80, "height": 80,
        "frames": [{"index": 0, "time_s": 0.0, "column": 0, "row": 0}],
        "colour_converted": False, "at": STAMP,
    }
    item["speech"] = {
        "status": "no_speech", "engine": None, "model": None, "words": 0,
        "text": "", "path": None, "message": None, "at": STAMP,
    }
    item["provenance"] = {"kind": "own", "source": None, "license": None, "url": None}


def test_update_supports_all_reserved_fields_and_returns_mutation_value(document: dict) -> None:
    def enrich(doc: dict) -> str:
        _enrich(doc["items"][0])
        return "saved"

    assert update_catalog(enrich) == "saved"
    expected = document["items"][0]
    _enrich(expected)
    assert get_item("m1") == expected


@pytest.mark.parametrize("field", ["description", "decisions", "sheet", "speech", "frames"])
def test_update_rejects_invalid_nested_fields_without_writing(document: dict, field: str) -> None:
    before = catalog_path().read_bytes()

    def invalidate(doc: dict) -> None:
        item = doc["items"][0]
        _enrich(item)
        if field == "frames":
            item["sheet"]["frames"][0]["extra"] = True
        elif field == "decisions":
            item[field][0]["extra"] = True
        else:
            item[field]["extra"] = True

    with pytest.raises(InputError):
        update_catalog(invalidate)
    assert catalog_path().read_bytes() == before


@pytest.mark.parametrize("defect", ["duplicate", "future", "bool", "nonfinite", "status"])
def test_enrichment_value_validation(document: dict, defect: str) -> None:
    item = document["items"][0]
    _enrich(item)
    if defect == "duplicate":
        item["decisions"].append(copy.deepcopy(item["decisions"][0]))
    elif defect == "future":
        item["decisions"][0]["id"] = "d2"
    elif defect == "bool":
        item["description"]["revision"] = True
    elif defect == "nonfinite":
        item["sheet"]["frames"][0]["time_s"] = float("inf")
    else:
        item["speech"]["status"] = "unknown"
    with pytest.raises(InputError):
        validate_catalog(document, catalog_path())


def test_numeric_id_sorting(document: dict) -> None:
    def populate(doc: dict) -> None:
        first = doc["items"][0]
        for number in (10, 2):
            item = copy.deepcopy(first)
            item.update(id=f"m{number}", sha256=f"{number:064x}", locations=[])
            doc["items"].append(item)
        doc["next_id"] = 11

    update_catalog(populate)
    assert [item["id"] for item in list_items()] == ["m1", "m2", "m10"]


def test_hash_retries_when_content_changes(clip: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_hash = receipt.sha256_file
    hashes = []

    def changing(path: Path) -> str:
        digest = original_hash(path)
        hashes.append(digest)
        if len(hashes) == 1:
            synth.clip(path, colour="red")
        return digest

    monkeypatch.setattr(receipt, "sha256_file", changing)
    assert _add(clip)["added"] == ["m1"]
    assert len(hashes) == 2 and hashes[0] != hashes[1]
    assert get_item("m1")["sha256"] == original_hash(clip)
    assert load_catalog()["index"][str(clip)]["size"] == clip.stat().st_size


def test_hash_stops_after_three_changed_reads(clip: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_hash = receipt.sha256_file
    calls = []

    def changing(path: Path) -> str:
        calls.append(path)
        digest = original_hash(path)
        with path.open("ab") as stream:
            stream.write(b"changed")
        return digest

    monkeypatch.setattr(receipt, "sha256_file", changing)
    result = _add(clip)
    assert len(calls) == 3
    assert result["items"] == 0
    assert "kept changing" in result["skipped"][0]["reason"]
    assert load_catalog()["index"] == {}


def test_completed_batches_survive_an_interrupted_add(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    files = [
        synth.clip(archive / f"{number}.mov", colour=colour)
        for number, colour in enumerate(("blue", "red", "green", "yellow", "white"), start=1)
    ]
    monkeypatch.setattr(catalog, "_BATCH_FILES", 2)
    original_hash = receipt.sha256_file
    hashed = []

    def interrupt_fourth(path: Path) -> str:
        hashed.append(path)
        if len(hashed) == 4:
            raise KeyboardInterrupt
        return original_hash(path)

    monkeypatch.setattr(receipt, "sha256_file", interrupt_fourth)
    with pytest.raises(KeyboardInterrupt):
        _add(archive)
    assert hashed == files[:4]
    document = load_catalog()
    assert validate_catalog(json.loads(catalog_path().read_text()), catalog_path()) == document
    assert [item["id"] for item in document["items"]] == ["m1", "m2"]
    assert [item["locations"][0]["path"] for item in document["items"]] == [
        str(path) for path in files[:2]
    ]
    assert set(document["index"]) == {str(path) for path in files[:2]}
    assert document["next_id"] == 3

    monkeypatch.setattr(receipt, "sha256_file", original_hash)
    result = _add(archive)
    assert result["added"] == ["m3", "m4", "m5"]
    assert result["unchanged"] == ["m1", "m2"]
    assert result["located"] == result["skipped"] == []
    assert result["items"] == 5
    assert [item["id"] for item in list_items()] == ["m1", "m2", "m3", "m4", "m5"]
    assert [item["locations"][0]["path"] for item in list_items()] == [str(path) for path in files]
    assert load_catalog()["next_id"] == 6


def test_batch_results_are_aggregated_and_deduplicated(
    tmp_path: Path, clip: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _add(clip)
    archive = tmp_path / "archive"
    archive.mkdir()
    for number in range(1, 4):
        shutil.copyfile(clip, archive / f"{number}.mov")
    synth.clip(archive / "4.mov", colour="red")
    corrupt = archive / "5.mov"
    corrupt.write_bytes(b"broken")
    link = archive / "linked.mov"
    link.symlink_to(clip)
    monkeypatch.setattr(catalog, "_BATCH_FILES", 2)
    result = _add(clip, archive, archive)
    assert result == {
        "catalog": str(catalog_path()), "added": ["m2"], "located": ["m1"],
        "unchanged": ["m1"], "items": 2,
        "unreadable": [],
        "skipped": [
            {"path": str(corrupt), "reason": "ffmpeg cannot decode it as audio or video"},
            {"path": str(link), "reason": "symbolic link"},
        ],
    }
    assert len(get_item("m1")["locations"]) == 4
    assert load_catalog()["next_id"] == 3


def test_guarded_media_cannot_be_written_by_mutation(clip: Path) -> None:
    before = clip.read_bytes()

    def unsafe(doc: dict) -> None:
        paths.validate_input(clip)
        paths.atomic_write_text(clip, "changed")

    with pytest.raises(OutputPathError, match="input file"):
        update_catalog(unsafe)
    assert clip.read_bytes() == before
    assert not catalog_path().exists()


def test_threaded_adds_do_not_lose_updates(tmp_path: Path, clip: Path) -> None:
    second = synth.png(tmp_path / "second.png")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda path: _add(path), [clip, second]))
    assert sum(len(result["added"]) for result in results) == 2
    assert [item["id"] for item in list_items()] == ["m1", "m2"]
    assert load_catalog()["next_id"] == 3


def test_timestamps_are_converted_to_utc(clip: Path) -> None:
    instant = datetime(2026, 10, 5, 7, 34, 56, tzinfo=timezone(timedelta(hours=-5)))
    _add(clip, now=instant)
    assert get_item("m1")["added_at"] == STAMP


def test_heif_tile_grid_fixture() -> None:
    payload = {"stream_groups": [{
        "index": 0, "type": "Tile Grid", "nb_streams": 48,
        "components": [{
            "nb_tiles": 48, "width": 4032, "height": 3024,
            "coded_width": 4096, "coded_height": 3072,
            "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}],
        }],
        "streams": [{"width": 512, "height": 512}],
    }]}
    assert catalog._tile_grid(payload) == (3024, 4032, 270)
    assert catalog._tile_grid({"stream_groups": []}) is None


def test_heif_extra_probe_failure_preserves_stream_dimensions(
    tmp_path: Path, clip: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = Path(shutil.copyfile(clip, tmp_path / "photo.heic"))
    original = catalog._probe_json

    def unsupported(path: Path, options: list[str]) -> dict:
        if "-show_stream_groups" in options:
            raise InputError(str(path), "unsupported option")
        return original(path, options)

    monkeypatch.setattr(catalog, "_probe_json", unsupported)
    _add(image)
    item = get_item("m1")
    assert item["kind"] == "image" and item["media_type"] == "photo"
    assert (item["width"], item["height"]) == (160, 90)


def test_heif_grid_overrides_tiled_stream_dimensions(
    tmp_path: Path, clip: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = Path(shutil.copyfile(clip, tmp_path / "photo.heif"))
    original = catalog._probe_json
    calls = []

    def with_grid(path: Path, options: list[str]) -> dict:
        if "-show_stream_groups" in options:
            calls.append(path)
            return {"stream_groups": [{
                "type": "Tile Grid", "components": [{
                    "width": 4032, "height": 3024,
                    "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}],
                }],
            }]}
        return original(path, options)

    monkeypatch.setattr(catalog, "_probe_json", with_grid)
    assert _add(image)["added"] == ["m1"]
    item = get_item("m1")
    assert (item["width"], item["height"], item["rotation"]) == (3024, 4032, 270)
    assert item["orientation"] == "vertical"
    assert item["kind"] == "image" and item["duration_s"] is None
    assert calls == [image]


def test_camera_frame_tags_are_trimmed_and_source_order_is_stable(
    clip: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def tags(path: Path, options: list[str]) -> dict:
        if "frame_tags" in options:
            return {"frames": [{"tags": {
                "Make": " Frame Camera ", "Model": " Frame Model ",
                "ExifIFD/LensModel": " Lens ", "DateTimeOriginal": " 2026:10:05 12:34:56 ",
                "GPSPosition": "private", "Software": " " + "s" * 250 + " ",
            }}]}
        return {"streams": [{"tags": {"make": " Stream Camera ", "location": "private"}}]}

    monkeypatch.setattr(catalog, "_probe_json", tags)
    info = SimpleNamespace(format_tags={"com.apple.quicktime.make": " Container Camera "})
    camera = catalog._camera(clip, info, image=True)
    assert camera == {
        "make": "Container Camera", "model": "Frame Model", "software": "s" * 200,
        "lens": "Lens", "captured_at": "2026:10:05 12:34:56", "container_time": None,
    }


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("captured_at", ["2024-01-01T12:00:00", None])
def test_cli_add_list_show_and_location_exists(
    tmp_path: Path, clip: Path, capsys: pytest.CaptureFixture[str], as_json: bool,
    captured_at: str | None,
) -> None:
    flag = ["--json"] if as_json else []
    corrupt = tmp_path / "corrupt.mov"
    corrupt.write_text("broken")
    assert cli.main(["media", "add", str(clip), str(corrupt), *flag]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    if as_json:
        assert json.loads(output.out)["added"] == ["m1"]
        assert json.loads(output.out)["skipped"][0]["path"] == str(corrupt)
    else:
        assert "Added 1, located 0, unchanged 0, skipped 1." in output.out
        assert f"skipped: {corrupt}:" in output.out
        assert str(catalog_path()) in output.out
    assert cli.main(["media", "list", "--type", "clip", *flag]) == 0
    output = capsys.readouterr()
    if as_json:
        assert json.loads(output.out) == list_items(media_type="clip")
    else:
        assert output.out == "m1 clip horizontal 160x90 2.0 s undescribed clip.mov\n"

    def camera_date(data: dict) -> None:
        data["items"][0]["camera"] = {
            "make": "Synthetic", "model": "Camera", "software": None, "lens": None,
            "captured_at": captured_at, "container_time": None,
        }

    update_catalog(camera_date)
    assert cli.main(["media", "show", "m1", *flag]) == 0
    output = capsys.readouterr().out
    if as_json:
        assert json.loads(output)["locations"][0]["exists"] is True
        assert json.loads(output)["camera"]["captured_at"] == captured_at
    else:
        if captured_at is not None:
            assert f"Camera: Synthetic Camera\nCaptured: {captured_at}\n" in output
        else:
            assert "Captured:" not in output
        assert f"Location (exists): {clip}" in output
        assert "ID: m1" in output
        assert "Kind: video; media type: clip" in output
        assert "Size: 160x90; orientation: horizontal" in output
        assert "Duration: 2.000 s; fps: 30/1" in output
        assert "HDR: none" in output and "Audio: yes" in output
        assert "Provenance: unknown" in output and "No description yet." in output
        assert "No sheet yet. Run yap-studio media sheet m1." in output
        assert "Speech: not checked yet; run yap-studio media sheet m1" in output
    assert "exists" not in catalog_path().read_text()
    clip.unlink()
    assert cli.main(["media", "show", "m1", *flag]) == 0
    output = capsys.readouterr().out
    if as_json:
        assert json.loads(output)["locations"][0]["exists"] is False
    else:
        assert f"Location (missing): {clip}" in output
    assert cli._parser().parse_args(["style", "presets"]).style_command == "presets"


@pytest.mark.parametrize("as_json", [False, True])
def test_cli_unknown_id_and_missing_path_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], as_json: bool,
) -> None:
    flag = ["--json"] if as_json else []
    assert cli.main(["media", "show", "m99", *flag]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    if as_json:
        assert json.loads(output.err)["error"]["code"] == "unknown_media"
    else:
        assert "error: Unknown media id: m99" in output.err
        assert "fix: Run yap-studio media list to see catalog ids." in output.err
    missing = tmp_path / "missing.mov"
    assert cli.main(["media", "add", str(missing), *flag]) == 2
    output = capsys.readouterr()
    if as_json:
        assert json.loads(output.err)["error"]["code"] == "invalid_input"
    else:
        assert "file or folder not found" in output.err and "fix:" in output.err
    assert not catalog_path().parent.exists()


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("error", [PermissionError("permission denied"), ValueError("bad path")])
def test_cli_wraps_read_and_write_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    as_json: bool, error: Exception,
) -> None:
    source = tmp_path / "source.mov"

    def fail(paths: list[str]) -> dict:
        raise error

    monkeypatch.setattr(media_commands, "add_paths", fail)
    flag = ["--json"] if as_json else []
    assert cli.main(["media", "add", str(source), *flag]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    expected = InputError(str(source), f"cannot read or write media: {error}")
    if as_json:
        assert json.loads(output.err) == {"error": expected.to_dict()}
    else:
        assert output.err == f"error: {expected.message}\nfix: {expected.suggested_action}\n"
    assert not catalog_path().parent.exists()


def test_cli_list_uses_most_recently_seen_location(
    tmp_path: Path, clip: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _add(clip)
    duplicate = Path(shutil.copyfile(clip, tmp_path / "copy.mov"))
    _add(duplicate, now=NOW + timedelta(hours=1))
    assert cli.main(["media", "list"]) == 0
    assert capsys.readouterr().out.endswith("copy.mov\n")
    _add(clip, now=NOW + timedelta(hours=2))
    assert cli.main(["media", "list"]) == 0
    assert capsys.readouterr().out.endswith("clip.mov\n")
