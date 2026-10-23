"""Review persistence, hash cache, and settings tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import pytest

from yap_studio import receipt
from yap_studio.errors import InputError, OutputPathError, ValidationError
from yap_studio.review import store as store_module
from yap_studio.review.export import format_text, to_export_json
from yap_studio.review.media import file_identity
from yap_studio.review.model import MediaFacts
from yap_studio.review.series import SeriesEntry
from yap_studio.review.store import MediaHash, ReviewStore, UnknownCommentError, data_dir

SHA = "a" * 64
NOW = datetime(2026, 9, 29, 12, 34, 56, tzinfo=UTC)


def test_pin_store_and_old_unpinned_bytes(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path / "data", now=lambda: NOW)
    old = store.add_comment(SHA, _facts(), _payload(3))
    file = store.root / "reviews" / f"{SHA}.json"
    before = file.read_bytes()
    assert store.comments(SHA) == [old]
    assert file.read_bytes() == before and b'"pin"' not in before
    record_before = json.dumps(json.loads(before)["comments"][0]).encode()
    pinned = store.add_comment(SHA, _facts(), {**_payload(10), "end_frame": 20,
                                             "pin": {"x": 0.5, "y": 0.6, "frame": 15}})
    assert json.dumps(json.loads(file.read_bytes())["comments"][0]).encode() == record_before
    assert store.comments(SHA)[1].pin == pinned.pin
    with pytest.raises(ValidationError, match="move or remove the pin"):
        store.update_comment(SHA, _facts(), pinned.id, {"end_frame": 15})
    assert store.update_comment(SHA, _facts(), pinned.id, {"kind": "keep"}).pin == pinned.pin
    assert store.update_comment(SHA, _facts(), pinned.id, {"pin": None}).pin is None


def _facts(sha: str = SHA) -> MediaFacts:
    return MediaFacts("sample.mp4", sha, 12.0, Fraction(30000, 1001), True)


def _payload(frame: int = 0) -> dict:
    return {"text": "Useful note", "start_frame": frame}


def test_series_link_is_durable_private_and_preserves_port(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path / "data")
    series = store.open_series([SeriesEntry(SHA, "first.mp4")])
    first = store.series_link(series.id)
    assert first["schema"] == "yap-studio/review-link@1" and first["series"] == series.id
    assert first["port"] is None and len(first["token"]) >= 20
    path = store.root / "links" / f"{series.id}.json"
    assert path.stat().st_mode & 0o777 == 0o600
    reopened = ReviewStore(store.root)
    assert reopened.series_link(series.id) == first
    reopened.save_link_port(series.id, 12345)
    assert store.series_link(series.id) == {**first, "port": 12345}
    assert path.stat().st_mode & 0o777 == 0o600
    other = store.open_series([SeriesEntry("b" * 64, "second.mp4")])
    assert store.series_link(other.id)["token"] != first["token"]


def test_concurrent_series_links_create_one_token(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path / "data")
    series = store.open_series([SeriesEntry(SHA, "first.mp4")])
    with ThreadPoolExecutor(max_workers=8) as pool:
        links = list(pool.map(lambda _: ReviewStore(store.root).series_link(series.id), range(8)))
    assert all(item == links[0] for item in links)


@pytest.mark.parametrize("field,value", [("schema", "foreign"), ("series", "b" * 64),
                                         ("port", 0), ("port", True), ("token", "bad")])
def test_corrupt_series_link_is_refused_without_overwrite(tmp_path: Path, field, value) -> None:
    store = ReviewStore(tmp_path / "data")
    series = store.open_series([SeriesEntry(SHA, "first.mp4")])
    data = store.series_link(series.id)
    path = store.root / "links" / f"{series.id}.json"
    data[field] = value
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    with pytest.raises(InputError, match="not a valid review link"):
        store.series_link(series.id)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("system", "env", "expected"),
    [
        ("darwin", {}, "Library/Application Support/yap-studio"),
        ("win32", {"APPDATA": "/roaming"}, "/roaming/yap-studio"),
        ("win32", {}, "AppData/Roaming/yap-studio"),
        ("linux", {}, ".local/share/yap-studio"),
        ("linux", {"XDG_DATA_HOME": "/data"}, "/data/yap-studio"),
        ("linux", {"XDG_DATA_HOME": "relative"}, ".local/share/yap-studio"),
    ],
)
def test_data_dir(system: str, env: dict, expected: str, tmp_path: Path) -> None:
    result = data_dir(env, system, tmp_path)
    assert result == (Path(expected) if expected.startswith("/") else tmp_path / expected)


def test_data_dir_override_expands_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert data_dir({"YAP_STUDIO_DATA_DIR": "~/reviews"}, "linux", tmp_path) == (
        tmp_path / "reviews"
    )


def test_persistence_order_timestamps_and_ids(tmp_path: Path) -> None:
    root = tmp_path / "data"
    store = ReviewStore(root, now=lambda: NOW)
    assert not root.exists()
    second = store.add_comment(SHA, _facts(), _payload(8))
    first = store.add_comment(SHA, _facts(), _payload(2))
    assert (second.id, first.id) == ("c1", "c2")
    assert second.created_at == second.updated_at == "2026-09-29T12:34:56Z"
    reopened = ReviewStore(root, now=lambda: NOW)
    assert [comment.id for comment in reopened.comments(SHA)] == ["c2", "c1"]
    document = json.loads((root / "reviews" / f"{SHA}.json").read_text())
    assert document["schema"] == "yap-studio/review@2"
    assert set(document) == {"schema", "media", "next_id", "comments"}
    assert document["media"] == {
        "name": "sample.mp4",
        "sha256": SHA,
        "duration_s": 12.0,
        "fps": "30000/1001",
        "cfr": True,
    }
    assert document["next_id"] == 3
    assert reopened.review_facts(SHA) == _facts()

    later = datetime(2026, 9, 30, 1, 2, 3, tzinfo=UTC)
    patched = ReviewStore(root, now=lambda: later).update_comment(
        SHA, _facts(), "c1", {"text": "Changed"}
    )
    assert patched.created_at == "2026-09-29T12:34:56Z"
    assert patched.updated_at == "2026-09-30T01:02:03Z"
    assert patched.start_frame == 8
    reopened.delete_comment(SHA, "c1")
    assert reopened.add_comment(SHA, _facts(), _payload()).id == "c3"
    assert [item.id for item in ReviewStore(root).comments(SHA)] == ["c3", "c2"]


def test_current_records_without_request_load_and_next_write_includes_it(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path / "data", now=lambda: NOW)
    first = store.add_comment(SHA, _facts(), _payload())
    file = store.root / "reviews" / f"{SHA}.json"
    data = json.loads(file.read_text())
    assert data["comments"][0]["request"] is None
    del data["comments"][0]["request"]
    file.write_text(json.dumps(data))
    before = file.read_bytes()
    assert ReviewStore(store.root).comments(SHA) == [first]
    assert file.read_bytes() == before
    request = {"type": "spelling", "from": "tail net", "to": "tailnet"}
    added = store.add_comment(SHA, _facts(), {**_payload(3), "target": "caption",
                                             "request": request})
    data = json.loads(file.read_text())
    assert [record["request"] for record in data["comments"]] == [None, request]
    assert ReviewStore(store.root).comments(SHA)[1] == added
    with pytest.raises(ValidationError, match="a spelling fix needs"):
        store.update_comment(SHA, _facts(), added.id, {"kind": "keep"})
    assert json.loads(file.read_text()) == data
    assert store.update_comment(SHA, _facts(), added.id,
                                {"kind": "keep", "request": None}).request is None


def test_unwritable_review_root_reports_storage_error(tmp_path: Path) -> None:
    if os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0):
        pytest.skip("chmod permissions are not enforced for this user")
    parent = tmp_path / "read-only"
    parent.mkdir()
    parent.chmod(0o500)
    root = parent / "data"
    try:
        with pytest.raises(OutputPathError) as error:
            ReviewStore(root).add_comment(SHA, _facts(), _payload())
        assert error.value.code == "review_storage"
        assert str(root) in error.value.message
        assert error.value.suggested_action == (
            "Set YAP_STUDIO_DATA_DIR to a writable folder."
        )
    finally:
        parent.chmod(0o700)


def _write_legacy_review(root: Path, labels: list[str]) -> Path:
    path = root / "reviews" / f"{SHA}.json"
    path.parent.mkdir(parents=True)
    records = [
        {"id": f"c{number}", "label": label, "text": f"Note {number}",
         "start_frame": number * 3, "end_frame": None, "start_s": number * 0.1001,
         "end_s": None, "created_at": "2026-09-29T12:34:56Z",
         "updated_at": "2026-09-29T12:34:56Z"}
        for number, label in enumerate(labels, 1)
    ]
    document = {"schema": "yap-studio/review@1", "media": ReviewStore._media(_facts()),
                "next_id": 9, "comments": records}
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def test_keep_and_fix_on_the_same_video_export_their_kinds(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path, now=lambda: NOW)
    keep = store.add_comment(SHA, _facts(), {**_payload(), "kind": "keep", "target": "story"})
    fix = store.add_comment(SHA, _facts(), {**_payload(5), "kind": "fix", "target": "caption"})
    comments = store.comments(SHA)
    assert comments == [keep, fix]
    exported = to_export_json(_facts(), comments, None, NOW)
    assert [(item["kind"], item["target"]) for item in exported["comments"]] == [
        ("keep", "story"), ("fix", "caption")
    ]
    text = format_text(_facts(), comments, None, NOW.date())
    assert "[Keep: story]" in text and "[Fix: caption]" in text


@pytest.mark.parametrize("labels", [["change", "change"], ["like", "change"]])
def test_legacy_reads_and_exports_leave_file_bytes_unchanged(
    tmp_path: Path, labels: list[str]
) -> None:
    path = _write_legacy_review(tmp_path, labels)
    original = path.read_bytes()
    store = ReviewStore(tmp_path, now=lambda: NOW)
    comments = store.comments(SHA)
    kinds = ["keep" if label == "like" else "fix" for label in labels]
    assert [comment.kind for comment in comments] == kinds
    assert all(comment.target is None for comment in comments)
    assert store.review_facts(SHA) == _facts()
    exported = to_export_json(_facts(), comments, None, NOW)
    assert [item["kind"] for item in exported["comments"]] == kinds
    assert all("label" not in item and item["target"] is None for item in exported["comments"])
    assert format_text(_facts(), comments, None, NOW.date()).count("[Fix]") == kinds.count("fix")
    assert path.read_bytes() == original


@pytest.mark.parametrize("operation", ["add", "update", "delete"])
def test_first_legacy_write_upgrades_all_records_and_preserves_sequence(
    tmp_path: Path, operation: str
) -> None:
    path = _write_legacy_review(tmp_path, ["like", "change"])
    original = json.loads(path.read_text())
    store = ReviewStore(tmp_path, now=lambda: NOW)
    if operation == "add":
        assert store.add_comment(SHA, _facts(), _payload(10)).id == "c9"
    elif operation == "update":
        store.update_comment(SHA, _facts(), "c2", {"target": "caption"})
    else:
        store.delete_comment(SHA, "c2")
    document = json.loads(path.read_text())
    assert document["schema"] == "yap-studio/review@2"
    assert document["next_id"] == (10 if operation == "add" else 9)
    assert document["comments"][0] == {
        **{key: value for key, value in original["comments"][0].items() if key != "label"},
        "kind": "keep", "target": None, "request": None,
    }
    assert [item["id"] for item in document["comments"]] == (
        ["c1", "c2", "c9"] if operation == "add" else
        ["c1", "c2"] if operation == "update" else ["c1"]
    )
    assert all("label" not in item for item in document["comments"])
    assert ReviewStore(tmp_path).comments(SHA)[0].kind == "keep"


@pytest.mark.parametrize(
    "malformation", ["new_in_old", "old_in_new", "missing_target", "bad_label"]
)
def test_schema_requires_its_own_record_shape_and_leaves_invalid_file_untouched(
    tmp_path: Path, malformation: str
) -> None:
    path = _write_legacy_review(tmp_path, ["change"])
    document = json.loads(path.read_text())
    record = document["comments"][0]
    if malformation in ("new_in_old", "missing_target"):
        del record["label"]
        record["kind"] = "fix"
        record["target"] = None
    if malformation in ("old_in_new", "missing_target"):
        document["schema"] = "yap-studio/review@2"
    if malformation == "missing_target":
        del record["target"]
    if malformation == "bad_label":
        record["label"] = "other"
    path.write_text(json.dumps(document), encoding="utf-8")
    original = path.read_bytes()
    store = ReviewStore(tmp_path)
    for action in (
        lambda: store.comments(SHA),
        lambda: store.review_facts(SHA),
        lambda: store.add_comment(SHA, _facts(), _payload()),
        lambda: store.update_comment(SHA, _facts(), "c1", {"kind": "keep"}),
        lambda: store.delete_comment(SHA, "c1"),
    ):
        with pytest.raises(InputError) as error:
            action()
        assert error.value.reason == (
            "not a valid review file; move it aside to start a new review"
        )
        assert path.read_bytes() == original


def test_invalid_current_kind_is_refused_and_file_bytes_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "reviews" / f"{SHA}.json"
    path.parent.mkdir()
    document = {
        "schema": "yap-studio/review@2",
        "media": ReviewStore._media(_facts()),
        "next_id": 2,
        "comments": [{
            "id": "c1",
            "kind": "maybe",
            "target": None,
            "text": "Note",
            "start_frame": 3,
            "end_frame": None,
            "start_s": 0.1001,
            "end_s": None,
            "created_at": "2026-09-29T12:34:56Z",
            "updated_at": "2026-09-29T12:34:56Z",
        }],
    }
    original = (json.dumps(document, indent=2) + "\n").encode("utf-8")
    path.write_bytes(original)
    store = ReviewStore(tmp_path, now=lambda: NOW)
    for action in (
        lambda: store.comments(SHA),
        lambda: store.add_comment(SHA, _facts(), _payload()),
    ):
        with pytest.raises(InputError) as error:
            action()
        assert error.value.reason == (
            "not a valid review file; move it aside to start a new review"
        )
        assert path.read_bytes() == original


def test_unwritable_lock_file_reports_storage_error(tmp_path: Path) -> None:
    if os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0):
        pytest.skip("chmod permissions are not enforced for this user")
    root = tmp_path / "data"
    root.mkdir()
    lock = root / ".lock"
    lock.write_bytes(b"")
    lock.chmod(0o400)
    try:
        with pytest.raises(OutputPathError) as error:
            ReviewStore(root).add_comment(SHA, _facts(), _payload())
        assert error.value.code == "review_storage"
        assert error.value.suggested_action == (
            "Set YAP_STUDIO_DATA_DIR to a writable folder."
        )
    finally:
        lock.chmod(0o600)


def test_atomic_write_error_reports_storage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_write(_path: Path, _text: str) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(store_module.paths, "atomic_write_text", fail_write)
    root = tmp_path / "data"
    with pytest.raises(OutputPathError) as error:
        ReviewStore(root).add_comment(SHA, _facts(), _payload())
    assert error.value.code == "review_storage"
    assert error.value.message == f"Cannot write review data in {root}: No space left on device"


def test_review_facts_empty_and_corrupt(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path)
    assert store.review_facts(SHA) is None
    path = tmp_path / "reviews" / f"{SHA}.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {"schema": "yap-studio/review@1", "media": {}, "next_id": 1, "comments": []}
        ),
        encoding="utf-8",
    )
    assert store.review_facts(SHA) is None
    invalid = {
        "schema": "yap-studio/review@1",
        "media": {"name": "sample.mp4"},
        "next_id": 1,
        "comments": [],
    }
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(InputError):
        store.add_comment(SHA, _facts(), _payload())
    assert json.loads(path.read_text()) == invalid
    path.write_text("{bad", encoding="utf-8")
    with pytest.raises(InputError):
        store.review_facts(SHA)


def test_unknown_id_on_update_and_delete(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path)
    for action in (
        lambda: store.update_comment(SHA, _facts(), "c9", {"text": "missing"}),
        lambda: store.delete_comment(SHA, "c9"),
    ):
        with pytest.raises(UnknownCommentError) as error:
            action()
        assert error.value.parameter == "comment_id"


def test_update_seconds_only_comment_after_media_is_classified_constant_rate(
    tmp_path: Path,
) -> None:
    store = ReviewStore(tmp_path / "data")
    variable_facts = MediaFacts("sample.mp4", SHA, 12.0, None, False)
    stored = store.add_comment(SHA, variable_facts, {"text": "Original", "start_s": 1.2})
    updated = store.update_comment(SHA, _facts(), stored.id, {"text": "Updated"})
    assert updated.text == "Updated"
    assert updated.start_frame == 35
    assert updated.start_s == 1.1678
    assert store.comments(SHA) == [updated]


def test_hash_cache_and_previous_review_survives_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"first")
    original = receipt.sha256_file
    calls: list[Path] = []

    def counted(path: Path) -> str:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(receipt, "sha256_file", counted)
    root = tmp_path / "data"
    store = ReviewStore(root, now=lambda: NOW)
    first, previous, identity = store.hash_media(source)
    assert previous is None
    assert identity == file_identity(source)
    assert store.hash_media(source) == MediaHash(first, None, identity)
    assert len(calls) == (1 if store_module._TRUST_FILE_METADATA else 2)
    store.add_comment(first, _facts(first), _payload())
    source.write_bytes(b"second version")
    second, previous, identity = store.hash_media(source)
    assert second != first
    assert previous == first
    assert ReviewStore(root).hash_media(source) == MediaHash(second, first, identity)
    assert len(calls) == (2 if store_module._TRUST_FILE_METADATA else 4)
    index = json.loads((root / "media-index.json").read_text())
    assert index["schema"] == "yap-studio/media-index@1"
    assert set(index) == {"schema", "entries"}
    index = index["entries"]
    assert index[str(source.resolve())]["previous_sha256"] == first
    assert index[str(source.resolve())]["size"] == len(b"second version")
    assert index[str(source.resolve())]["ctime_ns"] == source.stat().st_ctime_ns
    assert index[str(source.resolve())]["ino"] == source.stat().st_ino
    assert index[str(source.resolve())]["dev"] == source.stat().st_dev
    store.delete_comment(first, "c1")
    assert ReviewStore(root).hash_media(source) == MediaHash(second, None, identity)
    assert len(calls) == (2 if store_module._TRUST_FILE_METADATA else 5)


def test_hash_cache_rehashes_when_device_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"video")
    original = receipt.sha256_file
    calls: list[Path] = []

    def counted(path: Path) -> str:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(store_module, "_TRUST_FILE_METADATA", True)
    monkeypatch.setattr(receipt, "sha256_file", counted)
    root = tmp_path / "data"
    store = ReviewStore(root)
    first, _, identity = store.hash_media(source)
    index_path = root / "media-index.json"
    index = json.loads(index_path.read_text())
    key = str(source.resolve())
    index["entries"][key]["dev"] = source.stat().st_dev + 1
    index_path.write_text(json.dumps(index), encoding="utf-8")

    assert store.hash_media(source) == MediaHash(first, None, identity)
    assert len(calls) == 2
    assert json.loads(index_path.read_text())["entries"][key]["dev"] == source.stat().st_dev
    assert store.hash_media(source) == MediaHash(first, None, identity)
    assert len(calls) == 2


def test_hash_cache_detects_same_size_change_with_restored_mtime(tmp_path: Path) -> None:
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"first")
    store = ReviewStore(tmp_path / "data")
    first, _, _ = store.hash_media(source)
    store.add_comment(first, _facts(first), _payload())
    before = source.stat()
    source.write_bytes(b"other")
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    second, previous, _ = store.hash_media(source)
    assert second != first
    assert previous == first


def test_hash_cache_rehashes_when_metadata_is_untrusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"video")
    original = receipt.sha256_file
    calls: list[Path] = []

    def counted(path: Path) -> str:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(store_module, "_TRUST_FILE_METADATA", False)
    monkeypatch.setattr(receipt, "sha256_file", counted)
    store = ReviewStore(tmp_path / "data")
    first, _, identity = store.hash_media(source)
    assert store.hash_media(source) == MediaHash(first, None, identity)
    assert len(calls) == 2


def test_hash_retries_when_file_changes_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"first")
    original = receipt.sha256_file
    calls = 0

    def changing(path: Path) -> str:
        nonlocal calls
        calls += 1
        digest = original(path)
        if calls == 1:
            source.write_bytes(b"final version")
        return digest

    monkeypatch.setattr(receipt, "sha256_file", changing)
    result = ReviewStore(tmp_path / "data").hash_media(source)
    assert calls == 2
    assert result.sha256 == original(source)
    assert result.identity == file_identity(source)
    assert result.previous_sha256 is None


def test_hash_fails_when_file_keeps_changing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"first")
    original = receipt.sha256_file
    calls = 0

    def changing(path: Path) -> str:
        nonlocal calls
        calls += 1
        digest = original(path)
        source.write_bytes(f"version {calls}".encode())
        return digest

    monkeypatch.setattr(receipt, "sha256_file", changing)
    with pytest.raises(InputError) as error:
        ReviewStore(tmp_path / "data").hash_media(source)
    assert calls == 3
    assert error.value.path == str(source)
    assert error.value.reason == (
        "the file kept changing while it was being hashed; "
        "wait for the export to finish and try again"
    )


def test_hash_reports_input_error_when_file_disappears_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"video")
    original = receipt.sha256_file
    calls = 0

    def disappearing(path: Path) -> str:
        nonlocal calls
        calls += 1
        digest = original(path)
        source.unlink()
        return digest

    monkeypatch.setattr(receipt, "sha256_file", disappearing)
    with pytest.raises(InputError) as error:
        ReviewStore(tmp_path / "data").hash_media(source)
    assert calls == 1
    assert error.value.path == str(source)
    assert error.value.reason == (
        "the file disappeared or could not be read while it was being hashed; "
        "try again once the export has finished"
    )


def test_changed_file_does_not_offer_empty_earlier_review(tmp_path: Path) -> None:
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"first")
    store = ReviewStore(tmp_path / "data")
    first, _, _ = store.hash_media(source)
    source.write_bytes(b"second version")
    second, previous, _ = store.hash_media(source)
    assert first != second
    assert previous is None
    index = json.loads((tmp_path / "data" / "media-index.json").read_text())
    assert index["entries"][str(source.resolve())]["previous_sha256"] is None


def test_earlier_review_survives_an_unreviewed_intermediate_version(tmp_path: Path) -> None:
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"first")
    store = ReviewStore(tmp_path / "data")
    first, _, _ = store.hash_media(source)
    store.add_comment(first, _facts(first), _payload())
    source.write_bytes(b"second version")
    second, previous, _ = store.hash_media(source)
    assert previous == first
    source.write_bytes(b"third version")
    third, previous, identity = store.hash_media(source)
    assert third not in (first, second)
    assert previous == first
    assert ReviewStore(tmp_path / "data").hash_media(source) == MediaHash(third, first, identity)


@pytest.mark.parametrize("file_kind", ["review", "index", "settings"])
def test_corrupt_file_is_named_and_untouched(tmp_path: Path, file_kind: str) -> None:
    root = tmp_path / "data"
    store = ReviewStore(root)
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"video")
    if file_kind == "review":
        path = root / "reviews" / f"{SHA}.json"
    elif file_kind == "index":
        path = root / "media-index.json"
    else:
        path = root / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(InputError) as error:
        if file_kind == "review":
            store.add_comment(SHA, _facts(), _payload())
        elif file_kind == "index":
            store.hash_media(source)
        else:
            store.set_reviewer("Alex Reviewer")
    assert error.value.path == str(path)
    assert "move it aside" in error.value.reason
    assert path.read_text(encoding="utf-8") == "{broken"


def test_invalid_review_shape_is_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "reviews" / f"{SHA}.json"
    path.parent.mkdir()
    path.write_text('{"schema":"yap-studio/review@1","comments":{}}')
    with pytest.raises(InputError):
        ReviewStore(tmp_path).add_comment(SHA, _facts(), _payload())
    assert '"comments":{}' in path.read_text()


@pytest.mark.parametrize(
    ("file_kind", "schema"),
    [
        ("review", "yap-studio/review@2"),
        ("index", "yap-studio/media-index@1"),
        ("settings", "yap-studio/review-settings@1"),
    ],
)
def test_store_documents_can_be_overwritten_twice(
    tmp_path: Path, file_kind: str, schema: str
) -> None:
    root = tmp_path / "data"
    store = ReviewStore(root)
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"first")
    if file_kind == "review":
        path = root / "reviews" / f"{SHA}.json"

        def write(number: int) -> None:
            store.add_comment(SHA, _facts(), _payload(number))

    elif file_kind == "index":
        path = root / "media-index.json"

        def write(number: int) -> None:
            source.write_bytes(f"version {number}".encode())
            store.hash_media(source)

    else:
        path = root / "settings.json"

        def write(number: int) -> None:
            store.set_reviewer(f"Reviewer {number}")

    for number in range(3):
        write(number)
        assert json.loads(path.read_text())["schema"] == schema


@pytest.mark.parametrize("file_kind", ["review", "index", "settings"])
def test_wrong_schema_is_not_overwritten(tmp_path: Path, file_kind: str) -> None:
    root = tmp_path / "data"
    store = ReviewStore(root)
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"video")
    if file_kind == "review":
        path = root / "reviews" / f"{SHA}.json"
        document = {"schema": "yap-studio/review@3", "media": {}, "next_id": 1, "comments": []}

        def write() -> None:
            store.add_comment(SHA, _facts(), _payload())

    elif file_kind == "index":
        path = root / "media-index.json"
        document = {"schema": "yap-studio/media-index@2", "entries": {}}

        def write() -> None:
            store.hash_media(source)

    else:
        path = root / "settings.json"
        document = {"schema": "yap-studio/review-settings@2", "reviewer": None}

        def write() -> None:
            store.set_reviewer("Reviewer")

    path.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps(document)
    path.write_text(original, encoding="utf-8")
    with pytest.raises(InputError) as error:
        write()
    assert error.value.path == str(path)
    assert "move it aside" in error.value.reason
    assert path.read_text(encoding="utf-8") == original


def test_reviewer_validation_and_clearing(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path)
    assert store.reviewer() is None
    store.set_reviewer("  Alex Reviewer  ")
    assert ReviewStore(tmp_path).reviewer() == "Alex Reviewer"
    for bad in ("x" * 101, "Alex\nReviewer", "Alex\x00Reviewer", 4):
        with pytest.raises(ValidationError) as error:
            store.set_reviewer(bad)
        assert error.value.parameter == "reviewer"
        assert store.reviewer() == "Alex Reviewer"
    store.set_reviewer("  ")
    assert store.reviewer() is None
    store.set_reviewer("A")
    store.set_reviewer(None)
    assert json.loads((tmp_path / "settings.json").read_text()) == {
        "schema": "yap-studio/review-settings@1",
        "reviewer": None,
    }


def test_concurrent_additions_have_unique_ids_and_no_lost_comments(tmp_path: Path) -> None:
    store = ReviewStore(tmp_path, now=lambda: NOW)
    with ThreadPoolExecutor(max_workers=8) as pool:
        comments = list(
            pool.map(lambda frame: store.add_comment(SHA, _facts(), _payload(frame)), range(40))
        )
    assert {item.id for item in comments} == {f"c{number}" for number in range(1, 41)}
    assert len(ReviewStore(tmp_path).comments(SHA)) == 40
    document = json.loads((tmp_path / "reviews" / f"{SHA}.json").read_text())
    assert document["next_id"] == 41


def test_process_additions_have_unique_ids_and_no_lost_comments(tmp_path: Path) -> None:
    script = """
import sys
from fractions import Fraction
from pathlib import Path
from yap_studio.review.export import format_text, to_export_json
from yap_studio.review.model import MediaFacts
from yap_studio.review.store import ReviewStore

root, sha = sys.argv[1:]
store = ReviewStore(Path(root))
facts = MediaFacts("sample.mp4", sha, 12.0, Fraction(30), True)
for number in range(10):
    store.add_comment(sha, facts, {"text": "Useful note", "start_frame": number})
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src") + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path), SHA],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    try:
        for process in processes:
            _, stderr = process.communicate(timeout=120)
            assert process.returncode == 0, stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)
    document = json.loads((tmp_path / "reviews" / f"{SHA}.json").read_text())
    assert len(document["comments"]) == 40
    assert {item["id"] for item in document["comments"]} == {
        f"c{number}" for number in range(1, 41)
    }
    assert document["next_id"] == 41
