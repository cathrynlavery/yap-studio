"""Album exports use only synthetic records and the fake exporter."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from yap_studio import cli
from yap_studio.media.album import DEFAULT_WAIT_S, _edited, _matches, _progress, _run

FAKE = Path(__file__).parent / "fakes/osxphotos"


def record(number: int, **changes: object) -> dict:
    return {"uuid": f"synthetic-{number}", "original_filename": f"IMG_{number:04}.HEIC",
            "date": f"2024-01-{number:02}T12:00:00", "ismissing": False,
            "ismovie": False, "isphoto": True, "live_photo": False,
            "original_filesize": 16, **changes}


@pytest.fixture
def album(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    _no_real_osxphotos: Callable[[Path], None],
) -> tuple[Path, Path, Path]:
    _no_real_osxphotos(FAKE)
    monkeypatch.setenv("PATH", str(FAKE.parent) + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr(sys, "platform", "darwin")
    records = tmp_path / "records.json"
    records.write_text(json.dumps([
        record(1, live_photo=True),
        record(2, original_filename="IMG_0002.MOV", ismovie=True, isphoto=False),
        record(3, ismissing=True),
    ]))
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_OSXPHOTOS_RECORDS", str(records))
    monkeypatch.setenv("FAKE_OSXPHOTOS_LOG", str(log))
    return tmp_path / "export", records, log


def calls(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_album_dry_run_counts_without_writes(
    album: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    out, _, log = album
    assert cli.main(["media", "album", "Sample", "--out", str(out), "--dry-run", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["counts"] == {"items": 3, "photos": 2, "videos": 1, "live_photos": 1,
                                "edited": 0, "files": 0, "not_downloaded": 1, "already_present": 0}
    assert result["dry_run"] is True and result["report"] is None
    assert result["library"] is None
    assert result["download"] is True
    assert result["exported"] == 0
    assert not out.exists()
    assert [entry["argv"] for entry in calls(log)] == [["query", "--album", "Sample", "--json"]]


def test_album_export_report_and_resume(
    album: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    out, _, log = album
    args = ["media", "album", "Sample", "--out", str(out)]
    assert cli.main(args) == 0
    captured = capsys.readouterr()
    text = captured.out
    assert "0 edited; 0 not downloaded" in text
    assert "1 items are not downloaded" in captured.err and "click Allow" in captured.err
    assert text.rstrip().endswith(f"next: yap-studio media add {out}")
    report = json.loads((out / "album-report.json").read_text())
    assert report["schema"] == "yap-studio/album-export@1"
    assert report["album"] == "Sample" and report["written_at"]
    assert report["library"] is None
    assert report["counts"] == {"items": 3, "photos": 2, "videos": 1, "live_photos": 1,
                               "edited": 0, "files": 4, "not_downloaded": 0}
    assert [item["kind"] for item in report["files"]] == ["photo", "live_motion", "video", "photo"]
    assert report["files"][0] == {"file": "IMG_0001.HEIC", "captured": "2024-01-01T12:00:00",
                                   "bytes": 16, "kind": "photo", "uuid": "synthetic-1",
                                   "version": "original"}
    assert report["not_downloaded"] == []
    export = calls(log)[1]["argv"]
    assert calls(log)[1]["uuids"] == ["synthetic-1", "synthetic-2", "synthetic-3"]
    assert {"--uuid-from-file", "--skip-original-if-edited", "--update", "--no-progress",
            "--download-missing", "--verbose"} <= set(export)
    assert "--skip-edited" not in export
    assert "--album" not in export and "--cleanup" not in export
    assert cli.main([*args, "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["exported"] == 0 and result["counts"]["already_present"] == 3
    assert len([entry for entry in calls(log) if entry["argv"][0] == "export"]) == 1
    assert calls(log)[-1]["argv"][0] == "query"


def test_album_rerun_restores_missing_live_motion(
    album: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    out, records, log = album
    records.write_text(json.dumps([record(1, live_photo=True)]))
    args = ["media", "album", "Sample", "--out", str(out), "--json"]
    assert cli.main(args) == 0
    capsys.readouterr()
    (out / "IMG_0001.mov").unlink()
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    exports = [entry for entry in calls(log) if entry["argv"][0] == "export"]
    assert len(exports) == 2
    assert exports[-1]["uuids"] == ["synthetic-1"]
    report = json.loads((out / "album-report.json").read_text())
    motion = [item for item in report["files"] if item["kind"] == "live_motion"]
    assert motion and all((out / item["file"]).is_file() for item in motion)
    assert result["counts"]["already_present"] == 0
    assert cli.main(args) == 0
    capsys.readouterr()
    assert calls(log)[-1]["argv"][0] == "query"


def test_album_matches_hand_exports_by_size_case_and_collision_name(
    album: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    out, records, log = album
    records.write_text(json.dumps([record(1, live_photo=True), record(2, original_filesize=None)]))
    out.mkdir()
    (out / "IMG_0001.HEIC").write_bytes(b"wrong size")
    (out / "img_0001 (1).heic").write_bytes(b"x" * 16)
    (out / "img_0001 (1).mov").write_bytes(b"synthetic motion")
    (out / "IMG_0002.HEIC").write_bytes(b"unknown size")
    (out / "extra.jpg").write_bytes(b"extra")
    (out / ".hidden.jpg").write_bytes(b"hidden")
    (out / "link.jpg").symlink_to(out / "extra.jpg")
    assert cli.main(["media", "album", "Sample", "--out", str(out), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["counts"]["already_present"] == 2
    assert len(calls(log)) == 1
    report = json.loads((out / "album-report.json").read_text())
    assert len(report["files"]) == 5
    assert report["files"][0]["file"] == "img_0001 (1).heic"
    extra = next(row for row in report["files"] if row["file"] == "IMG_0001.HEIC")
    assert extra["uuid"] is None and extra["version"] is None
    assert report["files"][-1]["captured"] is None
    assert report["files"][-1]["version"] is None


def test_album_shared_filename_pairs_live_motion_and_resumes(
    album: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    out, records, log = album
    records.write_text(json.dumps([
        record(1, live_photo=True, original_filesize=32),
        record(2, original_filename="IMG_0001.HEIC", live_photo=True, original_filesize=16),
    ]))
    args = ["media", "album", "Sample", "--out", str(out), "--json"]
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["exported"] == result["counts"]["files"] == 4
    assert result["counts"]["not_downloaded"] == 0
    report = json.loads((out / "album-report.json").read_text())
    for uuid, size in (("synthetic-1", 32), ("synthetic-2", 16)):
        pair = [row for row in report["files"] if row["uuid"] == uuid]
        assert len(pair) == 2
        assert {row["kind"] for row in pair} == {"photo", "live_motion"}
        assert len({Path(row["file"]).stem for row in pair}) == 1
        assert all(row["version"] == "original" for row in pair)
        assert next(row["bytes"] for row in pair if row["kind"] == "photo") == size
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["counts"]["already_present"] == 2 and result["exported"] == 0
    assert [entry["argv"][0] for entry in calls(log)] == ["query", "export", "query"]


@pytest.mark.parametrize("command", ["query", "export"])
def test_album_stall_stops_process_group(
    album: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], command: str,
) -> None:
    out, _, log = album
    out.mkdir()
    kept = out / "kept.jpg"
    kept.write_bytes(b"synthetic earlier export")
    monkeypatch.setenv("FAKE_OSXPHOTOS_HANG", command)
    started = time.monotonic()
    assert cli.main(["media", "album", "Sample", "--out", str(out), "--wait", "1"]) == 2
    assert time.monotonic() - started < 8
    text = capsys.readouterr().err
    assert "click Allow on the Mac" in text and "files already exported are kept" in text
    with pytest.raises(ProcessLookupError):
        os.killpg(calls(log)[-1]["pgid"], 0)
    assert kept.read_bytes() == b"synthetic earlier export"
    assert all(not Path(entry["tmpdir"]).exists() for entry in calls(log))


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_album_platform_and_missing_tool_give_hand_export_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], platform: str,
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    out = tmp_path / "export"
    assert cli.main(["media", "album", "Sample", "--out", str(out)]) == 2
    error = capsys.readouterr().err
    assert "by hand" in error and f"yap-studio media add {out}" in error
    assert ("uv tool install osxphotos" if platform == "darwin" else "only on a Mac") in error
    assert not out.exists()


@pytest.mark.parametrize("contents", ["[]", "{}", "bad json"])
def test_album_empty_or_bad_query(
    album: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str], contents: str,
) -> None:
    out, records, _ = album
    records.write_text(contents)
    assert cli.main(["media", "album", "Sample", "--out", str(out)]) == 2
    error = capsys.readouterr().err
    if contents == "[]":
        assert ("No items found in album Sample in the Photos library "
                "synthetic/Photos Library.photoslibrary.") in error
        assert "Folder/Album" in error and "pass --library PATH" in error
    assert not out.exists()


@pytest.mark.parametrize("dry_run", [False, True])
def test_album_library_reaches_query_and_export_and_is_recorded(
    album: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str], dry_run: bool,
) -> None:
    out, _, log = album
    library = out.parent / "Other Photos Library.photoslibrary"
    library.mkdir()
    args = ["media", "album", "Sample", "--out", str(out), "--library", str(library), "--json"]
    assert cli.main([*args, *(["--dry-run"] if dry_run else [])]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    if not dry_run:
        assert f"Open {library} in Photos first" in captured.err
    else:
        assert not captured.err
    assert result["library"] == str(library)
    entries = calls(log)
    expected = ["query"] if dry_run else ["query", "export"]
    assert [entry["argv"][0] for entry in entries] == expected
    for entry in entries:
        argv = entry["argv"]
        assert argv[argv.index("--library") + 1] == str(library)
    if not dry_run:
        report = json.loads((out / "album-report.json").read_text())
        assert report["library"] == str(library)
    else:
        assert not out.exists()


def test_album_missing_library_refuses_before_running_exporter(
    album: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    out, _, log = album
    library = out.parent / "missing.photoslibrary"
    assert cli.main(["media", "album", "Sample", "--out", str(out),
                     "--library", str(library)]) == 2
    error = capsys.readouterr().err
    assert str(library) in error and "does not exist" in error
    assert not log.exists() and not out.exists()


def test_album_empty_names_explicit_library(
    album: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    out, records, _ = album
    records.write_text("[]")
    library = out.parent / "Other Photos Library.photoslibrary"
    library.mkdir()
    assert cli.main(["media", "album", "Sample", "--out", str(out),
                     "--library", str(library)]) == 2
    error = capsys.readouterr().err
    assert f"No items found in album Sample in the Photos library {library}." in error
    assert "Folder/Album" in error and "pass --library PATH" in error
    assert not out.exists()


def test_album_empty_without_library_diagnostic_keeps_message_and_hint(
    album: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, records, _ = album
    records.write_text("[]")
    monkeypatch.setenv("FAKE_OSXPHOTOS_NO_LIBRARY_NOTE", "1")
    assert cli.main(["media", "album", "Sample", "--out", str(out)]) == 2
    error = capsys.readouterr().err
    assert "No items found in album Sample." in error
    assert "Folder/Album" in error and "pass --library PATH" in error
    assert not out.exists()


def test_album_query_keeps_stderr_separate_from_json(
    album: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    out, _, _ = album
    assert cli.main(["media", "album", "Sample", "--out", str(out), "--dry-run"]) == 0
    assert "3 items" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["query", "export"])
def test_album_nonzero_exit_reports_log_tail(
    album: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], command: str,
) -> None:
    out, _, _ = album
    monkeypatch.setenv("FAKE_OSXPHOTOS_FAIL", command)
    assert cli.main(["media", "album", "Sample", "--out", str(out)]) == 2
    assert "Synthetic exporter failure." in capsys.readouterr().err


@pytest.mark.parametrize("stderr", ["Synthetic stderr failure.", ""])
def test_album_runner_error_uses_stderr_tail_or_stdout_fallback(
    tmp_path: Path, stderr: str,
) -> None:
    from yap_studio.errors import YapError

    code = ("import sys; print('Synthetic stdout failure.'); "
            f"sys.stderr.write({stderr!r}); sys.exit(4)")
    with pytest.raises(YapError) as caught:
        _run(sys.executable, ["-c", code], tmp_path, 5)
    assert (stderr or "Synthetic stdout failure.") in caught.value.message
    if stderr:
        assert "Synthetic stdout failure." not in caught.value.message


def test_album_no_download_then_default_downloads_only_missing(
    album: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    out, _, log = album
    args = ["media", "album", "Sample", "--out", str(out)]
    assert cli.main([*args, "--no-download"]) == 1
    captured = capsys.readouterr()
    assert "--no-download" in captured.out and "were skipped" in captured.out
    assert not captured.err
    assert "--download-missing" not in calls(log)[-1]["argv"]
    report = json.loads((out / "album-report.json").read_text())
    assert report["not_downloaded"] == [{"original_filename": "IMG_0003.HEIC",
                                         "captured": "2024-01-03T12:00:00",
                                         "uuid": "synthetic-3"}]
    assert cli.main([*args, "--json"]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["download"] is True
    assert result["exported"] == 1 and result["counts"]["not_downloaded"] == 0
    assert "click Allow" in captured.err
    assert calls(log)[-1]["uuids"] == ["synthetic-3"]
    assert "--download-missing" in calls(log)[-1]["argv"]


@pytest.mark.parametrize("download", [False, True])
def test_album_dry_run_download_notice_is_in_summary(
    album: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str], download: bool,
) -> None:
    out, _, _ = album
    args = ["media", "album", "Sample", "--out", str(out), "--dry-run"]
    assert cli.main([*args, *([] if download else ["--no-download"])]) == 0
    captured = capsys.readouterr()
    assert not captured.err
    expected = "The export downloads them." if download else "--no-download skips them."
    assert expected in captured.out
    assert ("click Allow" in captured.out) is download
    assert not out.exists()


@pytest.mark.parametrize(
    "local,download,slow_mo,movie,live,motion,version",
    [(True, True, False, False, False, False, "edited"),
     (True, False, False, False, False, False, "edited"),
     (True, True, True, False, False, False, "edited"),
     (True, True, False, True, False, False, "edited"),
     (True, True, False, False, True, True, "edited"),
     (False, True, False, False, True, False, "edited"),
     (False, True, False, False, False, False, "edited"),
     (False, False, False, False, False, False, "original"),
     (False, True, True, True, False, False, "original")],
)
def test_album_wanted_version_and_resume(
    album: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str],
    local: bool, download: bool, slow_mo: bool, movie: bool, live: bool, motion: bool, version: str,
) -> None:
    out, records, log = album
    source = out.parent / "synthetic-edited"
    source.write_bytes(b"synthetic edited media")
    original = "IMG_0001.MOV" if movie else "IMG_0001.HEIC"
    records.write_text(json.dumps([record(
        1, hasadjustments=True, path_edited=str(source) if local else None,
        path_edited_live_photo=str(source) if motion else None, slow_mo=slow_mo,
        uti_edited="com.apple.quicktime-movie" if movie else "public.jpeg",
        original_filename=original, ismovie=movie, isphoto=not movie, live_photo=live,
    )]))
    args = ["media", "album", "Sample", "--out", str(out)]
    if not download:
        args.append("--no-download")
    assert cli.main([*args, "--dry-run", "--json"]) == 0
    captured = capsys.readouterr()
    counts = json.loads(captured.out)["counts"]
    assert counts["edited"] == (version == "edited")
    assert counts["not_downloaded"] == (not local and version == "edited")
    assert not captured.err
    assert cli.main([*args, "--json"]) == 0
    captured = capsys.readouterr()
    assert ("click Allow" in captured.err) is (not local and version == "edited")
    result = json.loads(captured.out)
    assert result["download"] is download and result["counts"]["not_downloaded"] == 0
    report = json.loads((out / "album-report.json").read_text())
    filename = "IMG_0001_edited.mov" if movie else "IMG_0001_edited.jpeg"
    if version == "original":
        filename = original
    assert report["files"][0]["file"] == filename
    assert report["files"][0]["kind"] == ("video" if movie else "photo")
    assert all(row["version"] == version for row in report["files"])
    assert len(report["files"]) == (2 if motion else 1)
    if motion:
        assert report["files"][1]["file"] == "IMG_0001_edited.mov"
        assert report["files"][1]["kind"] == "live_motion"
    if version == "edited":
        assert not (out / original).exists()
    assert source.read_bytes() == b"synthetic edited media"
    assert cli.main(args) == 0
    assert not capsys.readouterr().err
    assert calls(log)[-1]["argv"][0] == "query"
    assert all(not Path(entry["tmpdir"]).exists() for entry in calls(log))


def test_album_original_filename_takes_priority_over_another_items_edit(
    album: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    out, records, log = album
    records.write_text(json.dumps([
        record(1, hasadjustments=True, path_edited=None, uti_edited="public.jpeg"),
        record(2, original_filename="IMG_0001_edited.jpeg", original_filesize=24),
    ]))
    out.mkdir()
    (out / "IMG_0001_edited.jpeg").write_bytes(b"x" * 24)
    args = ["media", "album", "Sample", "--out", str(out), "--json"]
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["counts"]["already_present"] == 1 and result["exported"] == 1
    assert calls(log)[-1]["uuids"] == ["synthetic-1"]
    report = json.loads((out / "album-report.json").read_text())
    rows = {row["file"]: row for row in report["files"]}
    assert rows["IMG_0001_edited.jpeg"]["uuid"] == "synthetic-2"
    assert rows["IMG_0001_edited.jpeg"]["version"] == "original"
    assert rows["IMG_0001_edited (1).jpeg"]["uuid"] == "synthetic-1"
    assert rows["IMG_0001_edited (1).jpeg"]["version"] == "edited"
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["counts"]["already_present"] == 2 and result["exported"] == 0
    assert [entry["argv"][0] for entry in calls(log)] == ["query", "export", "query"]


def test_album_nonempty_edit_takes_priority_when_size_is_unknown(
    album: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    out, records, log = album
    records.write_text(json.dumps([
        record(1, hasadjustments=True, path_edited=None, uti_edited="public.jpeg"),
    ]))
    out.mkdir()
    (out / "IMG_0001_edited (1).jpeg").write_bytes(b"")
    (out / "IMG_0001_edited (2).jpeg").write_bytes(b"x" * 12)
    assert cli.main(["media", "album", "Sample", "--out", str(out), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["counts"]["already_present"] == 1 and result["exported"] == 0
    assert [entry["argv"][0] for entry in calls(log)] == ["query"]
    report = json.loads((out / "album-report.json").read_text())
    rows = {row["file"]: row for row in report["files"]}
    assert rows["IMG_0001_edited (2).jpeg"]["uuid"] == "synthetic-1"
    assert rows["IMG_0001_edited (2).jpeg"]["version"] == "edited"
    assert rows["IMG_0001_edited (1).jpeg"]["uuid"] is None
    assert rows["IMG_0001_edited (1).jpeg"]["version"] is None


def test_album_leftover_original_and_wrong_edited_size_do_not_satisfy_export(
    album: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str],
) -> None:
    out, records, log = album
    source = out.parent / "synthetic-edited.jpeg"
    source.write_bytes(b"synthetic edited file")
    records.write_text(json.dumps([record(1, hasadjustments=True, path_edited=str(source),
                                         uti_edited="public.jpeg")]))
    out.mkdir()
    (out / "IMG_0001.HEIC").write_bytes(b"x" * 16)
    args = ["media", "album", "Sample", "--out", str(out), "--json"]
    assert cli.main(args) == 0
    captured = capsys.readouterr()
    assert not captured.err
    assert json.loads(captured.out)["counts"]["already_present"] == 0
    report = json.loads((out / "album-report.json").read_text())
    assert report["files"][0] == {"file": "IMG_0001.HEIC", "captured": "2024-01-01T12:00:00",
                                   "bytes": 16, "kind": "photo", "uuid": "synthetic-1",
                                   "version": "original"}
    (out / "IMG_0001_edited.jpeg").write_bytes(b"short")
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["counts"]["already_present"] == 0
    assert calls(log)[-1]["uuids"] == ["synthetic-1"]
    assert (out / "IMG_0001_edited (1).jpeg").stat().st_size == source.stat().st_size
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["counts"]["already_present"] == 1
    assert calls(log)[-1]["argv"][0] == "query"
    report = json.loads((out / "album-report.json").read_text())
    extra = next(row for row in report["files"] if row["file"] == "IMG_0001_edited.jpeg")
    assert extra["uuid"] is None and extra["version"] is None


@pytest.mark.parametrize("where", ["db", "tmp", "library", "inferred_library"])
def test_album_silent_download_progress_does_not_stop(
    album: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], where: str,
) -> None:
    out, _, log = album
    library = out.parent / "Synthetic.photoslibrary"
    library.mkdir()
    source = "library" if where == "inferred_library" else where
    monkeypatch.setenv("FAKE_OSXPHOTOS_SLOW", f"{source}:3")
    if where == "inferred_library":
        monkeypatch.setenv("FAKE_OSXPHOTOS_LIBRARY", str(library))
    args = ["media", "album", "Sample", "--out", str(out), "--wait", "1"]
    if where == "library":
        args.extend(["--library", str(library)])
    assert cli.main(args) == 0
    assert "no progress" not in capsys.readouterr().err
    assert (out / "IMG_0003.HEIC").is_file()
    assert all(not Path(entry["tmpdir"]).exists() for entry in calls(log))


def test_album_progress_detects_renames_and_same_size_writes(tmp_path: Path) -> None:
    destination, scratch, library = (tmp_path / name for name in ("export", "scratch", "library"))
    destination.mkdir()
    scratch.mkdir()
    staging = library / "resources/cpl/cloudsync.noindex/storage/tmp"
    staging.mkdir(parents=True)
    for root in (destination, scratch, staging):
        target = root / ".progress"
        target.write_bytes(b"x")
        before = _progress(destination, scratch, library)
        info = target.stat()
        os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
        assert _progress(destination, scratch, library) != before
        before = _progress(destination, scratch, library)
        target.rename(root / ".renamed")
        assert _progress(destination, scratch, library) != before
    nested = scratch / "photoscript_synthetic/deep"
    nested.mkdir(parents=True)
    before = _progress(destination, scratch, library)
    (nested / "download").write_bytes(b"x")
    assert _progress(destination, scratch, library) != before


@pytest.mark.parametrize("movie,extension", [(False, ".jpeg"), (False, ".heic"), (True, ".mov")])
def test_album_edited_filename_matching(movie: bool, extension: str) -> None:
    assert _matches(Path(f"img_0001_edited (2){extension}"), "IMG_0001.HEIC",
                    edited=True, movie=movie)
    assert not _matches(Path("IMG_0001_edited.mov"), "IMG_0001.HEIC", edited=True)
    assert _matches(Path("IMG_0001_edited.mov"), "IMG_0001.HEIC", edited=True, motion=True)
    assert not _matches(Path("IMG_0001.jpeg"), "IMG_0001.HEIC")


def test_album_no_adjustments_never_selects_edited() -> None:
    assert not _edited(record(1, path_edited="synthetic"), True)


def test_album_wait_default() -> None:
    args = cli._parser().parse_args(["media", "album", "Sample", "--out", "synthetic"])
    assert args.wait == DEFAULT_WAIT_S == 1800
    assert args.download is True
