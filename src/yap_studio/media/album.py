"""Export album items to a local media folder."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from yap_studio import tools
from yap_studio.errors import YapError
from yap_studio.media.catalog import _IMAGE_EXTENSIONS, _VIDEO_EXTENSIONS

# osxphotos may stay quiet 1445 s on its own: 600 s to open Photos, 120 s/export, one retry.
DEFAULT_WAIT_S = 1800


def _error(message: str, fix: str | None = None) -> YapError:
    return YapError(message, error_type="media_error", code="album_export_failed",
                    suggested_action=fix)


def _media_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir()
                  if not path.name.startswith(".") and not path.is_symlink() and path.is_file()
                  and path.suffix.lower() in _IMAGE_EXTENSIONS | _VIDEO_EXTENSIONS)


def _progress(folder: Path, scratch: Path, library: Path | None) -> tuple:
    entries = []

    def scan(root: Path, recursive: bool = False) -> None:
        try:
            with os.scandir(root) as children:
                for entry in children:
                    try:
                        info = os.lstat(entry.path)
                    except OSError:
                        continue
                    entries.append((entry.path, info.st_size, info.st_mtime_ns))
                    if recursive and stat.S_ISDIR(info.st_mode):
                        scan(Path(entry.path), True)
        except OSError:
            pass

    scan(folder)
    scan(scratch, True)
    if library is not None:
        scan(library / "resources/cpl/cloudsync.noindex/storage/tmp")
    return tuple(sorted(entries))


def _stop(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _run(executable: str, args: list[str], folder: Path, wait: float,
         library: Path | None = None) -> tuple[str, str]:
    """Stop the whole exporter session when its log and destination stop changing."""
    with (tempfile.TemporaryDirectory(prefix="yap-album-", ignore_cleanup_errors=True) as scratch,
          tempfile.TemporaryFile() as log, tempfile.TemporaryFile() as stdout):
        process = subprocess.Popen([executable, *args], stdin=subprocess.DEVNULL,
                                   stdout=stdout, stderr=log, start_new_session=True,
                                   env={**os.environ, "TMPDIR": scratch})
        previous = (0, 0, _progress(folder, Path(scratch), library))
        changed = time.monotonic()
        try:
            while process.poll() is None:
                state = (os.fstat(log.fileno()).st_size, os.fstat(stdout.fileno()).st_size,
                         _progress(folder, Path(scratch), library))
                now = time.monotonic()
                if state != previous:
                    changed = now
                    previous = state
                if now - changed >= wait:
                    _stop(process)
                    raise _error(
                        f"osxphotos made no progress for {wait:g} s. macOS may be showing a "
                        "permission prompt: click Allow on the Mac, then run this again; files "
                        f"already exported are kept. Or export the album by hand into {folder} "
                        "(in Photos: select the album's items, then File > Export > Export Items)."
                    )
                time.sleep(min(1, max(0.01, wait - (now - changed))))
        except BaseException:
            if process.poll() is None:
                _stop(process)
            raise
        log.seek(0)
        stderr = log.read().decode("utf-8", errors="replace")
        stdout.seek(0)
        output = stdout.read().decode("utf-8", errors="replace")
        if process.returncode:
            raise _error(f"osxphotos exited {process.returncode}: "
                         + "\n".join((stderr.strip() or output).splitlines()[-10:]))
        return output, stderr


def _edited(record: dict, download: bool) -> bool:
    if not record.get("hasadjustments"):
        return False
    if record.get("path_edited"):
        return True
    if record.get("slow_mo"):
        return False
    return download


def _local(record: dict, download: bool) -> bool:
    return bool(record.get("path_edited")) if _edited(record, download) else not record.get(
        "ismissing"
    )


def _matches(path: Path, original: str, *, motion: bool = False, edited: bool = False,
             movie: bool = False) -> bool:
    name = Path(original)
    extensions = ({".mov"} if motion else (_VIDEO_EXTENSIONS if movie else _IMAGE_EXTENSIONS)
                  if edited else {name.suffix})
    stem = name.stem + ("_edited" if edited else "")
    pattern = re.escape(stem) + r"(?: \(\d+\))?(?:" + "|".join(
        re.escape(ext) for ext in extensions
    ) + ")"
    return re.fullmatch(pattern, path.name, re.IGNORECASE) is not None


def _complete(path: Path, size: int | None, edited: bool) -> bool:
    actual = path.stat().st_size
    return actual == size if size is not None else actual > 0 if edited else True


def _inventory(records: list[dict], folder: Path, download: bool) -> tuple[list[dict], set[int]]:
    files = _media_files(folder)
    remaining = set(files)
    rows = []
    complete_versions = set()
    for edited in (False, True):
        version = "edited" if edited else "original"
        for index, record in enumerate(records):
            if edited and not record.get("hasadjustments"):
                continue
            size = None if edited else record.get("original_filesize")
            if edited and record.get("path_edited"):
                try:
                    info = os.stat(record["path_edited"])
                    if stat.S_ISREG(info.st_mode):
                        size = info.st_size
                except OSError:
                    pass
            matches = [path for path in files if path in remaining
                       and _matches(path, record["original_filename"], edited=edited,
                                    movie=bool(record.get("ismovie")))]
            matches.sort(key=lambda path: (not _complete(path, size, edited), path.name))
            matched = matches[0] if matches else None
            complete = False
            if matched is not None:
                remaining.remove(matched)
                complete = _complete(matched, size, edited)
                rows.append(_file_row(matched, record,
                                      "video" if record.get("ismovie") else "photo", version))
            if record.get("live_photo"):
                motions = [path for path in files if path in remaining
                           and _matches(path, record["original_filename"], motion=True,
                                        edited=edited)]
                motions.sort(key=lambda path: (
                    matched is None or path.stem.lower() != matched.stem.lower(), path.name
                ))
                if motions:
                    motion = motions[0]
                    remaining.remove(motion)
                    rows.append(_file_row(motion, record, "live_motion", version))
                if not edited or record.get("path_edited_live_photo"):
                    complete = complete and bool(motions)
            if complete:
                complete_versions.add((index, edited))
    present = {index for index, record in enumerate(records)
               if (index, _edited(record, download)) in complete_versions}
    rows.extend(_file_row(path, {}, "video" if path.suffix.lower() in _VIDEO_EXTENSIONS
                          else "photo") for path in remaining)
    rows.sort(key=lambda row: (row["captured"] is None, row["captured"] or "", row["file"]))
    return rows, present


def _file_row(path: Path, record: dict, kind: str, version: str | None = None) -> dict:
    return {"file": path.name, "captured": record.get("date"), "bytes": path.stat().st_size,
            "kind": kind, "uuid": record.get("uuid"), "version": version}


def _write_report(folder: Path, report: dict) -> Path:
    target = folder / "album-report.json"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=folder, prefix=".album-", delete=False,
                                         encoding="utf-8") as output:
            temporary = Path(output.name)
            json.dump(report, output, indent=2, ensure_ascii=False, allow_nan=False)
            output.write("\n")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def export_album(album: str, folder: Path, *, dry_run: bool = False,
                 download: bool = True, wait: float = DEFAULT_WAIT_S,
                 library: Path | None = None, notify: Callable[[str], None] | None = None) -> dict:
    """List an album, export absent items, and report the resulting files."""
    folder = folder.expanduser().absolute()
    hand = (f"Export the album by hand into {folder}, then run "
            f"yap-studio media add {folder}.")
    if sys.platform != "darwin":
        raise _error("Photos albums export only on a Mac.", hand)
    if library is not None:
        library = library.expanduser().absolute()
        if not library.exists():
            raise _error(f"Photos library {library} does not exist.")
    library_path = str(library) if library is not None else None
    library_args = ["--library", library_path] if library_path is not None else []
    executable = tools.find_tool("osxphotos")
    if executable is None:
        error = tools._missing_tool("osxphotos")
        error.suggested_action = (
            "Install it with uv tool install osxphotos, or export the album by hand from "
            f"Photos into {folder} (select the album's items, File > Export > Export "
            f"Items), then run yap-studio media add {folder}."
        )
        raise error
    if not math.isfinite(wait) or wait <= 0:
        raise _error("--wait must be a positive number of seconds.")
    if folder.exists() and not folder.is_dir():
        raise _error(f"{folder} is not a folder.")
    output, diagnostics = _run(
        executable, ["query", *library_args, "--album", album, "--json"], folder, wait, library,
    )
    try:
        records = json.loads(output)
    except ValueError as exc:
        raise _error("osxphotos query did not return valid JSON.") from exc
    if (not isinstance(records, list)
            or any(not isinstance(row, dict) or not isinstance(row.get("uuid"), str)
                   or not isinstance(row.get("original_filename"), str)
                   or Path(row["original_filename"]).name != row["original_filename"]
                   or not row["original_filename"] or "\n" in row["uuid"]
                   or "\r" in row["uuid"] for row in records)):
        raise _error("osxphotos query did not return a list of album records.")
    match = re.search(r"^Using last opened Photos library: (.+)$", diagnostics, re.MULTILINE)
    searched_library = library_path or (match[1].strip() if match else None)
    if not records:
        where = f" in the Photos library {searched_library}" if searched_library else ""
        raise _error(
            f"No items found in album {album}{where}.",
            'Check the album name in Photos; an album inside a folder is "Folder/Album". '
            "If the album is in another library, pass --library PATH.",
        )
    before = {path.name for path in _media_files(folder)}
    files, present = _inventory(records, folder, download)
    already_present = len(present)
    if not dry_run:
        folder.mkdir(parents=True, exist_ok=True)
        requested = [row["uuid"] for index, row in enumerate(records) if index not in present
                     and (_local(row, download) or download)]
        if requested:
            not_local = sum(index not in present and not _local(row, download)
                            for index, row in enumerate(records))
            if download and not_local and notify is not None:
                notice = (f"{not_local} items are not downloaded to this Mac. Photos will download "
                          "them now and macOS may ask for permission: stay at the Mac to click "
                          "Allow.")
                if library is not None:
                    notice += (f" Open {library} in Photos first; downloads go through the library "
                               "Photos has open.")
                notify(notice)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8") as uuids:
                uuids.write("\n".join(requested) + "\n")
                uuids.flush()
                args = ["export", str(folder), "--uuid-from-file", uuids.name,
                        "--skip-original-if-edited", "--update", "--no-progress", "--verbose",
                        *library_args]
                if download:
                    args.append("--download-missing")
                _run(executable, args, folder, wait,
                     Path(searched_library) if searched_library else None)
        files, present = _inventory(records, folder, download)
    missing = [{"original_filename": row["original_filename"], "captured": row.get("date"),
                "uuid": row["uuid"]} for index, row in enumerate(records)
               if not _local(row, download) and index not in present]
    counts = {"items": len(records), "photos": sum(bool(row.get("isphoto")) for row in records),
              "videos": sum(bool(row.get("ismovie")) for row in records),
              "live_photos": sum(bool(row.get("live_photo")) for row in records),
              "edited": sum(_edited(row, download) for row in records),
              "files": len(files), "not_downloaded": len(missing)}
    report = None
    if not dry_run:
        report = _write_report(folder, {
            "schema": "yap-studio/album-export@1", "album": album, "library": library_path,
            "written_at": datetime.now().astimezone().isoformat(), "counts": counts,
            "files": files, "not_downloaded": missing,
        })
        if len(present) + len(missing) < len(records):
            raise _error("Some album items were not exported. " + hand)
    return {"counts": {**counts, "already_present": already_present}, "dry_run": dry_run,
            "library": library_path, "download": download,
            "exported": len({row["file"] for row in files} - before),
            "report": str(report) if report else None,
            "next": f"yap-studio media add {folder}" if files else None}


def summary(result: dict, folder: Path) -> str:
    counts = result["counts"]
    text = (f"{counts['items']} items: {counts['photos']} photos, {counts['videos']} videos, "
            f"{counts['live_photos']} live photos, {counts['edited']} edited; "
            f"{counts['not_downloaded']} not downloaded "
            f"to this Mac; {counts['already_present']} already in {folder}.")
    if counts["not_downloaded"]:
        if result["dry_run"]:
            text += ("\nThe export downloads them. Photos may ask this Mac for permission, so the "
                     "creator should be at the Mac to click Allow. Add --no-download to take only "
                     "what is on this Mac."
                     if result["download"] else "\n--no-download skips them.")
        elif result["download"]:
            text += (f"\n{counts['not_downloaded']} items did not download. If Photos asked for "
                     "permission, click Allow on the Mac and run this again, or export them by "
                     f"hand into {folder}.")
        else:
            text += (f"\n{counts['not_downloaded']} items are not downloaded to this Mac and were "
                     "skipped. Run again without --no-download to download them, or export them "
                     f"by hand into {folder}.")
    if result["next"]:
        text += "\nnext: " + result["next"]
    return text
