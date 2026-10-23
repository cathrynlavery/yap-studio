"""Cache validated person mattes of project cuts, publishing complete directories."""

from __future__ import annotations

import os
import re
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from yap_studio import ffmpeg, paths, receipt
from yap_studio.errors import YapError
from yap_studio.matte import chroma
from yap_studio.matte.check import _decoded_count
from yap_studio.matte.composite import _matte_path
from yap_studio.matte.make import make_matte
from yap_studio.matte.model import Cleanup, load_matte
from yap_studio.matte.refine import shot_starts
from yap_studio.matte.timing import require_constant_rate
from yap_studio.project.lock import project_lock

MATTE_IDLE_SECONDS = 10 * 60


class CutMatteError(YapError):
    """Refuse behind graphics when a cut cannot supply an aligned person matte."""

    def __init__(self, message: str, *, timing: bool = False) -> None:
        super().__init__(
            message, error_type="assembly_error", code="cut_matte_invalid",
            suggested_action=(
                "Rebuild the cut with constant-rate timestamps, then assemble again."
                if timing else
                "Run yap-studio doctor, rebuild the cut if needed, then assemble again; "
                "or set the affected elements to layer: front."
            ),
        )


def _validated(folder: Path, cut: Path, info: ffmpeg.MediaInfo) -> Path:
    """Check the document, media identity, exact decode count and alpha geometry."""
    data = load_matte(paths.validate_input(folder / "matte.json"))
    if data.status != "measured":
        raise CutMatteError(f"Behind graphics need a usable person matte: {data.reason}.")
    if (data.video.width, data.video.height, data.video.fps) != (
        info.width, info.height, info.fps
    ):
        raise CutMatteError("Cut matte document dimensions or frame rate differ from the cut.")
    matte = _matte_path(cut, info, folder)
    assert matte is not None
    if _decoded_count(matte) != info.video_frames:
        raise CutMatteError("Cut matte decoded frame count differs from the cut.")
    return matte


@dataclass(frozen=True)
class _CachedMatte:
    path: Path
    identity: tuple[tuple[int, ...], ...]


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _entry_identity(folder: Path, matte: Path) -> tuple[tuple[int, ...], ...]:
    """Cheap entry identity; usage touches must not invalidate the directory."""
    directory = folder.stat()
    return ((directory.st_dev, directory.st_ino),
            _file_identity((folder / "matte.json").stat()), _file_identity(matte.stat()))


def _cached(folder: Path, cut: Path, info: ffmpeg.MediaInfo,
            joins: Sequence[int] = ()) -> _CachedMatte | None:
    """Validate without the project lock and reject changes during validation."""
    try:
        directory = folder.stat()
        document = _file_identity((folder / "matte.json").stat())
        data = load_matte(paths.validate_input(folder / "matte.json"))
        if data.backend.get("name") == "vision-person-segmentation" and (
            data.cleanup != Cleanup() or data.joins != tuple(joins)
        ):
            return None
        if data.matte is None:
            return None
        expected = folder / data.matte["file"]
        identity = ((directory.st_dev, directory.st_ino), document,
                    _file_identity(expected.stat()))
        matte = _validated(folder, cut, info)
        if _entry_identity(folder, matte) != identity:
            return None
        return _CachedMatte(matte, identity)
    except (YapError, OSError, ValueError):
        return None


def _still_cached(folder: Path, cached: _CachedMatte) -> bool:
    """Confirm the validated files are unchanged using only metadata under lock."""
    try:
        return _entry_identity(folder, cached.path) == cached.identity
    except OSError:
        return False


def _mark_used(folder: Path) -> None:
    """Refresh the entry's use time under the same lock as retirement."""
    paths._guard_write(folder)
    now = time.time()
    os.utime(folder, (now, now), follow_symlinks=False)


def _retire_other_mattes(cache: Path, current: Path, work: Path) -> None:
    """Atomically retire idle cut-hash directories, leaving files and links alone.

    Returning an entry refreshes its directory mtime under the publication lock.
    Keep it for at least ten minutes so another render can still open its matte.
    Rename each idle entry into temporary storage before removing it, without
    exposing a partly deleted cache entry.
    """
    retired = work / "retired"
    retired.mkdir()
    cutoff = time.time() - MATTE_IDLE_SECONDS
    for entry in cache.iterdir():
        if (entry == current or re.fullmatch(r"[0-9a-f]{64}", entry.name) is None
                or entry.is_symlink() or not entry.is_dir()):
            continue
        if entry.stat().st_mtime >= cutoff:
            continue
        paths._guard_write(entry)
        os.replace(entry, retired / entry.name)


def ensure_cut_matte(project_dir: Path, cut: Path, *, cutout: str = "auto") -> Path:
    """Reuse a valid cut-hash cache entry or atomically publish a checked replacement.

    Assembly supplies its retained, verified cut. Build beside the cache so the
    final directory rename is atomic. A failed rebuild leaves the old entry in
    place. Hashing and decoding run outside the project lock. Under the short
    transaction lock, confirm the validated entry's metadata and refresh its use
    timestamp, or publish the checked replacement. Publication retires only other
    entries idle for more than ten minutes, outside the visible cache.
    """
    with paths.operation():
        source = paths.validate_input(cut)
        try:
            info = ffmpeg.probe(source, count_frames=True)
            if (not info.has_video or not info.video_frames or not info.width
                    or not info.height or info.fps is None or info.fps <= 0):
                raise CutMatteError(
                    "Behind graphics need a cut with usable picture and frame count."
                )
            try:
                require_constant_rate(source, info)
            except YapError as exc:
                raise CutMatteError(
                    "Behind graphics need a constant-rate cut: " + str(exc), timing=True,
                ) from exc
            decoded = _decoded_count(source)
            if decoded != info.video_frames:
                raise CutMatteError("Cut decoded frame count differs from its reported count.")
            info = replace(info, video_frames=decoded)
            spans = cut.parent / "spans.json"
            joins = shot_starts(spans, frames=decoded) if spans.exists() else ()
            digest = receipt.sha256_file(source)
            cache = paths.validate_output_dir(Path(project_dir) / ".yap/cache/matte")
            folder = cache / digest
            paths._guard_write(folder)
            cached = _cached(folder, source, info, joins)
            cached = chroma.for_request(cached, cutout)
            if cached is not None:
                with project_lock(Path(project_dir)):
                    if _still_cached(folder, cached):
                        _mark_used(folder)
                        return cached.path
            with tempfile.TemporaryDirectory(prefix=".matte-", dir=cache) as temporary:
                work = Path(temporary)
                staged = work / "new"
                chroma.make_cut_matte(source, staged, cutout, make_matte, joins=joins)
                matte = _validated(staged, source, info)
                if receipt.sha256_file(source) != digest:
                    raise CutMatteError("Cut changed while its person matte was being made.")
                cached = _cached(folder, source, info, joins)
                cached = chroma.for_request(cached, cutout)
                with project_lock(Path(project_dir)):
                    if cached is not None and _still_cached(folder, cached):
                        _mark_used(folder)
                        _retire_other_mattes(cache, folder, work)
                        return cached.path
                    paths._guard_write(folder)
                    old = work / "old"
                    if folder.exists():
                        os.replace(folder, old)
                    try:
                        os.replace(staged, folder)
                    except BaseException:
                        if old.exists():
                            os.replace(old, folder)
                        raise
                    _mark_used(folder)
                    _retire_other_mattes(cache, folder, work)
                return folder / matte.name
        except CutMatteError:
            raise
        except (YapError, OSError) as exc:
            raise CutMatteError("Could not prepare the cut person matte: " + str(exc)) from exc
