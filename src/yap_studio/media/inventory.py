"""Mirror the creator's catalog into a private, offline HyperFrames inventory."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from yap_studio import cache, paths, process
from yap_studio.errors import DependencyError, OutputPathError, ProcessingError, ValidationError
from yap_studio.media.catalog import load_catalog
from yap_studio.render import hyperframes
from yap_studio.render.hyperframes import RendererInstall

INVENTORY_SCHEMA = "yap-studio/media-inventory@1"
# These extension sets mirror the pinned HyperFrames renderer's adopt support.
_ADOPTABLE_VIDEO_EXTENSIONS = frozenset({".mp4", ".webm", ".mov"})
_ADOPTABLE_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico"})
_LOCK_TIMEOUT_S = 30.0
_RETRY = (
    "Check that catalog files are readable and the pinned renderer is installed, "
    "then run yap-studio media inventory sync again."
)


def inventory_root() -> Path:
    """Return the derived inventory's folder without creating it."""
    return cache.cache_root() / "media-inventory"


def _lock_error() -> DependencyError:
    return DependencyError(
        "Another operation holds the media inventory lock.",
        error_type="dependency_error",
        code="media_inventory_locked",
        suggested_action="Wait for the media inventory operation to finish, then try again.",
    )


@contextmanager
def _inventory_lock(root: Path, *, timeout_s: float | None = None) -> Iterator[None]:
    """Serialize sync and search across processes using a stable sibling lock file."""
    paths.validate_output_dir(root.parent)
    lock_file = root.parent / ".media-inventory.lock"
    if lock_file.is_symlink():
        raise OutputPathError("Media inventory lock must not be a symlink.", code="unsafe_path")
    descriptor = os.open(lock_file, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    locked = False
    deadline = time.monotonic() + (_LOCK_TIMEOUT_S if timeout_s is None else timeout_s)
    try:
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        raise
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise _lock_error() from exc
                    time.sleep(min(0.05, remaining))
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                        raise
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise _lock_error() from exc
                    time.sleep(min(0.05, remaining))
        locked = True
        yield
    finally:
        try:
            if locked:
                if os.name == "nt":
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60].rstrip("-") or "media"


@dataclass(frozen=True)
class _MirrorLink:
    """One catalog item and the selected source and relative inventory path."""

    item: dict
    source: Path
    relative: Path


@dataclass(frozen=True)
class _MirrorPlan:
    """The links, skipped items, and fingerprint for one current catalog snapshot."""

    links: list[_MirrorLink]
    skipped: list[dict]
    sha256: str


def _mirror_plan(catalog: dict) -> _MirrorPlan:
    """Choose adoptable existing locations and fingerprint every mirror decision."""
    links, skipped, decisions = [], [], []
    for item in catalog["items"]:
        adoptable = (
            _ADOPTABLE_VIDEO_EXTENSIONS if item["kind"] == "video"
            else _ADOPTABLE_IMAGE_EXTENSIONS
        )
        source, first_existing = None, None
        for location in item["locations"]:
            candidate = Path(location["path"])
            try:
                if not candidate.is_file():
                    continue
            except OSError:
                continue
            if first_existing is None:
                first_existing = candidate
            if candidate.suffix.lower() in adoptable:
                source = candidate
                break
        description = item["description"]
        text = description["text"] if description is not None else None
        relative, reason = None, None
        if source is None:
            reason = "no file on disk" if first_existing is None else (
                "format not adopted by the HyperFrames inventory "
                f"({first_existing.suffix.lower()})"
            )
            skipped.append({"id": item["id"], "reason": reason})
        else:
            name = f"{_slug(text or source.stem)}-{item['id']}{source.suffix.lower()}"
            relative = Path("assets") / f"{item['kind']}s" / name
            links.append(_MirrorLink(item, source, relative))
        decisions.append({
            "id": item["id"], "sha256": item["sha256"], "kind": item["kind"],
            "description": text, "source": str(source) if source is not None else None,
            "path": relative.as_posix() if relative is not None else None, "skipped": reason,
        })
    fingerprint = json.dumps(
        sorted(decisions, key=lambda entry: entry["id"]),
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )
    return _MirrorPlan(links, skipped, hashlib.sha256(fingerprint.encode("utf-8")).hexdigest())


def _error(
    detail: str, *, code: str = "media_inventory_failed", returncode: int = -1,
) -> ProcessingError:
    error = ProcessingError("hyperframes media-use resolve", returncode, detail, code=code)
    error.suggested_action = _RETRY
    return error


def _resolve(root: Path, install: RendererInstall, arguments: list[str]) -> dict:
    """Run only local media resolution with a cache confined to this inventory."""
    if not root.is_dir():
        raise _error("The media inventory project folder is missing.")
    home = paths.validate_output_dir(root / "home")
    try:
        result = process.spawn(
            [str(install.executable), "media-use", "resolve", *arguments,
             "--local-only", "--json", "--project", str(root)],
            timeout_s=120,
            tool="hyperframes",
            env={**hyperframes.renderer_env(), "HYPERFRAMES_MEDIA_HOME": str(home)},
        )
    except ProcessingError as exc:
        exc.suggested_action = _RETRY
        raise
    if result.returncode != 0:
        raise _error(
            "HyperFrames could not resolve the media inventory.", returncode=result.returncode,
        )
    try:
        document = json.loads(result.stdout)
    except (TypeError, ValueError, RecursionError):
        raise _error("HyperFrames did not return valid media inventory JSON.") from None
    if not isinstance(document, dict) or document.get("ok") is not True:
        raise _error("HyperFrames did not report a successful media inventory resolution.")
    return document


def _mapped_items(document: dict, expected: dict[str, dict]) -> list[dict]:
    """Require a complete, unambiguous mapping from adopted paths to catalog ids."""
    assets = document.get("assets")
    if not isinstance(assets, list):
        raise _error("HyperFrames did not return an adopted asset list.")
    items, seen_paths, seen_ids = [], set(), set()
    for asset in assets:
        if not isinstance(asset, dict):
            raise _error("HyperFrames returned an invalid adopted asset.")
        path, hf_id = asset.get("path"), asset.get("id")
        item = expected.get(path) if isinstance(path, str) else None
        if (
            item is None or path in seen_paths
            or not isinstance(hf_id, str) or not hf_id or hf_id in seen_ids
            or asset.get("type") != item["kind"]
        ):
            raise _error(
                "HyperFrames adopted assets do not match the catalog media paths.",
                code="media_inventory_mismatch",
            )
        seen_paths.add(path)
        seen_ids.add(hf_id)
        items.append({
            "id": item["id"], "hf_id": hf_id, "type": asset["type"], "path": path,
            "description": asset.get("description", ""),
        })
    if seen_paths != expected.keys():
        raise _error(
            "HyperFrames did not adopt every available catalog item.",
            code="media_inventory_mismatch",
        )
    return items


def _publish(staging: Path, root: Path) -> None:
    """Swap a completed inventory into place, restoring the old one if rename fails."""
    previous = None
    if root.exists():
        previous = Path(tempfile.mkdtemp(dir=root.parent, prefix=".media-inventory-old-"))
        try:
            root.replace(previous)
        except BaseException:
            previous.rmdir()
            raise
    try:
        staging.replace(root)
    except BaseException:
        if previous is not None:
            previous.replace(root)
        raise
    if previous is not None:
        shutil.rmtree(previous)


def sync_inventory(*, install: RendererInstall | None = None) -> dict:
    """Adopt existing catalog files in staging and publish a complete private inventory."""
    root = inventory_root()
    with _inventory_lock(root):
        load_catalog()  # Validate before any renderer download.
        install = install if install is not None else hyperframes.ensure_installed()
        plan = _mirror_plan(load_catalog())  # Reload: the catalog may change while installing.
        return _sync_inventory(root, plan, install)


def _sync_inventory(root: Path, plan: _MirrorPlan, install: RendererInstall) -> dict:
    """Publish a prepared mirror plan while the caller holds the inventory lock."""
    if root.exists() or root.is_symlink():
        paths.validate_output_dir(root)
    staging = Path(tempfile.mkdtemp(dir=root.parent, prefix=".media-inventory-"))
    try:
        for kind in ("videos", "images"):
            (staging / "assets" / kind).mkdir(parents=True)
        expected = {}
        for link in plan.links:
            (staging / link.relative).symlink_to(link.source)
            expected[link.relative.as_posix()] = link.item
        adopted = _resolve(staging, install, ["--adopt"])
        document = {
            "schema": INVENTORY_SCHEMA,
            "renderer": {"name": "hyperframes", "version": install.version},
            "synced_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "plan_sha256": plan.sha256,
            "items": _mapped_items(adopted, expected),
            "skipped": plan.skipped,
        }
        paths.atomic_write_text(
            staging / "inventory.json",
            json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        )
        _publish(staging, root)
        return document
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def search_inventory(
    intent: str, *, kind: str, install: RendererInstall | None = None,
) -> list[dict]:
    """Return the renderer's ordered local candidates mapped to creator catalog ids."""
    if kind not in {"video", "image"}:
        raise ValidationError("kind", "choose video or image")
    root = inventory_root()
    with _inventory_lock(root):
        load_catalog()  # Validate before any renderer download.
        install = install if install is not None else hyperframes.ensure_installed()
        plan = _mirror_plan(load_catalog())  # Reload: the catalog may change while installing.
        return _search_inventory(root, plan, install, intent, kind)


def _search_inventory(
    root: Path, plan: _MirrorPlan, install: RendererInstall, intent: str, kind: str,
) -> list[dict]:
    """Sync a stale plan and search while the caller holds the inventory lock."""
    try:
        document = json.loads((root / "inventory.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeError, ValueError, RecursionError):
        document = None
    if (
        not isinstance(document, dict) or document.get("schema") != INVENTORY_SCHEMA
        or document.get("plan_sha256") != plan.sha256
    ):
        document = _sync_inventory(root, plan, install)
    candidates = _resolve(root, install, ["--type", kind, "--intent", intent, "--candidates"])
    entries = candidates.get("candidates")
    if not isinstance(entries, list):
        raise _error("HyperFrames did not return a media candidate list.")
    by_path = {item["path"]: item for item in document["items"]}
    matches = []
    for candidate in entries:
        if not isinstance(candidate, dict):
            raise _error("HyperFrames returned an invalid media candidate.")
        if candidate.get("scope") != "project":
            continue
        path = candidate.get("path")
        item = by_path.get(path) if isinstance(path, str) else None
        if item is None:
            continue
        matches.append({
            "id": item["id"], "hf_id": candidate["id"],
            "description": candidate.get("description"), "score": candidate.get("score"),
            "duration_s": candidate.get("duration"), "width": candidate.get("width"),
            "height": candidate.get("height"),
        })
    return matches
