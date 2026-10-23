"""Checks needed before editing from a workspace."""

from __future__ import annotations

import shutil
from pathlib import Path

MIN_FREE_GB = 10
SYNC_FOLDERS = {
    "mobile": "Library/Mobile Documents",
    "cloud": "Library/CloudStorage",
    "cloud_docs": "com~apple~CloudDocs",
    "db": "Dropbox",
    "od": "OneDrive",
    "gd": "Google Drive",
    "gd_cloud": "GoogleDrive",
    "marker": ".dropbox",
}
SYNC_LABELS = {
    "ic": "iCloud Drive",
    "db": "Dropbox",
    "od": "OneDrive",
    "gd": "Google Drive",
}


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def synced_service(path: Path, home: Path | None = None) -> str | None:
    """Name the sync service containing a folder, if one is known."""
    path = path.resolve()
    home = (home or Path.home()).resolve()
    mobile = home / SYNC_FOLDERS["mobile"]
    cloud = home / SYNC_FOLDERS["cloud"]
    if _inside(path, mobile):
        return SYNC_LABELS["ic"]
    if _inside(path, cloud):
        relative = path.relative_to(cloud)
        if relative.parts:
            provider = relative.parts[0].split("-", 1)[0]
            return {SYNC_FOLDERS["gd_cloud"]: SYNC_LABELS["gd"]}.get(
                provider, provider)
    for key in ("db", "gd"):
        name, service = SYNC_FOLDERS[key], SYNC_LABELS[key]
        if _inside(path, home / name):
            return service
    if any(_inside(path, folder) for folder in home.glob(SYNC_FOLDERS["od"] + "*")):
        return SYNC_LABELS["od"]
    cloud_docs = mobile / SYNC_FOLDERS["cloud_docs"]
    for name in ("Desktop", "Documents"):
        if _inside(path, home / name) and (cloud_docs / name).exists():
            return SYNC_LABELS["ic"]
    if any((folder / SYNC_FOLDERS["marker"]).is_file()
           for folder in (path, *path.parents) if folder != home):
        return SYNC_LABELS["db"]
    return None


def free_disk_gb(path: Path) -> float:
    """Return free space on the folder's volume in binary gigabytes."""
    return shutil.disk_usage(path).free / 1024**3


def speech_model_ready(backend: object | None) -> bool:
    """Report whether the selected speech backend has its model cached."""
    return backend is not None and backend.download_notice() is None


def cutout_reason() -> str | None:
    """Explain why the person cutout helper cannot be built."""
    from yap_studio.matte.vision import unavailable_reason

    return unavailable_reason()
