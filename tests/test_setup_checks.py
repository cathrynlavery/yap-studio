"""Setup checks use synthetic folders and backend stubs."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest

from yap_studio import setup_checks


@pytest.mark.parametrize(("relative", "service"), [
    (f"{setup_checks.SYNC_FOLDERS['mobile']}/{setup_checks.SYNC_FOLDERS['cloud_docs']}/Projects",
     setup_checks.SYNC_LABELS["ic"]),
    (f"{setup_checks.SYNC_FOLDERS['cloud']}/{setup_checks.SYNC_FOLDERS['db']}-Personal/Projects",
     setup_checks.SYNC_LABELS["db"]),
    (f"{setup_checks.SYNC_FOLDERS['cloud']}/{setup_checks.SYNC_FOLDERS['od']}-Work/Projects",
     setup_checks.SYNC_LABELS["od"]),
    (f"{setup_checks.SYNC_FOLDERS['cloud']}/{setup_checks.SYNC_FOLDERS['gd_cloud']}-personal/Projects",
     setup_checks.SYNC_LABELS["gd"]),
    (f"{setup_checks.SYNC_FOLDERS['cloud']}/Box-Work/Projects", "Box"),
    (f"{setup_checks.SYNC_FOLDERS['db']}/Projects", setup_checks.SYNC_LABELS["db"]),
    (f"{setup_checks.SYNC_FOLDERS['od']} - Work/Projects",
     setup_checks.SYNC_LABELS["od"]),
    (f"{setup_checks.SYNC_FOLDERS['gd']}/Projects", setup_checks.SYNC_LABELS["gd"]),
])
def test_synced_service_known_folders(
    tmp_path: Path, relative: str, service: str,
) -> None:
    home = tmp_path / "home"
    folder = home / relative
    folder.mkdir(parents=True)
    assert setup_checks.synced_service(folder, home) == service


@pytest.mark.parametrize("name", ["Desktop", "Documents"])
def test_synced_service_icloud_desktop_documents(tmp_path: Path, name: str) -> None:
    home = tmp_path / "home"
    folder = home / name / "Project"
    folder.mkdir(parents=True)
    assert setup_checks.synced_service(folder, home) is None
    (home / setup_checks.SYNC_FOLDERS["mobile"]
     / setup_checks.SYNC_FOLDERS["cloud_docs"] / name).mkdir(parents=True)
    assert setup_checks.synced_service(folder, home) == setup_checks.SYNC_LABELS["ic"]


def test_synced_service_marker_and_local_folder(tmp_path: Path) -> None:
    home = tmp_path / "home"
    local = home / "Yap Studio"
    local.mkdir(parents=True)
    marker = setup_checks.SYNC_FOLDERS["marker"]
    (home / marker).mkdir()
    assert setup_checks.synced_service(local, home) is None
    assert setup_checks.synced_service(home, home) is None
    (home / marker).rmdir()
    (home / marker).touch()
    assert setup_checks.synced_service(local, home) is None
    folder = home / "custom sync" / "Project"
    folder.mkdir(parents=True)
    (folder.parent / marker).touch()
    assert setup_checks.synced_service(folder, home) == setup_checks.SYNC_LABELS["db"]


def test_free_disk_gb_uses_available_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(setup_checks.shutil, "disk_usage", lambda path: Usage(100, 90, 9 * 1024**3))
    assert setup_checks.free_disk_gb(tmp_path) == 9
    assert setup_checks.free_disk_gb(tmp_path) < setup_checks.MIN_FREE_GB
    monkeypatch.setattr(
        setup_checks.shutil, "disk_usage", lambda path: Usage(100, 90, 10 * 1024**3)
    )
    assert setup_checks.free_disk_gb(tmp_path) == setup_checks.MIN_FREE_GB


def test_speech_model_ready() -> None:
    assert setup_checks.speech_model_ready(None) is False
    assert setup_checks.speech_model_ready(SimpleNamespace(download_notice=lambda: None))
    assert not setup_checks.speech_model_ready(
        SimpleNamespace(download_notice=lambda: "Download needed")
    )


def test_cutout_reason_wraps_vision_check(monkeypatch: pytest.MonkeyPatch) -> None:
    from yap_studio.matte import vision

    monkeypatch.setattr(vision, "unavailable_reason", lambda: "Xcode tools missing")
    assert setup_checks.cutout_reason() == "Xcode tools missing"
