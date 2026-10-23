"""Platform cache locations and test-local cache creation."""

from __future__ import annotations

from pathlib import Path

import pytest

from yap_studio.cache import cache_dir, cache_root


@pytest.fixture(autouse=True)
def clean_cache_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep default roots and directory creation inside the test directory."""
    monkeypatch.delenv("YAP_STUDIO_CACHE", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))


def test_override_takes_precedence_on_every_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "override"
    monkeypatch.setenv("YAP_STUDIO_CACHE", str(override))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr("yap_studio.cache.sys.platform", "darwin")
    assert cache_root() == override
    assert not override.exists()


def test_relative_override_is_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("YAP_STUDIO_CACHE", "relative-cache")
    assert cache_root() == tmp_path / "relative-cache"
    assert cache_root().is_absolute()


def test_override_expands_user(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("YAP_STUDIO_CACHE", "~/custom-cache")
    monkeypatch.setattr("os.path.expanduser", lambda value: value.replace("~", str(tmp_path), 1))
    assert cache_root() == tmp_path / "custom-cache"


def test_macos_cache_ignores_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("yap_studio.cache.sys.platform", "darwin")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_root() == tmp_path / "Library" / "Caches" / "yap-studio"


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_other_platforms_use_default_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    monkeypatch.setattr("yap_studio.cache.sys.platform", platform)
    assert cache_root() == tmp_path / ".cache" / "yap-studio"


def test_xdg_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("yap_studio.cache.sys.platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_root() == tmp_path / "xdg" / "yap-studio"


@pytest.mark.parametrize("parts", [(), ("renderer", "v1")])
def test_cache_dir_creates_nested_directory_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parts: tuple[str, ...]
) -> None:
    root = tmp_path / "cache"
    monkeypatch.setenv("YAP_STUDIO_CACHE", str(root))
    directory = cache_dir(*parts)
    assert directory == root.joinpath(*parts)
    assert directory.is_dir()
    assert cache_dir(*parts) == directory


@pytest.mark.parametrize("part", ["../outside", "/outside"])
def test_cache_dir_rejects_parts_outside_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, part: str
) -> None:
    root = tmp_path / "cache"
    monkeypatch.setenv("YAP_STUDIO_CACHE", str(root))
    with pytest.raises(ValueError, match="under the cache root"):
        cache_dir(part)
    assert not root.exists()
