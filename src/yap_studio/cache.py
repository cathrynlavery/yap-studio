"""User cache locations for engine-managed tools and assets."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def cache_root() -> Path:
    """Return the absolute user cache root, honoring YAP_STUDIO_CACHE."""
    override = os.environ.get("YAP_STUDIO_CACHE")
    if override:
        return Path(override).expanduser().absolute()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "yap-studio"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser().absolute() if xdg else Path.home() / ".cache"
    return base / "yap-studio"


def cache_dir(*parts: str) -> Path:
    """Create and return a folder under the user cache root."""
    if any(Path(part).is_absolute() or ".." in Path(part).parts for part in parts):
        raise ValueError("cache parts must stay under the cache root")
    directory = cache_root().joinpath(*parts)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
