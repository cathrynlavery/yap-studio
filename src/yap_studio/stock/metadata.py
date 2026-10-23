"""Validation of the optional catalog stock metadata."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from urllib.parse import urlsplit

FIELDS = {
    "provider",
    "clip_id",
    "page_url",
    "author",
    "author_url",
    "license",
    "license_url",
    "file_url",
    "downloaded_at",
}


Clock = datetime | Callable[[], datetime] | None


def timestamp(now: Clock = None) -> str:
    """UTC seconds for catalog metadata and project approvals."""
    instant = now() if callable(now) else now
    return (instant or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_stock(stock: object) -> dict:
    if not isinstance(stock, dict) or set(stock) != FIELDS:
        raise ValueError("invalid stock fields")
    if any(not isinstance(value, str) or not value.strip() for value in stock.values()):
        raise ValueError("invalid stock strings")
    if re.fullmatch(r"[a-z0-9_]{1,40}", stock["provider"]) is None:
        raise ValueError("invalid stock provider")
    for field in ("page_url", "author_url", "license_url", "file_url"):
        parsed = urlsplit(stock[field])
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or any(ord(c) < 33 for c in stock[field])
        ):
            raise ValueError("invalid stock URL")
        _ = parsed.port
    from yap_studio.media.catalog import _timestamp

    _timestamp(stock["downloaded_at"])
    return stock
