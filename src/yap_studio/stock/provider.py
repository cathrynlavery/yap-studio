"""Provider contracts and pure vertical-frame fit rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from yap_studio.canvas import canvas
from yap_studio.errors import YapError


@dataclass(frozen=True)
class StockFile:
    file_id: str
    url: str
    width: int
    height: int
    fps: float | None
    quality: str | None


@dataclass(frozen=True)
class StockClip:
    provider: str
    clip_id: str
    page_url: str
    author: str
    author_url: str
    duration_s: float
    width: int
    height: int
    preview_url: str | None
    files: tuple[StockFile, ...]
    license: str
    license_url: str


class StockProvider(Protocol):
    name: str
    label: str
    home_url: str

    def status(self) -> str: ...
    def setup_lines(self) -> list[str]: ...
    def search(
        self, query: str, *, orientation: str | None, size: str | None, per_page: int
    ) -> list[StockClip]: ...
    def details(self, clip_id: str) -> StockClip: ...
    def credit(self, clip: StockClip) -> str: ...


class StockUnavailable(YapError):
    """A stock service failure that lets editing continue with supplied footage."""

    def __init__(
        self,
        reason: str,
        *,
        setup: list[str] | None = None,
        detail: str = "",
        http_status: int | None = None,
    ) -> None:
        if reason not in {
            "missing_cli",
            "signed_out",
            "offline",
            "rate_limited",
            "search_failed",
            "download_failed",
        }:
            raise ValueError("unknown stock failure")
        self.reason = reason
        self.setup = list(setup or [])
        explanation = {
            "missing_cli": "Pexels isn't set up on this Mac",
            "signed_out": "Pexels isn't set up on this Mac",
            "offline": "no internet connection",
            "rate_limited": "Pexels says too many searches this hour",
            "search_failed": "Pexels could not complete the search",
            "download_failed": "Pexels could not send the file"
            + (f" (HTTP {http_status})" if http_status is not None else ""),
        }[reason]
        message = f"Stock clips are unavailable: {explanation}. Carrying on with your own footage."
        if detail:
            message += " " + redact(detail)[-1000:]
        super().__init__(
            message,
            error_type="stock_unavailable",
            code=reason,
            suggested_action="\n".join(self.setup) or None,
        )


def redact(text: str) -> str:
    """Drop complete credential-bearing stderr lines before making a message."""
    return "\n".join(
        line
        for line in text.splitlines()
        if not any(
            term in line.lower() for term in ("authorization", "api key", "api_key", "token")
        )
    )


def crop_width(width: int, height: int, *, shape: str = "vertical") -> float:
    """Width left after a cover crop to the delivery aspect ratio."""
    frame = canvas(shape)
    return min(width, height * frame.width / frame.height)


def _id_order(file: StockFile) -> tuple[int, int | str]:
    return (0, int(file.file_id)) if file.file_id.isdecimal() else (1, file.file_id)


def best_file(clip: StockClip, *, shape: str = "vertical") -> StockFile | None:
    """Choose the smallest fitting MP4, then the nearest 30 fps and lowest id.

    Providers admit only MP4 files into StockClip.files. Recheck dimensions and
    download hosts here so manually constructed providers obey the same fit rule.
    """
    from yap_studio.stock.fetch import allowed_url

    fitting = [
        file
        for file in clip.files
        if file.width > 0
        and file.height > 0
        and allowed_url(file.url)
        and crop_width(file.width, file.height, shape=shape) >= canvas(shape).width
    ]
    return min(
        fitting,
        key=lambda file: (
            file.width * file.height,
            abs(file.fps - 30) if file.fps is not None else float("inf"),
            _id_order(file),
        ),
        default=None,
    )


def tier(clip: StockClip, *, shape: str = "vertical") -> int | None:
    """Prefer sources with the delivery orientation, dropping files too small to cover."""
    preferred = clip.width > clip.height if shape == "horizontal" else clip.height > clip.width
    return (0 if preferred else 1) if best_file(clip, shape=shape) else None


def in_point(duration_s: float, window_s: float) -> float:
    """Skip up to the first second while retaining the entire phrase window."""
    return round(max(0.0, min(1.0, duration_s - window_s)), 3)
