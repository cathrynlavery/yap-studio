"""Pexels CLI adapter. Credentials remain entirely in the creator's CLI."""

from __future__ import annotations

import json
import math
from urllib.parse import urlsplit

from yap_studio import tools
from yap_studio.errors import DependencyError, ProcessingError, YapError
from yap_studio.stock.fetch import allowed_url
from yap_studio.stock.provider import StockClip, StockFile, StockUnavailable, redact

INSTALL_LINE = "npx -y @mvanhorn/printing-press-library install pexels --cli-only"
KEY_LINE = (
    "Create a free Pexels account, request your API key at https://www.pexels.com/api/ "
    "(it arrives at once), and save it: pexels-pp-cli auth set-token <your-key>"
)
LICENSE = "Pexels License"
LICENSE_URL = "https://www.pexels.com/license/"
FLAGS = ["--json", "--no-input", "--no-color", "--data-source", "live", "--timeout", "30s"]


def _number(value) -> float:
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        raise ValueError("invalid number")
    return float(value)


def _text(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing text")
    return value


def _id(value) -> str:
    if type(value) is not int or value < 1:
        raise ValueError("invalid id")
    return str(value)


def _author_url(value, page_url: str) -> str:
    if isinstance(value, str):
        try:
            parsed = urlsplit(value)
            if (
                parsed.scheme == "https"
                and parsed.hostname
                and parsed.username is None
                and parsed.password is None
                and not any(ord(c) < 33 for c in value)
            ):
                _ = parsed.port
                return value
        except ValueError:
            pass
    return page_url


def _file(raw) -> StockFile | None:
    try:
        if raw.get("file_type") != "video/mp4":
            return None
        width, height = int(_number(raw["width"])), int(_number(raw["height"]))
        link = _text(raw["link"])
        if not width or not height or not allowed_url(link):
            return None
        fps = _number(raw["fps"]) if raw.get("fps") is not None else None
        return StockFile(
            _id(raw["id"]),
            link,
            width,
            height,
            fps,
            raw.get("quality") if isinstance(raw.get("quality"), str) else None,
        )
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return None


def _clip(raw) -> StockClip | None:
    try:
        pictures = []
        raw_pictures = raw.get("video_pictures", [])
        for picture in raw_pictures if isinstance(raw_pictures, list) else []:
            try:
                pictures.append((_number(picture["nr"]), _text(picture["picture"])))
            except (KeyError, TypeError, ValueError):
                continue
        pictures.sort(key=lambda picture: picture[0])
        preview = pictures[len(pictures) // 2][1] if pictures else raw.get("image")
        files = raw.get("video_files", [])
        return StockClip(
            "pexels",
            _id(raw["id"]),
            _text(raw["url"]),
            _text(raw["user"]["name"]),
            _author_url(raw["user"].get("url"), raw["url"]),
            _number(raw["duration"]),
            int(_number(raw.get("width", 0))),
            int(_number(raw.get("height", 0))),
            preview if isinstance(preview, str) else None,
            tuple(
                file
                for value in (files if isinstance(files, list) else [])
                if (file := _file(value)) is not None
            ),
            LICENSE,
            LICENSE_URL,
        )
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return None


class PexelsProvider:
    name = "pexels"
    label = "Pexels"
    home_url = "https://www.pexels.com"

    def setup_lines(self) -> list[str]:
        return [INSTALL_LINE, KEY_LINE]

    def status(self) -> str:
        if tools.find_tool("pexels-pp-cli") is None:
            return "missing_cli"
        try:
            tools.run_tool("pexels-pp-cli", ["auth", "status", "--json"], timeout_s=10)
            return "ready"
        except ProcessingError as exc:
            return "signed_out" if exc.returncode == 4 else "unknown"
        except (YapError, OSError, ValueError):
            return "unknown"

    def _read(self, args: list[str]):
        try:
            result = tools.run_tool("pexels-pp-cli", [*args, *FLAGS], timeout_s=60)
            payload = json.loads(result.stdout)
            return payload.get("results", payload) if isinstance(payload, dict) else payload
        except DependencyError:
            raise StockUnavailable("missing_cli", setup=self.setup_lines()) from None
        except ProcessingError as exc:
            stderr = redact(exc.full_stderr or "")
            lower = stderr.lower()
            reason = (
                "signed_out"
                if exc.returncode == 4
                else "rate_limited"
                if "429" in lower or "rate limit" in lower
                else "offline"
                if any(
                    term in lower
                    for term in (
                        "no such host",
                        "dial tcp",
                        "connection refused",
                        "network is unreachable",
                        "timeout",
                        "i/o timeout",
                        "tls handshake",
                    )
                )
                else "search_failed"
            )
            raise StockUnavailable(
                reason, detail=stderr, setup=(self.setup_lines() if reason == "signed_out" else [])
            ) from None
        except (YapError, OSError, ValueError, TypeError):
            raise StockUnavailable("search_failed") from None

    def search(
        self, query: str, *, orientation: str | None, size: str | None, per_page: int
    ) -> list[StockClip]:
        args = ["videos", "search", f"--query={query}", "--per-page", str(per_page), "--page", "1"]
        if orientation:
            args.extend(["--orientation", orientation])
        if size:
            args.extend(["--size", size])
        data = self._read(args)
        values = data.get("videos", []) if isinstance(data, dict) else data
        if not isinstance(values, list):
            raise StockUnavailable("search_failed")
        return [clip for value in values if (clip := _clip(value)) is not None]

    def details(self, clip_id: str) -> StockClip:
        clip = _clip(self._read(["videos", "get", clip_id]))
        if clip is None or clip.clip_id != clip_id:
            raise StockUnavailable("search_failed")
        return clip

    def credit(self, clip: StockClip) -> str:
        return f"Video by {clip.author} on Pexels"
