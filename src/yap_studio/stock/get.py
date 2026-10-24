"""Download only the picks the creator approved and save ready B-roll requests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from yap_studio import ffmpeg, paths, receipt
from yap_studio.errors import ValidationError, YapError
from yap_studio.media.catalog import add_stock_item, load_catalog
from yap_studio.project.lock import project_lock
from yap_studio.stock.fetch import CLIP_MAX_BYTES, CLIP_TIMEOUT_S, allowed_url
from yap_studio.stock.guard import CLIPS_SCHEMA, approved_clips
from yap_studio.stock.metadata import Clock, timestamp
from yap_studio.stock.provider import StockProvider, best_file, fits, in_point, redact
from yap_studio.stock.sheet import SCHEMA
from yap_studio.style.library import config_root


def _load_sheet(project_dir: Path, sheet: str | None) -> dict:
    sheets = project_dir / "stock/sheets"
    if sheet is None:
        numbers = (
            [
                int(p.name[1:])
                for p in sheets.iterdir()
                if p.is_dir() and re.fullmatch(r"s[1-9][0-9]*", p.name)
            ]
            if sheets.exists()
            else []
        )
        if not numbers:
            raise ValidationError("sheet", "no pick sheet; run yap-studio stock search first")
        sheet = f"s{max(numbers)}"
    if re.fullmatch(r"s[1-9][0-9]*", sheet) is None:
        raise ValidationError("sheet", "expected sN")
    source = sheets / sheet / "pick-sheet.json"
    try:
        data = json.loads(paths.validate_input(source).read_bytes())
    except (OSError, ValueError, YapError) as exc:
        raise ValidationError("sheet", f"cannot read {sheet}") from exc
    if (
        not isinstance(data, dict)
        or data.get("schema") != SCHEMA
        or data.get("status") != "ok"
        or data.get("sheet") != sheet
        or not isinstance(data.get("phrases"), list)
    ):
        raise ValidationError("sheet", "invalid pick sheet")
    return data


def _cached(target: Path, provider: str, clip_id: str):
    for item in load_catalog()["items"]:
        stock = item.get("stock", {})
        if stock.get("provider") == provider and stock.get("clip_id") == clip_id:
            try:
                if (
                    target.is_file()
                    and not target.is_symlink()
                    and receipt.sha256_file(target) == item["sha256"]
                ):
                    return item
            except OSError:
                pass
    return None


def _obtain(candidate: dict, phrase: dict, *, provider, fetch, now=None, shape="vertical"):
    name, clip_id = candidate["provider"], candidate["clip_id"]
    if (
        name != provider.name
        or re.fullmatch(r"[a-z0-9_]{1,40}", name) is None
        or re.fullmatch(r"[0-9]+", clip_id) is None
    ):
        raise ValidationError("pick", "invalid provider or clip id")
    selected = candidate["file"]
    key = hashlib.sha256(selected["file_id"].encode()).hexdigest()[:16]
    target = config_root() / "assets/stock" / name / f"{clip_id}-{key}.mp4"
    # Serialize shared-cache writers; recheck the catalog after acquiring the lock.
    with project_lock(target.parent):
        item = _cached(target, name, clip_id)
        if (item is not None and item["stock"]["file_url"] == selected["url"]
                and (item["width"], item["height"]) == (selected["width"], selected["height"])
                and fits(item["width"], item["height"], shape=shape)):
            if item["duration_s"] < phrase["window_s"]:
                raise ValidationError("pick.duration_s", "cached clip is shorter than the phrase")
            return item, target, True, item["stock"]
        clip = provider.details(clip_id)
        file = next(
            (
                file
                for file in clip.files
                if file.file_id == candidate["file"]["file_id"]
                and allowed_url(file.url)
                and fits(file.width, file.height, shape=shape)
            ),
            None,
        ) or best_file(clip, shape=shape)
        if file is None:
            raise ValidationError("pick.file", "no fitting MP4 remains available")
        if clip.duration_s < phrase["window_s"]:
            raise ValidationError("pick.duration_s", "clip is shorter than the phrase")
        key = hashlib.sha256(file.file_id.encode()).hexdigest()[:16]
        target = target.with_name(f"{clip_id}-{key}.mp4")
        paths._guard_write(target)
        # Download to a fresh staging file so a failed refresh cannot replace cached bytes.
        fd, staged_name = tempfile.mkstemp(prefix=".stock-", suffix=".mp4", dir=target.parent)
        os.close(fd)
        staged = Path(staged_name)
        try:
            fetch(file.url, staged, max_bytes=CLIP_MAX_BYTES, timeout_s=CLIP_TIMEOUT_S)
            info = ffmpeg.probe(staged, count_frames=False)
            duration = info.video_duration_s or info.duration_s
            if (
                not info.has_video
                or duration is None
                or not math.isfinite(duration)
                or duration < phrase["window_s"]
            ):
                raise ValidationError(
                    "pick.duration_s", "download must contain video lasting the phrase"
                )
            if not fits(info.width or 0, info.height or 0, shape=shape):
                raise ValidationError("pick.file", "download is too small for the delivery canvas")
            paths._guard_write(target)
            os.replace(staged, target)
        finally:
            staged.unlink(missing_ok=True)
        at = timestamp(now)
        stock = {
            "provider": name,
            "clip_id": clip_id,
            "page_url": clip.page_url,
            "author": clip.author,
            "author_url": clip.author_url,
            "license": clip.license,
            "license_url": clip.license_url,
            "file_url": file.url,
            "downloaded_at": at,
        }
        item = add_stock_item(
            target,
            stock=stock,
            description_text=(phrase["query"] + f" (stock clip from {provider.label})"),
            now=now,
        )
        return item, target, False, stock


def get_picks(
    project_dir: Path,
    codes: Sequence[str],
    *,
    yes: bool,
    sheet: str | None = None,
    provider: StockProvider,
    fetch: Callable[..., None],
    now: Clock = None,
) -> dict:
    """Require explicit consent, validate all choices, then handle picks independently."""
    if not yes:
        raise YapError(
            "Downloading stock clips needs the creator's yes.",
            code="stock_needs_yes",
            suggested_action="Show the pick sheet, ask, then pass --yes.",
        )
    project_dir = Path(project_dir)
    with paths.operation():
        data = _load_sheet(project_dir, sheet)
        by_code = {c["code"]: (c, p) for p in data["phrases"] for c in p["candidates"]}
        if not codes:
            raise ValidationError("pick", "choose at least one code")
        unknown = [code for code in codes if code not in by_code]
        if unknown:
            raise ValidationError("pick", "unknown codes: " + ", ".join(unknown))
        identities = [(by_code[code][0]["provider"], by_code[code][0]["clip_id"]) for code in codes]
        if len(identities) != len(set(identities)):
            raise ValidationError("pick", "the same clip cannot be picked twice")
        chosen_phrases = {}
        for code in codes:
            phrase = by_code[code][1]["id"]
            if phrase in chosen_phrases:
                raise ValidationError(
                    "pick",
                    "pick one clip per phrase: "
                    f"{chosen_phrases[phrase]} and {code} are both for {phrase}",
                )
            chosen_phrases[phrase] = code
        result = {
            "schema": "yap-studio/stock-get@1",
            "sheet": data["sheet"],
            "downloaded": [],
            "cached": [],
            "failed": [],
            "requests": [],
            "credits": [],
        }
        for code in codes:
            candidate, phrase = by_code[code]
            try:
                item, target, cached, stock = _obtain(
                    candidate, phrase, provider=provider, fetch=fetch, now=now,
                    shape=data.get("shape", "vertical"),
                )
                request = {
                    "id": "broll-" + phrase["id"],
                    "item": item["id"],
                    "kind": "insert",
                    "start": phrase["start"],
                    "end": phrase["end"],
                    "fit": "cover",
                    "in_s": in_point(item["duration_s"], phrase["window_s"]),
                    "speed": 1,
                    "audio": "mute",
                }
                credit = f"Video by {stock['author']} on Pexels"
                entry = {
                    "code": code,
                    "sheet": data["sheet"],
                    "phrase": phrase["id"],
                    "text": phrase["text"],
                    "item": item["id"],
                    "sha256": item["sha256"],
                    "path": str(target),
                    **{
                        key: stock[key]
                        for key in (
                            "provider",
                            "clip_id",
                            "page_url",
                            "author",
                            "author_url",
                            "license",
                            "license_url",
                            "downloaded_at",
                        )
                    },
                    "credit": credit,
                    "approved_at": timestamp(now),
                    "cached": cached,
                    "request": request,
                }
                with project_lock(project_dir):
                    entries = [
                        old
                        for old in approved_clips(project_dir)
                        if not (
                            old.get("provider") == stock["provider"]
                            and old.get("clip_id") == stock["clip_id"]
                            and old.get("phrase") == phrase["id"]
                        )
                    ]
                    entries.append(entry)
                    paths.atomic_write_text(
                        project_dir / "stock/clips.json",
                        json.dumps(
                            {"schema": CLIPS_SCHEMA, "clips": entries}, indent=2, ensure_ascii=False
                        )
                        + "\n",
                    )
                result["cached" if cached else "downloaded"].append(entry)
                result["requests"].append(request)
                result["credits"].append(credit + ": " + stock["page_url"])
            except (YapError, OSError, ValueError) as exc:
                result["failed"].append({"code": code, "reason": redact(str(exc))})
    return result
