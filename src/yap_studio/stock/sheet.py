"""Validate footage gaps and publish one static, numbered approval sheet."""

from __future__ import annotations

import html
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlsplit

from yap_studio import paths
from yap_studio.errors import ValidationError, YapError
from yap_studio.media.select import (
    INSERT_MAX_S,
    INSERT_MIN_S,
    REQUEST_SCHEMA,
    parse_request,
    rank_candidates,
)
from yap_studio.planner.anchors import resolve_anchor
from yap_studio.planner.broll import _matching_location
from yap_studio.project.lock import project_lock
from yap_studio.project.model import CutWords, ProjectFileError, parse_anchor
from yap_studio.stock.fetch import PREVIEW_MAX_BYTES, PREVIEW_TIMEOUT_S
from yap_studio.stock.metadata import Clock, timestamp
from yap_studio.stock.provider import (
    StockClip,
    StockProvider,
    StockUnavailable,
    best_file,
    crop_width,
    in_point,
    tier,
)

SCHEMA = "yap-studio/stock-pick-sheet@1"
REQUESTS_SCHEMA = "yap-studio/stock-requests@1"


def _object(value, fields: set[str], location: str) -> dict:
    if not isinstance(value, dict):
        raise ValidationError(location, "expected an object")
    for key in value:
        if key not in fields:
            raise ValidationError(f"{location}.{key}", "unknown field")
    return value


def _phrases(requests: dict, cut_words: CutWords) -> list[dict]:
    data = _object(requests, {"schema", "phrases"}, "requests")
    if data.get("schema") != REQUESTS_SCHEMA:
        raise ValidationError("schema", f"expected {REQUESTS_SCHEMA}")
    phrases = data.get("phrases")
    if not isinstance(phrases, list) or not 1 <= len(phrases) <= 8:
        raise ValidationError("phrases", "expected 1 to 8 phrases")
    if cut_words.time_base != "output":
        raise ValidationError("cut_words.time_base", "expected output time")
    result, seen = [], set()
    for i, phrase in enumerate(phrases):
        loc = f"phrases[{i}]"
        phrase = _object(phrase, {"id", "start", "end", "query", "not_fitting"}, loc)
        identifier = phrase.get("id")
        if (
            not isinstance(identifier, str)
            or not identifier.startswith("stock-")
            or not identifier[6:].strip()
            or identifier in seen
        ):
            raise ValidationError(f"{loc}.id", "expected a unique id starting stock-")
        seen.add(identifier)
        query = phrase.get("query")
        if (
            not isinstance(query, str)
            or not query.isprintable()
            or not 2 <= len(query.strip()) <= 80
        ):
            raise ValidationError(
                f"{loc}.query", "expected 2 to 80 printable characters, no newlines"
            )
        rejected = phrase.get("not_fitting", [])
        if not isinstance(rejected, list) or any(
            not isinstance(item, str) or re.fullmatch(r"m[1-9][0-9]*", item) is None
            for item in rejected
        ):
            raise ValidationError(f"{loc}.not_fitting", "expected a list of catalog ids")
        times = []
        for edge in ("start", "end"):
            anchor = phrase.get(edge)
            if not isinstance(anchor, dict) or "word" not in anchor:
                raise ValidationError(f"{loc}.{edge}", "expected a word anchor")
            try:
                parsed = parse_anchor(anchor, edge=edge)
            except ProjectFileError as exc:
                field = exc.location.replace("anchor", f"{loc}.{edge}", 1)
                raise ValidationError(field, exc.detail) from None
            times.append(
                resolve_anchor(parsed, cut_words, start=edge == "start", location=f"{loc}.{edge}")
            )
        start, end = times
        window = round(end - start, 9)
        if not INSERT_MIN_S <= window <= INSERT_MAX_S:
            raise ValidationError(f"{loc}.end", "choose anchor words that span 1 to 4 seconds")
        text = " ".join(
            word.text for word in cut_words.words if word.end > start and word.start < end
        )
        result.append(
            {
                "number": i + 1,
                "id": identifier,
                "text": text,
                "query": query.strip(),
                "start": dict(phrase["start"]),
                "end": dict(phrase["end"]),
                "start_s": start,
                "end_s": end,
                "window_s": window,
                "not_fitting": rejected,
            }
        )
    return result


def _unavailable(error: StockUnavailable) -> dict:
    return {
        "schema": SCHEMA,
        "status": "unavailable",
        "reason": error.reason,
        "message": error.message,
        "setup": error.setup,
    }


def _time(value: float) -> str:
    tenths = round(value * 10)
    minutes, remainder = divmod(tenths, 600)
    return f"{minutes}:{remainder / 10:04.1f}"


def option_label(candidate: dict) -> str:
    return (
        f"{candidate['code']}: {candidate['duration_s']:g} s, "
        f"{candidate['width']}x{candidate['height']} {candidate['orientation']}, "
        f"by {candidate['author']}"
    )


def _candidate(clip: StockClip, phrase: dict, code: str, provider: StockProvider) -> dict:
    file = best_file(clip)
    return {
        "code": code,
        "provider": clip.provider,
        "clip_id": clip.clip_id,
        "page_url": clip.page_url,
        "author": clip.author,
        "author_url": clip.author_url,
        "duration_s": clip.duration_s,
        "width": clip.width,
        "height": clip.height,
        "orientation": (
            "portrait"
            if clip.height > clip.width
            else "landscape"
            if clip.width > clip.height
            else "square"
        ),
        "file": {
            "file_id": file.file_id,
            "url": file.url,
            "width": file.width,
            "height": file.height,
            "fps": file.fps,
        },
        "crop_width": crop_width(file.width, file.height),
        "tier": tier(clip),
        "in_s": in_point(clip.duration_s, phrase["window_s"]),
        "license": clip.license,
        "license_url": clip.license_url,
        "credit": provider.credit(clip),
        "preview_url": clip.preview_url,
        "thumb": None,
    }


def _link(url: str, label: str) -> str:
    try:
        parsed = urlsplit(url)
        safe = (
            parsed.scheme == "https"
            and parsed.hostname is not None
            and (parsed.hostname == "pexels.com" or parsed.hostname.endswith(".pexels.com"))
            and parsed.username is None
            and parsed.password is None
            and parsed.port in {None, 443}
            and not any(ord(c) < 33 for c in url)
        )
    except (ValueError, TypeError):
        safe = False
    escaped = html.escape(str(label))
    return f'<a href="{html.escape(url, quote=True)}">{escaped}</a>' if safe else escaped


def render_html(data: dict, project: str) -> str:
    esc = html.escape
    parts = [
        f'<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Stock clips for {esc(project)}</title><style>" + _CSS + "</style><main>",
        f"<h1>Stock clips for {esc(project)}</h1>",
        "<p>Nothing is downloaded until you say yes. Tell your agent the codes you want, "
        "for example 1a and 3b.</p>",
        '<p class="provider"><a href="https://www.pexels.com">Videos provided by Pexels</a></p>',
    ]
    for phrase in data["phrases"]:
        parts.extend(
            [
                f"<section><h2>{phrase['number']}. “{esc(phrase['text'])}”</h2>",
                f"<p>{_time(phrase['start_s'])} to {_time(phrase['end_s'])} "
                f"({phrase['window_s']:.1f} s)</p>",
                f'<p>Searched for: {esc(phrase["query"])}</p><div class="cards">',
            ]
        )
        for c in phrase["candidates"]:
            parts.append(f"<article><h3>{esc(c['code'])}</h3>")
            if c["thumb"]:
                parts.append(
                    '<div class="preview">'
                    f'<div class="media" style="aspect-ratio: {c["width"]}/{c["height"]}; '
                    f'width: min(100%, calc(var(--preview-h) * {c["width"] / c["height"]:.4f}))">'
                    f'<img src="{esc(c["thumb"], quote=True)}" '
                    f'alt="Stock clip {esc(c["code"])} by {esc(c["author"], quote=True)}">'
                )
                if c["orientation"] != "portrait":
                    percent = min(100, c["height"] * 9 / 16 / c["width"] * 100)
                    parts.append(f'<span class="crop-frame" style="width:{percent:.4f}%"></span>')
                parts.append("</div></div>")
                if c["orientation"] != "portrait":
                    parts.append("<p>Cropped to the white frame</p>")
            else:
                parts.append('<div class="preview no-preview">No preview</div>')
            parts.extend(
                [
                    f"<p>{c['duration_s']:g} s, {c['width']} x {c['height']}, "
                    f"{esc(c['orientation'])}</p>",
                    f"<p>Video by {_link(c['author_url'], c['author'])} on Pexels</p>",
                    f"<p>{_link(c['license_url'], c['license'])}</p>",
                    f"<p>{_link(c['page_url'], 'View on Pexels')}</p></article>",
                ]
            )
        parts.append("</div>")
        if phrase["note"]:
            parts.append(f"<p>{esc(phrase['note'])}</p>")
        parts.append("</section>")
    if data["skipped"]:
        parts.append("<section><h2>Skipped phrases</h2>")
        for entry in data["skipped"]:
            parts.append(f"<p>{esc(entry['id'])}: {esc(entry['reason'])}</p>")
        parts.append("</section>")
    parts.append(
        "<footer>Free to use under the Pexels License. Credit is optional; "
        "Yap Studio keeps a credits list for each video.</footer></main></html>"
    )
    return "\n".join(parts)


_CSS = """
:root{--preview-h:320px;
font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
"Helvetica Neue",Arial,sans-serif;color:#161616;background:#f2f2f0;line-height:1.5}
*{box-sizing:border-box}body{margin:0}main{max-width:1100px;margin:auto;padding:32px 20px}
h1{font-size:clamp(26px,5vw,40px);line-height:1.2}
h2{font-size:22px}h3{font-size:30px;margin:0 0 12px}
p{margin:8px 0;color:#4c4c47}a{color:#161616;text-underline-offset:3px}
.provider{padding:16px;background:white;border:1px solid #e2e2de;border-radius:12px;font-weight:650}
section{margin:32px 0}.cards{display:flex;flex-wrap:wrap;gap:16px;margin-top:20px}
article{background:white;border:1px solid #e2e2de;border-radius:16px;padding:18px;
flex:1 1 250px;max-width:340px;min-width:0;overflow:hidden}article p{overflow-wrap:anywhere}
.preview{height:var(--preview-h);display:flex;align-items:center;justify-content:center;
background:#e9e9e6;border-radius:10px}
.media{position:relative;overflow:hidden;border-radius:10px;flex:none;max-width:100%}
.media img{width:100%;height:100%;display:block;object-fit:cover}
.crop-frame{position:absolute;top:0;bottom:0;left:50%;transform:translateX(-50%);
border:2px solid white;box-shadow:0 0 0 1000px rgba(0,0,0,.55)}
.no-preview{text-align:center}
footer{border-top:1px solid #e2e2de;padding:24px 0;color:#4c4c47;font-size:14px}
@media(max-width:560px){:root{--preview-h:240px}
main{padding:20px 16px}article{max-width:none;flex-basis:100%}}
"""


def search_sheet(
    requests: dict,
    cut_words: CutWords,
    catalog_items: Sequence[Mapping],
    style: Mapping,
    *,
    project_dir: Path,
    provider: StockProvider,
    fetch: Callable[..., None],
    out_dir: Path | None = None,
    per_phrase: int = 3,
    now: Clock = None,
) -> dict:
    """Search only gaps in supplied media, then publish previews for the creator's yes."""
    from yap_studio.video_type import STOCK_ACTION, stock_allowed

    allowed, message = stock_allowed(style, project_dir)
    if not allowed:
        raise YapError(
            message or "Stock clips are off in this style.",
            code="stock_off",
            suggested_action=STOCK_ACTION if message else "Turn on stock clips in the style first.",
        )
    state = provider.status()
    if state != "ready":
        reason = state if state in {"missing_cli", "signed_out"} else "search_failed"
        return _unavailable(StockUnavailable(reason, setup=provider.setup_lines()))
    if type(per_phrase) is not int or not 1 <= per_phrase <= 3:
        raise ValidationError("per_phrase", "expected an integer from 1 to 3")
    phrases = _phrases(requests, cut_words)
    selected, skipped = [], []
    own = [item for item in catalog_items if "stock" not in item and "generated" not in item]
    own_by_id = {item["id"]: item for item in own}
    try:
        for phrase in phrases:
            request = parse_request(
                {"schema": REQUEST_SCHEMA, "words": phrase["text"], "length_s": phrase["window_s"]}
            )
            request = replace(request, exact_length_s=Fraction(str(phrase["window_s"])))
            ranked = rank_candidates(own, request, style_broll=style.get("broll", {}))
            remaining = [
                item["item"]
                for item in ranked
                if item["item"] not in phrase["not_fitting"]
                and _matching_location(own_by_id[item["item"]])[0] is not None
            ]
            if remaining:
                skipped.append(
                    {
                        "id": phrase["id"],
                        "reason": "your own media may fit: "
                        + ", ".join(remaining)
                        + "; look at them, or list them in not_fitting",
                    }
                )
                continue
            usable, seen = [], set()
            for orientation, size in (("portrait", "medium"), (None, "large")):
                for clip in provider.search(
                    phrase["query"], orientation=orientation, size=size, per_page=15
                ):
                    identity = (clip.provider, clip.clip_id)
                    if (
                        identity not in seen
                        and best_file(clip) is not None
                        and clip.duration_s >= phrase["window_s"]
                    ):
                        seen.add(identity)
                        usable.append(clip)
                if len(usable) >= per_phrase:
                    break
            usable.sort(key=lambda clip: tier(clip))
            phrase.pop("not_fitting")
            phrase["candidates"] = [
                _candidate(clip, phrase, f"{phrase['number']}{chr(97 + i)}", provider)
                for i, clip in enumerate(usable[:per_phrase])
            ]
            phrase["note"] = None if usable else "Nothing suitable found; try other search words"
            selected.append(phrase)
    except StockUnavailable as exc:
        return _unavailable(exc)
    project_dir = Path(project_dir)
    with paths.operation(), project_lock(project_dir):
        sheets = project_dir / "stock/sheets"
        numbers = (
            [
                int(p.name[1:])
                for p in sheets.iterdir()
                if p.is_dir() and re.fullmatch(r"s[1-9][0-9]*", p.name)
            ]
            if sheets.exists()
            else []
        )
        name = f"s{max(numbers, default=0) + 1}"
        out = Path(out_dir) if out_dir is not None else sheets / name
        # Allocate a fresh folder: a prior approval sheet is immutable.
        paths._guard_write(out)
        if out.exists():
            raise ValidationError("out_dir", "choose a new folder for this pick sheet")
        paths.validate_output_dir(out)
        for phrase in selected:
            for c in phrase["candidates"]:
                if not c["preview_url"]:
                    continue
                thumb = f"thumbs/{c['code']}.jpg"
                try:
                    fetch(
                        c["preview_url"],
                        out / thumb,
                        max_bytes=PREVIEW_MAX_BYTES,
                        timeout_s=PREVIEW_TIMEOUT_S,
                    )
                    c["thumb"] = thumb
                except (YapError, OSError, ValueError):
                    pass
        data = {
            "schema": SCHEMA,
            "status": "ok",
            "sheet": name,
            "created_at": timestamp(now),
            "provider": {
                "name": provider.name,
                "label": provider.label,
                "home_url": provider.home_url,
            },
            "phrases": selected,
            "skipped": skipped,
            "ask": [
                {
                    "phrase": p["id"],
                    "question": (
                        f'For "{p["text"]}" ({_time(p["start_s"])} to {_time(p["end_s"])}), '
                        "which stock clip?"
                    ),
                    "options": [
                        {"code": c["code"], "label": option_label(c)} for c in p["candidates"]
                    ]
                    + [{"code": None, "label": "None, keep my own footage"}],
                }
                for p in selected
                if p["candidates"]
            ],
        }
        paths.atomic_write_text(
            out / "pick-sheet.json", json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        )
        paths.atomic_write_text(out / "pick-sheet.html", render_html(data, project_dir.name))
    return data
