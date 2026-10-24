"""Draft prompts and prices before asking the creator to spend."""

from __future__ import annotations

import html
import json
import re
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from yap_studio import paths
from yap_studio.errors import ValidationError
from yap_studio.media.select import (
    INSERT_MAX_S,
    INSERT_MIN_S,
    REQUEST_SCHEMA,
    parse_request,
    rank_candidates,
)
from yap_studio.pictures import guard, wavespeed
from yap_studio.planner.anchors import resolve_anchor
from yap_studio.planner.broll import _matching_location
from yap_studio.project.lock import project_lock
from yap_studio.project.model import CutWords, ProjectFileError, parse_anchor
from yap_studio.stock.metadata import timestamp
from yap_studio.stock.sheet import _CSS, _object, _time

SCHEMA = "yap-studio/picture-ask-sheet@1"
REQUESTS_SCHEMA = "yap-studio/picture-requests@1"


def money(value) -> str:
    text = format(Decimal(str(value)).quantize(Decimal("0.0001")), "f").rstrip("0")
    whole, fraction = text.split(".")
    return "$" + whole + "." + fraction.ljust(2, "0")


def total(values) -> float:
    return float(sum((Decimal(str(v)) for v in values), Decimal(0)).quantize(Decimal("0.0001")))


def require_allowed(style, project_dir) -> None:
    refusal = guard.generated_refusal(style, project_dir)
    if refusal:
        raise refusal


def unavailable(reason: str, *, schema=SCHEMA) -> dict:
    return {
        "schema": schema,
        "status": "unavailable",
        "reason": reason,
        "message": wavespeed.unavailable_message(reason),
        "setup": wavespeed.setup_lines(reason),
    }


def _lines(requests: dict, cut_words: CutWords) -> list[dict]:
    data = _object(requests, {"schema", "lines"}, "requests")
    if data.get("schema") != REQUESTS_SCHEMA:
        raise ValidationError("schema", f"expected {REQUESTS_SCHEMA}")
    lines = data.get("lines")
    if not isinstance(lines, list) or not 1 <= len(lines) <= 8:
        raise ValidationError("lines", "expected 1 to 8 lines")
    if cut_words.time_base != "output":
        raise ValidationError("cut_words.time_base", "expected output time")
    result, seen = [], set()
    for i, line in enumerate(lines):
        loc = f"lines[{i}]"
        line = _object(line, {"id", "start", "end", "prompt", "not_fitting"}, loc)
        identifier = line.get("id")
        if (
            not isinstance(identifier, str)
            or not identifier.startswith("pic-")
            or not identifier[4:].strip()
            or identifier in seen
        ):
            raise ValidationError(f"{loc}.id", "expected a unique id starting pic-")
        seen.add(identifier)
        prompt = line.get("prompt")
        if (
            not isinstance(prompt, str)
            or not prompt.isprintable()
            or not 3 <= len(prompt.strip()) <= 400
        ):
            raise ValidationError(
                f"{loc}.prompt", "expected 3 to 400 printable characters, no newlines"
            )
        rejected = line.get("not_fitting", [])
        if not isinstance(rejected, list) or any(
            not isinstance(item, str) or re.fullmatch(r"m[1-9][0-9]*", item) is None
            for item in rejected
        ):
            raise ValidationError(f"{loc}.not_fitting", "expected a list of catalog ids")
        times = []
        for edge in ("start", "end"):
            anchor = line.get(edge)
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
        text = " ".join(w.text for w in cut_words.words if w.end > start and w.start < end)
        result.append(
            {
                "number": i + 1,
                "id": identifier,
                "text": text,
                "word_ids": [w.id for w in cut_words.words if w.end > start and w.start < end],
                "prompt": prompt.strip(),
                "start": dict(line["start"]),
                "end": dict(line["end"]),
                "start_s": start,
                "end_s": end,
                "window_s": window,
                "not_fitting": rejected,
            }
        )
    return result


def render_html(data: dict, project: str) -> str:
    def esc(value):
        return html.escape(str(value), quote=True)

    title = "Pictures to generate for " + project
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{esc(title)}</title><style>{_CSS}</style><main><h1>{esc(title)}</h1>",
        "<p>Nothing is made until you say yes. Each picture is paid on your WaveSpeed "
        "account. Tell your agent the numbers you want, for example 1 and 3.</p>",
        f'<p class="provider">Made with {esc(data["provider"]["label"])} '
        f"({esc(data['model'])}, {esc(data['size'].replace('*', ' x '))})</p>",
    ]
    for line in data["lines"]:
        parts.extend(
            [
                f"<section><h2>{esc(line['number'])}. “{esc(line['text'])}”</h2>",
                f"<p>{esc(_time(line['start_s']))} to {esc(_time(line['end_s']))} "
                f"({esc(format(line['window_s'], '.1f'))} s)</p>",
                f"<p>Prompt: {esc(line['prompt'])}</p>",
                f"<p>About {esc(money(line['estimate_usd']))}</p></section>",
            ]
        )
    parts.append(f"<p>Total for every line: about {esc(money(data['total_usd']))}</p>")
    parts.append(
        "<p>The ceiling is checked against WaveSpeed's quotes before each picture. "
        "WaveSpeed has no price cap on a request. The run stops if a charge comes in "
        "above its quote.</p>"
    )
    if data["skipped"]:
        parts.append("<section><h2>Skipped lines</h2>")
        for entry in data["skipped"]:
            parts.append(f"<p>{esc(entry['id'])}: {esc(entry['reason'])}</p>")
        parts.append("</section>")
    parts.append(
        "<footer>WaveSpeed's terms say you own the pictures you generate and must check "
        "them before you publish. Never ask for a real person's likeness.</footer>"
        "</main></html>"
    )
    return "\n".join(parts)


def ask_sheet(
    requests, cut_words, catalog_items, style, *, project_dir, cut_words_path=None, now=None
) -> dict:
    """Price all uncovered lines before publishing an immutable ask sheet."""
    project_dir = Path(project_dir)
    size = wavespeed.picture_size(project_dir)
    require_allowed(style, project_dir)
    state = wavespeed.status()
    if state != "ready":
        return unavailable(state)
    lines = _lines(requests, cut_words)
    own = [item for item in catalog_items if "stock" not in item and "generated" not in item]
    by_id = {item["id"]: item for item in own}
    selected, skipped = [], []
    try:
        for line in lines:
            request = parse_request(
                {"schema": REQUEST_SCHEMA, "words": line["text"], "length_s": line["window_s"]}
            )
            request = replace(request, exact_length_s=Fraction(str(line["window_s"])))
            ranked = rank_candidates(own, request, style_broll=style.get("broll", {}))
            remaining = [
                item["item"]
                for item in ranked
                if item["item"] not in line["not_fitting"]
                and _matching_location(by_id[item["item"]])[0] is not None
            ]
            if remaining:
                skipped.append(
                    {
                        "id": line["id"],
                        "reason": "your own media may fit: "
                        + ", ".join(remaining)
                        + "; look at them, or list them in not_fitting",
                    }
                )
                continue
            line.pop("not_fitting")
            line["estimate_usd"] = wavespeed.estimate(
                line["prompt"], **({"size": size} if size != wavespeed.SIZE else {}))
            selected.append(line)
    except wavespeed.PicturesUnavailable as exc:
        return unavailable(exc.reason)
    with paths.operation(), project_lock(project_dir):
        sheets = project_dir / "pictures/sheets"
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
        out = sheets / name
        paths.validate_output_dir(out)
        root = project_dir.resolve()
        source = root / "cut/cut-words.json" if cut_words_path is None else Path(cut_words_path)
        source = source.resolve()
        if source.is_relative_to(root):
            source = source.relative_to(root)
        data = {
            "schema": SCHEMA,
            "status": "ok",
            "sheet": name,
            "cut_words": str(source),
            "created_at": timestamp(now),
            "provider": {
                "name": wavespeed.PROVIDER,
                "label": wavespeed.LABEL,
                "home_url": wavespeed.HOME_URL,
            },
            "model": wavespeed.MODEL_ID,
            "size": size,
            "lines": selected,
            "skipped": skipped,
            "currency": "USD",
            "total_usd": total(line["estimate_usd"] for line in selected),
            "ask": [
                {
                    "line": line["id"],
                    "question": (
                        f'Generate a picture for "{line["text"]}" '
                        f"({_time(line['start_s'])} to {_time(line['end_s'])})? "
                        f"About {money(line['estimate_usd'])}."
                    ),
                    "options": [
                        {"code": str(line["number"]), "label": "Yes: " + line["prompt"]},
                        {"code": None, "label": "No, keep my own footage"},
                    ],
                }
                for line in selected
            ],
        }
        paths.atomic_write_text(out / "ask-sheet.json", json.dumps(data, indent=2) + "\n")
        paths.atomic_write_text(out / "ask-sheet.html", render_html(data, project_dir.name))
    return data
