"""Require creator approval for stock media in B-roll and cards."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from yap_studio import cards, receipt
from yap_studio.errors import ValidationError
from yap_studio.media.catalog import load_catalog
from yap_studio.project.model import Element
from yap_studio.style.library import config_root
from yap_studio.video_type import STOCK_ACTION, stock_allowed

CLIPS_SCHEMA = "yap-studio/stock-clips@1"


def _timeline_clips(elements: Sequence[Element]) -> Iterator[tuple[Element, dict | str]]:
    for element in elements:
        if element.kind in {"insert", "overlay", "background"}:
            yield element, element.fields["media"]
        elif element.kind == "graphic" and element.fields.get("template") in cards.CARD_TEMPLATES:
            slots = element.fields.get("slots", {})
            for name, rule in cards.load_template(element.fields["template"])["slots"].items():
                if rule["type"] not in {"video", "video_list"}:
                    continue
                value = slots.get(name, rule.get("default"))
                values = value if rule["type"] == "video_list" else [value]
                if not isinstance(values, list):
                    continue
                for clip in values:
                    if isinstance(clip, str) and clip != "self":
                        yield element, clip
                    elif (isinstance(clip, dict) and isinstance(clip.get("path"), str)
                          and isinstance(clip.get("sha256"), str)):
                        yield element, clip


def _stock_clips(project_dir: Path, elements: Sequence[Element]
                 ) -> Iterator[tuple[str, str | None]]:
    """Identify stock by cache location or catalog hash, including clips inside cards."""
    clips = list(_timeline_clips(elements))
    if not clips:
        return
    stock_hashes = {item["sha256"] for item in load_catalog()["items"] if "stock" in item}
    cache = (config_root() / "assets/stock").resolve()
    for element, media in clips:
        source = (project_dir / (media if isinstance(media, str) else media["path"])).resolve()
        in_stock = source.is_relative_to(cache)
        digest = None if isinstance(media, str) else media["sha256"]
        if isinstance(media, str) and (in_stock or stock_hashes) and source.is_file():
            digest = receipt.sha256_file(source)
        if in_stock or digest in stock_hashes:
            yield element.id, digest


def stock_element_ids(project_dir: Path, elements: Sequence[Element]) -> list[str]:
    """Return each element using stock once, in timeline order."""
    return list(dict.fromkeys(identifier for identifier, _ in _stock_clips(project_dir, elements)))


def check_timeline(project_dir: Path, elements: Sequence[Element], style: Mapping) -> None:
    """Require style and project approval for stock media, including card clips."""
    from yap_studio.stages.timeline import TimelineRefusal

    approved = None
    for identifier, digest in _stock_clips(project_dir, elements):
        allowed, message = stock_allowed(style, project_dir)
        if not allowed:
            raise TimelineRefusal(
                message or "Stock clips are off in this project's style.",
                suggested_action=(STOCK_ACTION if message else
                                  "Turn on stock clips in the project's style first."),
            )
        if approved is None:
            approved = {clip["sha256"] for clip in approved_clips(project_dir)}
        if digest not in approved:
            raise TimelineRefusal(
                f"{identifier} uses a stock clip that is not approved for this video.",
                suggested_action="Ask with yap-studio stock search, then stock get.",
            )


def approved_clips(project_dir: Path | None) -> list[dict]:
    if project_dir is None:
        return []
    source = Path(project_dir) / "stock/clips.json"
    if not source.exists():
        return []
    try:
        data = json.loads(source.read_bytes())
        if data.get("schema") != CLIPS_SCHEMA or not isinstance(data.get("clips"), list):
            raise ValueError("invalid clips document")
        if any(
            not isinstance(clip, dict) or not isinstance(clip.get("sha256"), str)
            for clip in data["clips"]
        ):
            raise ValueError("invalid clip entry")
        return data["clips"]
    except (OSError, ValueError, AttributeError) as exc:
        raise ValidationError("stock.clips", "cannot read the approved stock list") from exc


def _allowed(items, style, project_dir):
    hashes = (
        {clip["sha256"] for clip in approved_clips(project_dir)}
        if stock_allowed(style, project_dir)[0]
        else set()
    )
    return {item["id"] for item in items if "stock" not in item or item["sha256"] in hashes}


def filter_approved(items, style, project_dir):
    """Return allowed catalog items and suggestion exclusions before ranking."""
    allowed = _allowed(items, style, project_dir)
    return (
        [item for item in items if item["id"] in allowed],
        [
            {"item": item["id"], "reasons": ["stock clip not approved for this video"]}
            for item in items
            if item["id"] not in allowed
        ],
    )


def validate_requests(requests, items, style, project_dir) -> None:
    """Name the precise B-roll item field when stock approval is missing."""
    allowed = _allowed(items, style, project_dir)
    stock = {item["id"] for item in items if "stock" in item}
    for i, request in enumerate(requests.get("broll", []) if isinstance(requests, dict) else []):
        item = request.get("item") if isinstance(request, dict) else None
        if isinstance(item, str) and item in stock and item not in allowed:
            permitted, message = stock_allowed(style, project_dir)
            reason = (
                "stock clips are off in this style"
                if not permitted
                else f"{item} is a stock clip not approved for this video; "
                "ask with yap-studio stock search first"
            )
            error = ValidationError(f"broll[{i}].item", message or reason)
            if message:
                error.suggested_action = STOCK_ACTION
            raise error
