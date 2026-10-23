"""Require project approval for generated media in B-roll and cards."""

from __future__ import annotations

import json
from pathlib import Path

from yap_studio import cards, receipt
from yap_studio.errors import ValidationError, YapError
from yap_studio.media.catalog import _timestamp, load_catalog

PICTURES_SCHEMA = "yap-studio/generated-pictures@1"
IMAGERY_SCHEMA = "yap-studio/imagery-answer@1"
IMAGERY_QUESTION = {
    "id": "imagery",
    "header": "Imagery",
    "prompt": "Do you want me to create imagery for this video?",
    "note": "Each picture is paid on your WaveSpeed account. "
            "You see each price before anything is made.",
    "kind": "choice",
    "choices": [
        {"value": "yes", "label": "Yes", "description":
         "I draft a picture for lines your media does not cover and show each price first.",
         "recommended": False},
        {"value": "no", "label": "No", "description":
         "No generated pictures in this video.", "recommended": False},
    ],
}


def imagery_answer(project_dir) -> str | None:
    """Read a per-video answer strictly; missing means not asked."""
    if project_dir is None:
        return None
    source = Path(project_dir) / "pictures/imagery.json"
    try:
        payload = source.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValidationError("pictures/imagery.json", "cannot read the imagery answer") from exc
    try:
        data = json.loads(payload)
        if (
            not isinstance(data, dict)
            or set(data) != {"schema", "question", "answer", "answered_at"}
            or data["schema"] != IMAGERY_SCHEMA
            or not isinstance(data["question"], str)
            or not data["question"]
            or data["answer"] not in ("yes", "no")
        ):
            raise ValueError("invalid imagery answer")
        _timestamp(data["answered_at"])
        return data["answer"]
    except (ValueError, TypeError) as exc:
        raise ValidationError("pictures/imagery.json", "cannot read the imagery answer") from exc


def generated_refusal(style, project_dir) -> YapError | None:
    """Use the same permission message and action at every entry point."""
    if style.get("broll", {}).get("generated_pictures") != "ask":
        return YapError(
            "Generated pictures are off in this style.",
            error_type="validation_error", code="pictures_off",
            suggested_action="Turn on generated pictures in the style first.",
        )
    answer = imagery_answer(project_dir)
    if answer is None:
        return YapError(
            "The creator has not said yes to generated pictures for this video.",
            error_type="validation_error", code="pictures_not_asked",
            suggested_action=f'Ask "{IMAGERY_QUESTION["prompt"]}" and record it with '
                             "yap-studio pictures answer DIR yes or no.",
        )
    if answer == "no":
        return YapError(
            "The creator said no to generated pictures for this video.",
            error_type="validation_error", code="pictures_declined",
            suggested_action="Carry on with their own media. Ask again only if they bring it up.",
        )
    return None


def generated_allowed(style, project_dir) -> tuple[bool, str | None]:
    """Require both style permission and this video's yes."""
    refusal = generated_refusal(style, project_dir)
    return (False, refusal.message) if refusal else (True, None)


def approved_pictures(project_dir) -> list[dict]:
    """Read the project's approval list strictly."""
    if project_dir is None:
        return []
    source = Path(project_dir) / "pictures/pictures.json"
    if not source.exists():
        return []
    try:
        data = json.loads(source.read_bytes())
        if data.get("schema") != PICTURES_SCHEMA or not isinstance(data.get("pictures"), list):
            raise ValueError("invalid pictures document")
        if any(
            not isinstance(p, dict) or not isinstance(p.get("sha256"), str)
            for p in data["pictures"]
        ):
            raise ValueError("invalid picture entry")
        return data["pictures"]
    except (OSError, ValueError, AttributeError) as exc:
        raise ValidationError("pictures.pictures", "cannot read the approved picture list") from exc


def _allowed(items, style, project_dir):
    if not any("generated" in item for item in items):
        return {item["id"] for item in items}
    hashes = (
        {p["sha256"] for p in approved_pictures(project_dir)}
        if generated_allowed(style, project_dir)[0]
        else set()
    )
    return {item["id"] for item in items if "generated" not in item or item["sha256"] in hashes}


def filter_approved(items, style, project_dir):
    """Filter before suggestion ranking."""
    allowed = _allowed(items, style, project_dir)
    return (
        [item for item in items if item["id"] in allowed],
        [
            {"item": item["id"], "reasons": ["generated picture not approved for this video"]}
            for item in items
            if item["id"] not in allowed
        ],
    )


def validate_requests(requests, items, style, project_dir) -> None:
    """Name the exact B-roll item that needs approval."""
    allowed = _allowed(items, style, project_dir)
    generated = {item["id"] for item in items if "generated" in item}
    for i, request in enumerate(requests.get("broll", []) if isinstance(requests, dict) else []):
        item = request.get("item") if isinstance(request, dict) else None
        if isinstance(item, str) and item in generated and item not in allowed:
            raise ValidationError(
                f"broll[{i}].item", "generated picture not approved for this video"
            )


def _timeline_media(elements):
    for element in elements:
        if element.kind in {"insert", "overlay", "background"}:
            yield element, element.fields["media"]
        elif element.kind == "graphic" and element.fields.get("template") in cards.CARD_TEMPLATES:
            slots = element.fields.get("slots", {})
            for name, rule in cards.load_template(element.fields["template"])["slots"].items():
                if rule["type"] not in {"image", "image_list", "video", "video_list"}:
                    continue
                value = slots.get(name, rule.get("default"))
                values = value if rule["type"].endswith("_list") else [value]
                for media in values if isinstance(values, list) else []:
                    if isinstance(media, str) and media != "self":
                        yield element, media
                    elif (
                        isinstance(media, dict)
                        and isinstance(media.get("path"), str)
                        and isinstance(media.get("sha256"), str)
                    ):
                        yield element, media


def _generated_media(project_dir, elements):
    media_items = list(_timeline_media(elements))
    if not media_items:
        return
    hashes = {item["sha256"] for item in load_catalog()["items"] if "generated" in item}
    root = (project_dir / "pictures/files").resolve()
    for element, media in media_items:
        source = (project_dir / (media if isinstance(media, str) else media["path"])).resolve()
        in_pictures = source.is_relative_to(root)
        digest = None if isinstance(media, str) else media["sha256"]
        if isinstance(media, str) and (in_pictures or hashes) and source.is_file():
            digest = receipt.sha256_file(source)
        if not in_pictures and digest not in hashes:
            continue
        yield element, digest


def generated_element_ids(project_dir, elements) -> list[str]:
    """Find generated pictures in paths, catalog hashes and card slots."""
    return list(dict.fromkeys(element.id for element, _ in _generated_media(project_dir, elements)))


def check_timeline(project_dir, elements, style) -> None:
    """Check generated media hashes and project file paths."""
    from yap_studio.stages.timeline import TimelineRefusal

    approved = None
    for element, digest in _generated_media(project_dir, elements):
        refusal = generated_refusal(style, project_dir)
        if refusal:
            raise TimelineRefusal(
                refusal.message, suggested_action=refusal.suggested_action,
            )
        if approved is None:
            approved = {p["sha256"] for p in approved_pictures(project_dir)}
        if digest not in approved:
            raise TimelineRefusal(
                f"{element.id} uses a generated picture that is not approved for this video.",
                suggested_action="Ask with yap-studio pictures ask, then pictures make.",
            )
