"""Split placement validation for explicit B-roll choices."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from yap_studio.errors import ValidationError
from yap_studio.media.select import item_placement


def requested_placement(request: dict, item: Mapping[str, Any], style: Mapping) -> str:
    """Use the style unless a request supplies an explicit placement."""
    default = item_placement(item, style)
    if request.get("kind") == "insert" and default in ("overlay", "green_screen"):
        default = "full_frame"
    placement = request.get("placement", "split" if request.get("fit") == "split" else default)
    return placement


def split_media(request: dict, placement: str, media: dict, location: str) -> bool:
    """Apply split fit after refusing conflicting source crops and fits."""
    if placement != "split":
        return False
    if "crop" in request:
        raise ValidationError(f"{location}.crop", "split does not accept crop; use fit cover")
    if request.get("fit", "split") != "split":
        raise ValidationError(f"{location}.fit", "split placement requires fit split")
    media["fit"] = "split"
    return True
