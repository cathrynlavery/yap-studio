"""Validation of generated picture provenance in the media catalog."""

from __future__ import annotations

import math
import re

FIELDS = {
    "provider",
    "model",
    "prompt",
    "seed",
    "size",
    "prediction_id",
    "cost_usd",
    "cost_source",
    "created_at",
}


def validate_generated(value: object) -> dict:
    """Validate exact fields without changing the metadata."""
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError("invalid generated fields")
    for field in ("provider", "model", "prompt", "size", "prediction_id"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError("invalid generated strings")
    if re.fullmatch(r"[a-z0-9_]{1,40}", value["provider"]) is None:
        raise ValueError("invalid generated provider")
    if value["seed"] is not None and type(value["seed"]) is not int:
        raise ValueError("invalid generated seed")
    cost = value["cost_usd"]
    if type(cost) not in {int, float} or not math.isfinite(cost) or cost < 0:
        raise ValueError("invalid generated cost")
    if value["cost_source"] not in ("charged", "estimate"):
        raise ValueError("invalid generated cost source")
    from yap_studio.media.catalog import _timestamp

    _timestamp(value["created_at"])
    return value
