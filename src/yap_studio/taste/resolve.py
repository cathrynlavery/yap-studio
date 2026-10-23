"""Apply approved style preferences without changing the input style."""

from __future__ import annotations

import copy
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from yap_studio.style.loader import ResolvedStyle, resolve_style_data, unbranded
from yap_studio.taste.model import TasteError, TasteSnapshot, validate_style_field
from yap_studio.taste.store import load_taste


@dataclass(frozen=True)
class TasteApplication:
    """A resolved style and deterministic receipt of preference decisions."""

    style: ResolvedStyle
    taste_revision: int
    taste_sha256: str
    applied: tuple[dict[str, Any], ...]
    skipped: tuple[dict[str, str], ...]
    explicit_fields: tuple[str, ...]
    enforced: tuple[dict[str, str], ...] = ()

    def pin(self) -> dict[str, Any]:
        """Return render identities and decisions without preference content."""
        return {
            "revision": self.taste_revision,
            "sha256": self.taste_sha256,
            "style_sha256": self.style.sha256(),
            "applied": [{"id": item["id"], "field": item["field"]} for item in self.applied],
            "skipped": [{"id": item["id"], "reason": item["reason"]} for item in self.skipped],
            "explicit_fields": list(self.explicit_fields),
            **({"enforced": copy.deepcopy(list(self.enforced))} if self.enforced else {}),
        }

    def to_json(self) -> dict[str, Any]:
        """Return the resulting style and its preference receipt as JSON data."""
        return {
            "style": self.style.to_json(),
            "taste_revision": self.taste_revision,
            "taste_sha256": self.taste_sha256,
            "applied": copy.deepcopy(list(self.applied)),
            "skipped": copy.deepcopy(list(self.skipped)),
            "explicit_fields": list(self.explicit_fields),
            **({"enforced": copy.deepcopy(list(self.enforced))} if self.enforced else {}),
        }


def apply_taste(
    style: ResolvedStyle,
    snapshot: TasteSnapshot | None = None,
    *,
    tags: Collection[str] = (),
    explicit: Mapping[str, Any] | None = None,
) -> TasteApplication:
    """Apply applicable style rules, then validated project instructions."""
    if snapshot is None:
        snapshot = load_taste().snapshot()
    if explicit is not None and not isinstance(explicit, Mapping):
        raise TasteError("explicit", "expected a mapping of style fields to values")
    overrides = copy.deepcopy(dict(explicit or {}))
    for field, value in overrides.items():
        validate_style_field(field, value, f"explicit.{field}")
    data = copy.deepcopy(
        style.unbranded_data if style.unbranded_data is not None else style.data
    )
    skipped = []
    enforced = []
    candidates: dict[str, list[dict[str, Any]]] = {}
    tag_set = set(tags)
    for rule in sorted(snapshot.rules, key=lambda rule: rule["id"]):
        machine, condition = rule["machine"], rule["condition"]
        reason = None
        if machine is None:
            reason = "followed_by_agent"
        elif machine["kind"] == "placement":
            if condition is None or (condition["tags"]
                                     and set(condition["tags"]).issubset(tag_set)):
                enforced.append({"id": rule["id"], "check": "faces.clearance"})
                continue
            reason = "condition_not_met" if condition["tags"] else "condition_needs_judgement"
        elif machine["kind"] != "style_field":
            reason = "not_a_style_rule"
        elif machine["field"] in overrides:
            reason = "project_override"
        elif condition is not None and not condition["tags"]:
            reason = "condition_needs_judgement"
        elif condition is not None and not set(condition["tags"]).issubset(tag_set):
            reason = "condition_not_met"
        if reason is not None:
            skipped.append({"id": rule["id"], "reason": reason})
        else:
            validate_style_field(machine["field"], machine["value"], f"taste.{rule['id']}.machine")
            candidates.setdefault(machine["field"], []).append(rule)
    applied = []
    for field, rules in sorted(candidates.items()):
        ordered = sorted(
            rules,
            key=lambda rule: (
                -len(rule["condition"]["tags"]) if rule["condition"] is not None else 0,
                rule["id"],
            ),
        )
        winner = ordered[0]
        value = copy.deepcopy(winner["machine"]["value"])
        section, key = field.split(".")
        data[section][key] = value
        applied.append({"id": winner["id"], "field": field, "value": value})
        skipped.extend({"id": rule["id"], "reason": "conflict"} for rule in ordered[1:])
    for field, value in overrides.items():
        section, key = field.split(".")
        data[section][key] = value
    resolved = ResolvedStyle(
        style.name, style.source, resolve_style_data(data),
        unbranded_data=unbranded(data),
        caption_placement_explicit=(
            style.caption_placement_explicit or "captions.placement" in overrides
            or any(item["field"] == "captions.placement" for item in applied)
        ),
    )
    return TasteApplication(
        resolved,
        snapshot.revision,
        snapshot.sha256,
        tuple(sorted(applied, key=lambda item: item["id"])),
        tuple(sorted(skipped, key=lambda item: item["id"])),
        tuple(sorted(overrides)),
        tuple(enforced),
    )
