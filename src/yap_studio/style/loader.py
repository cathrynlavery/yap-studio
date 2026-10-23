"""Validate and resolve portable styles using the engine's schema defaults."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from yap_studio import paths
from yap_studio.errors import ValidationError

SCHEMA_PATH = Path(__file__).parent / "schema-v1.json"
PRESETS_DIR = Path(__file__).parent / "presets"
SHIPPED_PRESETS = (
    "editorial_serif", "clean_minimal", "warm_editorial", "cobalt", "mono",
    "headline", "electric", "candy", "marker", "one_word", "hf_bold", "hf_editorial", "hf_neutral",
)
RESOLVED_SCHEMA = "yap-studio/resolved-style@1"


class StyleError(ValidationError):
    """Report a style field or file that needs correction."""


def _read(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise StyleError(str(path), f"cannot read JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise StyleError(str(path), "expected a JSON object")
    return data


def read_style_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a style JSON object without resolving its settings."""
    return _read(Path(path).expanduser())


@lru_cache(maxsize=1)
def _schema() -> dict[str, Any]:
    return _read(SCHEMA_PATH)


def _check(value: Any, rule: dict[str, Any], location: str) -> None:
    typ = rule["type"]
    if isinstance(typ, list):
        if value is None and "null" in typ:
            return
        typ = next(kind for kind in typ if kind != "null")
    valid = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "boolean": lambda v: isinstance(v, bool),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    }[typ](value)
    if not valid:
        raise StyleError(location, f"expected {typ}, got {type(value).__name__}")
    if "const" in rule and value != rule["const"]:
        raise StyleError(location, f"must be {rule['const']!r}")
    if "enum" in rule and value not in rule["enum"]:
        raise StyleError(location, f"choose one of {', '.join(map(str, rule['enum']))}")
    if typ in ("integer", "number"):
        if isinstance(value, float) and not math.isfinite(value):
            raise StyleError(location, "expected a finite number")
        minimum, maximum = rule.get("minimum"), rule.get("maximum")
        if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
            if minimum is not None and maximum is not None:
                detail = f"must be between {minimum} and {maximum}"
            elif minimum is not None:
                detail = f"must be at least {minimum}"
            else:
                detail = f"must be at most {maximum}"
            raise StyleError(location, detail)
    if typ == "string":
        if not rule.get("minLength", 0) <= len(value) <= rule.get("maxLength", float("inf")):
            raise StyleError(location, "length is outside the allowed range")
        if "pattern" in rule and not re.fullmatch(rule["pattern"], value):
            detail = (
                "expected a color like #1A2B3C"
                if rule["pattern"] == "^#[0-9A-Fa-f]{6}$"
                else "expected a color like #1A2B3C, or empty"
                if rule["pattern"] == "^(#[0-9A-Fa-f]{6})?$"
                else f"must match {rule['pattern']}"
            )
            raise StyleError(location, detail)
    if typ == "array":
        if "maxItems" in rule and len(value) > rule["maxItems"]:
            raise StyleError(location, f"must contain at most {rule['maxItems']} items")
        for index, item in enumerate(value):
            _check(item, rule["items"], f"{location}[{index}]")
        if rule.get("uniqueItems") and len(set(value)) != len(value):
            label = ("transition look" if location == "style.moves.transitions" else
                     "moment style" if location == "style.moments.styles" else
                     "bleep word" if location.startswith("style.music.bleep_words_") else
                     "card kind")
            raise StyleError(location, f"choose each {label} only once")
    if typ == "object":
        for required in rule.get("required", []):
            if required not in value:
                raise StyleError(f"{location}.{required}", "missing required field")
        for key, child in value.items():
            if key not in rule["properties"]:
                raise StyleError(f"{location}.{key}", "unknown field")
            _check(child, rule["properties"][key], f"{location}.{key}")


def _merge(value: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _merge(value.get(key, {}), child)
        if child["type"] == "object"
        else copy.deepcopy(value.get(key, child["default"]))
        for key, child in rule["properties"].items()
    }


def _style_data(data: dict[str, Any]) -> dict[str, Any]:
    if isinstance(data, dict) and data.get("schema") == RESOLVED_SCHEMA:
        return {key: value for key, value in data.items() if key != "schema"}
    return data


def check_card_kinds_on(kinds: list[str], location: str) -> None:
    """Validate opt-in card kinds after checking their schema shape."""
    from yap_studio.cards import OPT_IN_KINDS
    from yap_studio.hf_registry.cards import kinds as registry_kinds

    allowed = (*OPT_IN_KINDS, *registry_kinds())
    for index, kind in enumerate(kinds):
        if kind not in allowed:
            detail = (f"choose one of {', '.join(allowed)}" if allowed else
                      "no card kind needs turning on; leave this list empty")
            raise StyleError(f"{location}[{index}]", detail)


def validate_style_data(data: dict[str, Any]) -> None:
    """Validate a style, accepting the engine's resolved output marker."""
    _check(_style_data(data), _schema(), "style")
    check_card_kinds_on(data.get("graphics", {}).get("card_kinds_on", []),
                        "style.graphics.card_kinds_on")
    grade = _style_data(data).get("grade", {})
    if grade.get("look") == "eyedropper":
        from yap_studio.grade import GradeError, GradeSpec

        try:
            GradeSpec.from_dict({key: value for key, value in grade.items() if key != "slider"})
        except GradeError as exc:
            error = StyleError("style.grade.color", exc.message)
            error.suggested_action = exc.suggested_action
            raise error from exc


def resolve_style_data(data: dict[str, Any]) -> dict[str, Any]:
    """Validate a style and fill omitted settings from schema defaults."""
    validate_style_data(data)
    raw = _style_data(data)
    result = _merge(raw, _schema())
    grade = raw.get("grade", {})
    if grade.get("look") == "eyedropper" and "strength" not in grade:
        from yap_studio.grade import EYEDROPPER_DEFAULT_STRENGTH

        result["grade"]["strength"] = EYEDROPPER_DEFAULT_STRENGTH
    return apply_brand(result)


def _contrast(a: str, b: str) -> float:
    def luminance(color: str) -> float:
        channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                  for c in channels]
        return sum(c * weight for c, weight in zip(
            linear, (0.2126, 0.7152, 0.0722), strict=True))

    low, high = sorted((luminance(a), luminance(b)))
    return (high + 0.05) / (low + 0.05)


def apply_brand(data: dict[str, Any]) -> dict[str, Any]:
    """Apply a resolved brand to a copy of the look values."""
    result = copy.deepcopy(data)
    brand = result.get("brand", {})
    for key, section in (("captions_font", "captions"), ("titles_font", "hook"),
                         ("cards_font", "graphics")):
        if brand.get(key):
            result[section]["font"] = brand[key]
    accent = brand.get("accent_color")
    if accent:
        result["captions"]["highlight_color"] = accent
        result["hook"]["accent_color"] = accent
        result["graphics"]["accent_color"] = accent
        result["captions"]["highlight_text_color"] = max(
            ("#111111", "#FFFFFF"), key=lambda color: _contrast(color, accent))
    result["captions"] = caption_legibility(result["captions"])
    return result


def caption_legibility(captions: dict[str, Any]) -> dict[str, Any]:
    """Resolve readable ink on caption grounds and registry highlight surfaces."""
    captions = copy.deepcopy(captions)
    # Below half strength, unknown footage dominates the ground's opaque color.
    if (not captions.get("camera_3d", False) and captions["background_mode"] != "none"
            and captions["background_opacity"] >= 0.5):
        captions["text_color"] = max(
            ("#111111", "#FFFFFF"),
            key=lambda color: _contrast(color, captions["background_color"]),
        )
        if _contrast(captions["text_color"], captions["background_color"]) < 4.5:
            captions["text_color"] = "#000000"
    if captions["highlight_style"] in {"highlight", "pill_karaoke"}:
        candidates = (captions["highlight_text_color"], "#111111", "#FFFFFF")
        captions["highlight_text_color"] = max(
            candidates, key=lambda color: _contrast(color, captions["highlight_color"]),
        )
        if _contrast(captions["highlight_text_color"], captions["highlight_color"]) < 4.5:
            captions["highlight_text_color"] = "#000000"
    return captions


def unbranded(data: dict[str, Any]) -> dict[str, Any]:
    """Resolve raw source style values without brand overrides.

    Pass source data, not data already returned by resolve_style_data or load_style.
    A ResolvedStyle from resolve carries its source values in unbranded_data.
    """
    validate_style_data(data)
    return _merge(_style_data(data), _schema())


def cards_on(style: ResolvedStyle) -> tuple[str, ...]:
    """Return the card template ids allowed by a resolved style."""
    from yap_studio.cards import CARD_TEMPLATES, card_kind_off
    from yap_studio.hf_registry.cards import kinds as registry_kinds

    return tuple(kind for kind in (*CARD_TEMPLATES, *registry_kinds())
                 if not card_kind_off(style, kind))


def load_style(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate a style file, then fill omitted settings."""
    return resolve_style_data(_read(Path(path).expanduser()))


@dataclass(frozen=True)
class ResolvedStyle:
    """A resolved style with a reproducible JSON representation and hash."""

    name: str
    source: str
    data: dict[str, Any]
    unbranded_data: dict[str, Any] | None = None
    # Standalone clips must distinguish a supplied placement from the project default.
    caption_placement_explicit: bool = False

    def to_json(self) -> dict[str, Any]:
        """Return a separate JSON object carrying the engine output marker."""
        data = copy.deepcopy(self.data)
        music = data.get("music", {})
        # Omit the opt-in defaults so existing resolved bytes and receipt hashes stay stable.
        if not any(music.get(key) for key in (
            "bleep_swears", "bleep_words_add", "bleep_words_remove",
        )):
            for key in ("bleep_swears", "bleep_words_add", "bleep_words_remove"):
                music.pop(key, None)
        return {"schema": RESOLVED_SCHEMA, **data}

    def text(self) -> str:
        """Return canonical readable JSON with a final newline."""
        return json.dumps(self.to_json(), indent=2, sort_keys=False, ensure_ascii=False) + "\n"

    def sha256(self) -> str:
        """Hash the UTF-8 bytes written by text()."""
        return hashlib.sha256(self.text().encode("utf-8")).hexdigest()


def resolve(spec: str | os.PathLike[str] | None = None) -> ResolvedStyle:
    """Resolve a shipped preset, library id, file, or the current default."""
    from yap_studio.style.library import get_default, library_dir

    if spec is None:
        spec = get_default() or "preset:clean_minimal"
    given = os.fspath(spec)
    if given.startswith("preset:"):
        preset = given.removeprefix("preset:")
        if preset not in SHIPPED_PRESETS:
            raise StyleError("style", f"unknown preset; choose one of {', '.join(SHIPPED_PRESETS)}")
        path = PRESETS_DIR / f"{preset}.json"
        source = given
    elif (
        re.fullmatch(r"[a-z0-9_-]{1,64}", given)
        and (library_dir() / f"{given}.json").is_file()
    ):
        path = library_dir() / f"{given}.json"
        source = f"library:{given}"
    else:
        path = Path(spec).expanduser()
        source = str(path)
    raw = _read(path)
    data = resolve_style_data(raw)
    return ResolvedStyle(name=data["name"], source=source, data=data,
                         unbranded_data=unbranded(raw),
                         caption_placement_explicit="placement" in raw.get("captions", {}))


def write_resolved(style: ResolvedStyle, out: str | os.PathLike[str]) -> Path:
    """Atomically write an engine output that may replace its own JSON."""
    with paths.operation():
        return paths.atomic_write_text(Path(out).expanduser(), style.text())
