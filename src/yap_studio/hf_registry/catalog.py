"""Validate metadata for the registry release pinned to the renderer."""

from __future__ import annotations

import json
import math
import re
from functools import cache
from pathlib import Path, PurePosixPath
from string import Formatter

from yap_studio.errors import ValidationError
from yap_studio.render.hyperframes import HYPERFRAMES_VERSION

CATALOG_PATH = Path(__file__).with_name("catalog.json")
SCHEMA = "yap-studio/hf-registry-catalog@1"
# HyperFrames 0.8.184 token fallbacks would paint an opaque backdrop over the speaker.
HOST_TOKENS = {"--bg": "transparent"}

_NAMED_COLORS = frozenset("""
aliceblue antiquewhite aqua aquamarine azure beige bisque black blanchedalmond blue blueviolet
brown burlywood cadetblue chartreuse chocolate coral cornflowerblue cornsilk crimson cyan
darkblue darkcyan darkgoldenrod darkgray darkgreen darkgrey darkkhaki darkmagenta darkolivegreen
darkorange darkorchid darkred darksalmon darkseagreen darkslateblue darkslategray darkslategrey
darkturquoise darkviolet deeppink deepskyblue dimgray dimgrey dodgerblue firebrick floralwhite
forestgreen fuchsia gainsboro ghostwhite gold goldenrod gray green greenyellow grey honeydew
hotpink indianred indigo ivory khaki lavender lavenderblush lawngreen lemonchiffon lightblue
lightcoral lightcyan lightgoldenrodyellow lightgray lightgreen lightgrey lightpink lightsalmon
lightseagreen lightskyblue lightslategray lightslategrey lightsteelblue lightyellow lime limegreen
linen magenta maroon mediumaquamarine mediumblue mediumorchid mediumpurple mediumseagreen
mediumslateblue mediumspringgreen mediumturquoise mediumvioletred midnightblue mintcream mistyrose
moccasin navajowhite navy oldlace olive olivedrab orange orangered orchid palegoldenrod palegreen
paleturquoise palevioletred papayawhip peachpuff peru pink plum powderblue purple rebeccapurple red
rosybrown royalblue saddlebrown salmon sandybrown seagreen seashell sienna silver skyblue slateblue
slategray slategrey snow springgreen steelblue tan teal thistle tomato transparent turquoise violet
wheat white whitesmoke yellow yellowgreen
""".split())


def validate_color_tokens(name: str, entry: dict, source: str) -> None:
    """Require palette mappings for colour fallbacks in CSS and authored scripts."""
    # A font family may share a colour's name, such as Black.
    source = re.sub(r"(?<![\w-])font-family\s*:[^;{}]+", "", source, flags=re.I)
    missing = set()
    # A lookahead also visits nested fallbacks, such as var(--a, var(--b, red)).
    for token, fallback in re.findall(r"(?=var\(\s*(--[\w-]+)\s*,\s*([^,)]+))", source):
        literal = fallback.strip().lower()
        is_color = (literal in _NAMED_COLORS or re.fullmatch(r"#[0-9a-f]{3,8}", literal)
                    or re.match(r"(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\(", literal))
        if is_color and token not in entry["tokens"] and token not in HOST_TOKENS:
            missing.add(token)
    _require(not missing, f"{name}: unmapped colour token: " + ", ".join(sorted(missing)))


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValidationError("HyperFrames registry catalog", detail)


def _relative_path(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and "\\" not in value
        and ":" not in value
        and not PurePosixPath(value).is_absolute()
        and all(part not in {"..", ".", ""} for part in value.split("/"))
    )


def _number(value: object, low: float = 0) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= low)


def _dimensions(value: object) -> bool:
    return (isinstance(value, list) and len(value) == 2
            and all(type(n) is int and n > 0 for n in value))


def _card_metadata(name: str, entry: dict) -> None:
    """Check the opt-in adapter contract before it reaches card planning."""
    from yap_studio.cards import MIN_TEXT_PX, _slots

    kind = entry.get("kind")
    _require(isinstance(kind, dict), f"{name}: kind must be an object")
    _require(kind.get("category") in ("Words", "Numbers", "Media", "Chat", "Editor", "Video"),
             f"{name}: unknown kind category")
    _require(kind.get("role") in ("titles", "cards"), f"{name}: role must be titles or cards")
    for key in ("label", "use"):
        _require(isinstance(kind.get(key), str) and bool(kind[key].strip()),
                 f"{name}: kind.{key} must be nonempty text")
    _require(entry.get("mount") in ("fill", "scale"), f"{name}: mount must be fill or scale")
    _require(entry.get("readability") in ("halo", "none"),
             f"{name}: readability must be halo or none")
    _require(_dimensions(entry.get("min_panel")), f"{name}: min_panel must be [width, height]")
    if entry["mount"] == "scale":
        _require(_dimensions(entry.get("stage")), f"{name}: stage must be [width, height]")
        content = entry.get("content")
        _require(isinstance(content, list) and len(content) == 4
                 and all(_number(n) for n in content)
                 and content[2] > 0 and content[3] > 0,
                 f"{name}: content must be [x, y, width, height]")
        _require(content[0] + content[2] <= entry["stage"][0]
                 and content[1] + content[3] <= entry["stage"][1],
                 f"{name}: content must fit inside stage")
        maximum = entry.get("scale_max")
        _require(_number(maximum) and 0 < maximum <= 2, f"{name}: scale_max must be > 0 and <= 2")
    _require(_number(entry.get("min_s"), 0.5), f"{name}: min_s must be at least 0.5")
    slots = entry.get("slots")
    _require(isinstance(slots, dict), f"{name}: slots must be an object")
    for slot, rule in slots.items():
        _require(re.fullmatch(r"[a-z][a-z0-9_]*", slot) is not None,
                 f"{name}: invalid slot name {slot!r}")
        _require(isinstance(rule, dict) and type(rule.get("required")) is bool,
                 f"{name}: slots.{slot} needs a required boolean")
        slot_type = rule.get("type")
        _require(slot_type in ("string", "integer", "enum"), f"{name}: invalid slot type")
        if slot_type == "string":
            low, high = rule.get("min_length", 0), rule.get("max_length")
            _require(type(low) is int and type(high) is int and 0 <= low <= high,
                     f"{name}: slots.{slot} needs valid min/max_length")
        elif slot_type == "integer":
            low, high = rule.get("minimum"), rule.get("maximum")
            _require(type(low) is int and type(high) is int and low <= high,
                     f"{name}: slots.{slot} needs valid minimum/maximum")
        else:
            options = rule.get("options")
            _require(isinstance(options, list) and bool(options)
                     and all(isinstance(option, str) and bool(option) for option in options)
                     and len(set(options)) == len(options), f"{name}: invalid enum options")
        _require(rule["required"] or "default" in rule,
                 f"{name}: optional slots.{slot} needs a default")
        if "default" in rule:
            _slots({slot: rule["default"]}, {slot: rule})
    preview = entry.get("preview")
    _require(isinstance(preview, dict), f"{name}: preview must be an object")
    try:
        _slots(preview, slots)
    except ValidationError as exc:
        raise ValidationError("HyperFrames registry catalog",
                              f"{name}: preview.{exc.parameter}: {exc.detail}") from exc
    measured = entry.get("measured")
    _require(isinstance(measured, dict), f"{name}: measured must be an object")
    _require(_number(measured.get("min_text_px"), MIN_TEXT_PX),
             f"{name}: measured.min_text_px must be at least {MIN_TEXT_PX}")
    at = measured.get("at")
    _require(isinstance(at, dict), f"{name}: measured.at must be an object")
    _require(_dimensions(at.get("panel")) and at["panel"] == entry["min_panel"],
             f"{name}: measured.at.panel must equal min_panel")
    _require(isinstance(at.get("slots"), dict), f"{name}: measured.at.slots must be an object")
    try:
        values, _ = _slots(at["slots"], slots)
    except ValidationError as exc:
        raise ValidationError("HyperFrames registry catalog",
                              f"{name}: measured.at.{exc.parameter}: {exc.detail}") from exc
    for slot, rule in slots.items():
        if rule["type"] == "string":
            _require(len(values[slot]) == rule["max_length"],
                     f"{name}: measured.at.slots.{slot} must have exactly max_length characters")
        elif rule["type"] == "integer":
            _require(values[slot] == rule["maximum"],
                     f"{name}: measured.at.slots.{slot} must equal maximum")
    variables = entry.get("variables")
    _require(isinstance(variables, dict), f"{name}: variables must be an object")
    for variable, value in variables.items():
        _require(bool(variable) and (isinstance(value, str) or _number(value, -math.inf)),
                 f"{name}: variables.{variable} must be text or a number")
        if not isinstance(value, str):
            continue
        try:
            fields = list(Formatter().parse(value))
        except ValueError as exc:
            raise ValidationError("HyperFrames registry catalog",
                                  f"{name}: invalid variable placeholder") from exc
        for _, field, spec, conversion in fields:
            _require(field is None or (field in slots and not spec and conversion is None),
                     f"{name}: variables.{variable} must reference declared slots")
    tokens = entry.get("tokens")
    _require(isinstance(tokens, dict), f"{name}: tokens must be an object")
    for token, source in tokens.items():
        _require(re.fullmatch(r"--[a-z0-9-]+", token) is not None
                 and source in ("accent", "accent_tint", "ink", "paper"),
                 f"{name}: invalid token {token!r}")


@cache
def load_catalog() -> dict:
    """Read and validate the shipped catalog once per process."""
    try:
        data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValidationError(
            "HyperFrames registry catalog", f"cannot read metadata: {exc}"
        ) from exc
    _require(isinstance(data, dict), "expected an object")
    _require(data.get("schema") == SCHEMA, f"schema must be {SCHEMA}")
    _require(
        data.get("hyperframes_version") == HYPERFRAMES_VERSION,
        f"catalog version {data.get('hyperframes_version')!r} differs from renderer version "
        f"{HYPERFRAMES_VERSION}; a HyperFrames bump must update the catalog in the same change",
    )
    registry = data.get("registry")
    _require(isinstance(registry, dict), "registry must be an object")
    commit = registry.get("commit")
    _require(
        isinstance(commit, str) and re.fullmatch(r"[0-9a-fA-F]{40}", commit) is not None,
        "registry commit must be 40 hex characters",
    )
    expected_url = f"https://raw.githubusercontent.com/heygen-com/hyperframes/{commit}/registry"
    _require(registry.get("url") == expected_url, f"registry url must be {expected_url}")
    for key, expected in {
        "repository": "github.com/heygen-com/hyperframes",
        "tag": f"v{HYPERFRAMES_VERSION}",
        "license": "Apache-2.0",
        "license_file": "LICENSES/Apache-2.0.txt",
        "copyright": "Copyright 2026 HeyGen, Inc.",
    }.items():
        _require(registry.get(key) == expected, f"registry {key} must be {expected!r}")
    entries = data.get("items")
    _require(isinstance(entries, dict) and bool(entries), "items must be a nonempty object")
    for name, entry in entries.items():
        _require(
            re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is not None,
            f"invalid item name {name!r}",
        )
        _require(isinstance(entry, dict), f"{name}: item must be an object")
        _require(len(f"hf:{name}") <= 64, f"{name}: hf: kind id must be at most 64 characters")
        kind = entry.get("type")
        _require(kind in ("block", "component"), f"{name}: type must be block or component")
        upstream = f"registry/{kind}s/{name}/{name}.html"
        _require(
            entry.get("upstream_path") == upstream, f"{name}: upstream_path must be {upstream}"
        )
        _require(entry.get("status") in ("on", "pilot", "helper"),
                 f"{name}: status must be on, pilot or helper")
        if entry["status"] == "on":
            _card_metadata(name, entry)
        if entry["status"] == "helper":
            for field in ("kind", "slots", "preview"):
                _require(field not in entry, f"{name}: helper must not have {field}")
        files = entry.get("files")
        _require(
            isinstance(files, dict) and bool(files), f"{name}: files must be a nonempty object"
        )
        for file, digest in files.items():
            _require(_relative_path(file), f"{name}: unsafe file path {file!r}")
            _require(
                isinstance(digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", digest) is not None,
                f"{name}: hash for {file} must be 64 hex characters",
            )
        _require(
            isinstance(entry.get("main"), str) and entry["main"] in files,
            f"{name}: main must name one of files",
        )
    return data


def items() -> dict[str, dict]:
    """Return catalog items in their shipped order."""
    return load_catalog()["items"]


def item(name: str) -> dict:
    """Return one item or an error listing the known names."""
    entries = items()
    if name not in entries:
        raise ValidationError(
            "registry item", f"unknown name {name!r}; known items: " + ", ".join(entries)
        )
    return entries[name]


def registry_url() -> str:
    """Return the registry URL at the pinned commit."""
    return load_catalog()["registry"]["url"]


def card_catalog() -> dict:
    """Keep install-only helpers outside ordinary card pixel identities."""
    return {**load_catalog(), "items": {name: entry for name, entry in items().items()
                                      if entry["status"] != "helper"}}


def card_catalog_source(source: bytes) -> bytes:
    """Remove helper entries while preserving the card catalog's original formatting."""
    text = source.decode("utf-8")
    decoder = json.JSONDecoder()
    position = re.search(r'"items"\s*:\s*\{', text).end()
    entries = []
    while True:
        start = position
        position += re.match(r"[\s,]*", text[position:]).end()
        if text[position] == "}":
            break
        name, position = decoder.raw_decode(text, position)
        position += re.match(r"\s*:\s*", text[position:]).end()
        entry, end = decoder.raw_decode(text, position)
        entries.append((start, end, entry.get("status") == "helper"))
        position = end
    # Helpers are appended after the card entries; retain the separator for a following card.
    for index in range(len(entries) - 1, -1, -1):
        start, end, helper = entries[index]
        if helper:
            if index == 0 and index + 1 < len(entries):
                end = text.index(",", end) + 1
            text = text[:start] + text[end:]
    return text.encode("utf-8")
