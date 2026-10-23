"""Describe the edit contract from its validation spec."""

from __future__ import annotations

import copy
import json
from typing import Any

from yap_studio.project.model import SPECS

EDIT_EXAMPLE: dict[str, Any] = {
    "schema": SPECS["edit"]["properties"]["schema"]["const"],
    "words_sha256": {"t1": "0" * 64},
    "segments": [
        {"id": "opening", "take": "t1", "from": {"word": "t1:0", "edge": "start"},
         "to": {"word": "t1:3"}},
        {"id": "ending", "take": "t1", "from": {"time": 2.0}, "to": {"time": 3.5}},
    ],
    "end_hold_s": 0.5,
}


def edit_example() -> dict[str, Any]:
    """Return an independent edit example."""
    return copy.deepcopy(EDIT_EXAMPLE)


def _shape(spec: dict[str, Any]) -> str:
    if "const" in spec:
        return f"const {json.dumps(spec['const'])}"
    if "enum" in spec:
        return "enum " + ", ".join(json.dumps(value) for value in spec["enum"])
    if "anyOf" in spec:
        return " or ".join(_shape(choice) for choice in spec["anyOf"])
    if "oneOf" in spec:
        return "word or time anchor (shapes below)"
    kind = spec["type"]
    if kind == "array":
        return f"array of {_shape(spec['items'])}"
    if kind == "object":
        if "properties" in spec:
            return "object"
        keys = _shape(spec["propertyNames"])
        values = _shape(spec["additionalProperties"])
        return f"object (keys: {keys}; values: {values})"
    details = [kind]
    for key in ("pattern", "minimum", "exclusiveMinimum", "maximum", "minLength"):
        if key in spec:
            details.append(f"{key} {spec[key]}")
    return " ".join(details)


def _fields(spec: dict[str, Any], indent: str) -> list[str]:
    required = set(spec["required"])
    return [
        f"{indent}{name}: {'required' if name in required else 'optional'}; {_shape(rule)}"
        for name, rule in spec["properties"].items()
    ]


def edit_schema_text() -> str:
    """Describe edit.json fields and show a valid example."""
    spec = SPECS["edit"]
    segments = spec["properties"]["segments"]["items"]
    lines = ["edit.json format:", *_fields(spec, "  "), "", "Segment fields:",
             *_fields(segments, "    ")]
    for name in ("from", "to"):
        lines.extend(("", f"{name} anchor shapes:"))
        for choice in segments["properties"][name]["oneOf"]:
            anchor_type = "word" if "word" in choice["properties"] else "time"
            lines.extend((f"  {anchor_type} anchor:", *_fields(choice, "    ")))
    lines.extend((
        "",
        "words_sha256 maps each take id used by a segment to the SHA-256 of the bytes of",
        "DIR/ingest/<take>/words.json (for example, shasum -a 256 DIR/ingest/t1/words.json).",
        "Word ids come from `yap-studio words DIR`; t1:12 means take t1, word 12.",
        "from defaults to the word's start edge; to defaults to its end edge.",
        "A time anchor is seconds in the take's source time.",
        "Anchor text guards the source word's text; offset_s shifts a word edge in seconds.",
        "segment and snap are used by timeline anchors; edit.json can leave them out.",
        "Segment text is the words you expect to keep; edit check warns when the cut's "
        "words differ.",
        "Segments play in list order.",
        "end_hold_s null or missing uses the style's end hold.",
        "Check it with `yap-studio edit check DIR`.",
        "After a transcript change, run `yap-studio edit rebase DIR` to keep the recorded cut.",
        "",
        "The example passes the format check. Replace the hash, take and word ids with "
        "your project's before edit check.",
        "Example edit.json:",
        json.dumps(edit_example(), indent=2),
    ))
    return "\n".join(lines)
