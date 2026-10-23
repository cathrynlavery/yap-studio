"""Versioned project contracts, frozen models, and dependency-free validation.

SPECS is the single source for validation and generated draft 2020-12 schemas.
Cut words retain their source id; (id, segment) identifies a cut occurrence.
Template slots and removal evidence are deliberately open JSON objects.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from yap_studio.checks.platform import PLATFORMS
from yap_studio.errors import ValidationError
from yap_studio.transcript.words import norm_words
from yap_studio.video_type import BROLL_SOURCES, VIDEO_TYPES
from yap_studio.zoom_curve import EASES

SCHEMAS_DIR = Path(__file__).parent / "schemas"
RULE_DEFAULTS: dict[str, int | float] = {
    "max_pause_s": 0.3,
    "beat_breath_s": 0.4,
    "elements_at_once_max": 1,
    "zoom_min_s": 2.5,
    "zoom_max_s": 4.0,
    "zoom_peak_max": 1.10,
    "zoom_max_per_video": 8,
    "zoom_scale_max": 2.0,
    "zoom_step_min": 0.15,
    "zoom_rate_min": 0.03,
    "zoom_hold_min_s": 0.6,
    "graphic_hold_min_s": 2.0,
    "graphic_hold_max_s": 3.0,
    "graphic_hold_violation_s": 8.0,
    "insert_min_s": 1.0,
    "insert_max_s": 4.0,
    "overlay_min_s": 1.0,
    "overlay_max_s": 6.0,
    "montage_part_min_s": 0.7,
    "montage_part_max_s": 1.3,
}
ASR_ENGINES = ("auto", "parakeet-mlx", "parakeet-onnx", "faster-whisper", "none")
AUDIO_DEFAULTS = {
    "mode": "loudnorm", "integrated_lufs": -14.0, "true_peak_dbtp": -1.7, "lra_lu": 11.0,
}


class ProjectFileError(ValidationError):
    """Name the project file and precise field needing correction."""

    def __init__(self, filename: str, location: str, detail: str) -> None:
        self.filename = filename
        self.location = location
        super().__init__(f"{filename}: {location}", detail)


def _str(**limits: Any) -> dict[str, Any]:
    return {"type": "string", **limits}


def _num(**limits: Any) -> dict[str, Any]:
    return {"type": "number", **limits}


def _enum(*values: Any) -> dict[str, Any]:
    return {"enum": list(values)}


def _nullable(spec: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [spec, {"type": "null"}]}


def _obj(
    properties: dict[str, Any], required: tuple[str, ...] = (), **constraints: Any
) -> dict[str, Any]:
    return {
        "type": "object", "properties": properties, "required": list(required),
        "additionalProperties": False, **constraints,
    }


def _arr(items: dict[str, Any], **limits: Any) -> dict[str, Any]:
    return {"type": "array", "items": items, **limits}


def _map(items: dict[str, Any], *, keys: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": items,
        **({"propertyNames": keys} if keys else {}),
    }


_TEXT = _str()
_PATH = _str(minLength=1, pattern=r"^[^\u0000]+$")
_ID = _str(minLength=1)
_TAKE_ID = _str(pattern=r"^t[1-9][0-9]*$")
_WORD_ID = _str(pattern=r"^t[1-9][0-9]*:(0|[1-9][0-9]*)$")
_HEX64 = _str(pattern=r"^[0-9a-f]{64}$")
_HASHES = _map(_HEX64, keys=_TAKE_ID)
_BOOL = {"type": "boolean"}
_NONNEG = _num(minimum=0)
_POSITIVE = _num(exclusiveMinimum=0)
_OPEN = {"type": "object", "additionalProperties": True}
_WORD_ANCHOR = _obj({
    "word": _WORD_ID, "text": _TEXT, "edge": _enum("start", "end"),
    "offset_s": _num(), "segment": _ID, "snap": _enum("nearest"),
}, ("word",))
_ANCHOR = {"oneOf": [_WORD_ANCHOR, _obj({"time": _NONNEG}, ("time",))]}
_WORD = _obj({
    "id": _WORD_ID, "text": _TEXT, "start": _NONNEG, "end": _NONNEG,
    "conf": _nullable(_num(minimum=0, maximum=1)),
}, ("id", "text", "start", "end"))
_RECT = {
    "type": "array", "minItems": 4, "maxItems": 4,
    "prefixItems": [_NONNEG, _NONNEG, _POSITIVE, _POSITIVE], "items": False,
}
_BLUR_RECT = {
    **_RECT, "prefixItems": [_NONNEG, _NONNEG, _num(minimum=8), _num(minimum=8)],
}
_MEDIA = _obj({
    "path": _PATH, "sha256": _HEX64, "in_s": _NONNEG,
    "speed": {"anyOf": [_POSITIVE, _enum("fit")]},
    "fit": _enum("cover", "contain", "crop", "split"), "crop": _nullable(_RECT),
    "blur": _arr(_BLUR_RECT), "hdr": _enum("auto", "off"),
}, ("path", "sha256"), allOf=[{
    "if": {"properties": {"fit": {"const": "crop"}}, "required": ["fit"]},
    "then": {"properties": {"crop": _RECT}, "required": ["crop"]},
}, {
    "if": {"properties": {"fit": {"const": "split"}}, "required": ["fit"]},
    "then": {"not": {"required": ["crop"]}},
}])
_OVERLAY_MEDIA = _obj({
    key: value for key, value in _MEDIA["properties"].items()
    if key in {"path", "sha256", "in_s", "speed", "hdr"}
}, ("path", "sha256"))
_PROVENANCE = _obj({
    "kind": _enum("own", "licensed", "generated"), "source": _nullable(_TEXT),
    "license": _nullable(_TEXT), "url": _nullable(_TEXT),
}, ("kind",))
_CARD_BOX = _obj({
    key: {"type": "integer", "minimum": 0 if key in {"x", "y"} else 1,
          "maximum": 1080 if key in {"x", "width"} else 1920}
    for key in ("x", "y", "width", "height")
}, ("x", "y", "width", "height"))
_ELEMENT_FIELDS = {
    "title": {"lines": _arr(_TEXT, minItems=1, maxItems=3), "accent": _nullable(_TEXT)},
    "chapter": {"title": _str(minLength=1, maxLength=48)},
    "graphic": {"template": _ID, "slots": _OPEN, "layer": _enum("behind", "front"),
                "box": _CARD_BOX},
    "insert": {
        "media": _MEDIA, "audio": {"anyOf": [
            _enum("mute"), _obj({"gain_db": _num()}, ("gain_db",)),
        ]},
        "motion": _enum("push", "still"),
        "push_scale": _num(exclusiveMinimum=1, maximum=1.10),
        "group": _nullable(_ID), "matches": _TEXT, "provenance": _PROVENANCE,
        "chronology": _enum("hook", "before_after"),
        "letters": _obj({
            "text": _str(minLength=1, maxLength=40),
            "background": {**_enum("color", "blur"), "default": "color"},
            "word_hold_s": _num(minimum=1.0, maximum=1.5),
            "reveal_frames": {"type": "integer", "minimum": 10, "maximum": 14},
        }, ("text",)),
        "color_match": _BOOL,
    },
    "overlay": {
        "media": _OVERLAY_MEDIA,
        "placement": _enum("headroom", "side_left", "side_right", "tall_behind", "box"),
        "box": _RECT, "fit": {**_enum("cover", "contain"), "default": "cover"},
        "motion": {**_enum("none", "drift", "scroll"), "default": "none"},
        "matches": _TEXT, "provenance": _PROVENANCE,
        "chronology": _enum("hook", "before_after"),
        "color_match": _BOOL,
    },
    "background": {
        "media": _OVERLAY_MEDIA,
        "layout": _enum("full", "bottom", "corner_left", "corner_right"),
        "blend": _enum("none", "wrap"), "enter": _enum("cut", "wipe"),
        "motion": _enum("none", "push"),
        "fit": {**_enum("cover", "contain"), "default": "cover"},
        "matches": _TEXT, "provenance": _PROVENANCE,
        "chronology": _enum("hook", "before_after"),
    },
    "zoom": {
        "move": _enum("punch", "push", "keys", "opening"),
        "peak_scale": _num(exclusiveMinimum=1), "scale": _num(exclusiveMinimum=1),
        "center": _arr(_num(minimum=0, maximum=1), minItems=2, maxItems=2),
        "ease": _enum(*EASES), "push_s": _POSITIVE,
        "keys": _arr(_obj({
            "at": _ANCHOR, "scale": _num(minimum=1),
            "center": _arr(_num(minimum=0, maximum=1), minItems=2, maxItems=2),
            "ease": {**_enum(*EASES), "default": "snap"},
        }, ("at", "scale")), minItems=1),
    },
    "transition": {
        "look": _enum("whip_pan", "cinematic_zoom", "light_leak"),
        "frames": {"type": "integer", "enum": list(range(6, 25, 2)), "default": 12},
    },
    "caption_gap": {},
    "grade_compare": {},
    "music_level": {"level": _num(minimum=0, maximum=1)},
    "sound": {
        "sound": _enum("typing", "message-ping", "camera-shutter", "applause", "laugh", "cheer",
                       "record-scratch", "whoosh", "click", "tick-roll", "chime"),
        "gain_db": _num(minimum=-12, maximum=6),
        "align": _enum("start", "peak"),
    },
    "moment": {"style": _enum(
        "color_pop", "background_blur", "spotlight", "sticker_freeze", "outline",
        "clone_flash",
    )},
}
_ELEMENT_FIELDS = {kind: {**fields, "sfx": {"oneOf": [
    {"const": False}, _obj({"shift_s": _num(minimum=-1, maximum=1)}, ("shift_s",)),
]}} if kind not in {"grade_compare", "sound"} else fields
                   for kind, fields in _ELEMENT_FIELDS.items()}
_ELEMENT_REQUIRED = {
    "chapter": ("title",),
    "title": ("lines",), "graphic": ("template", "slots"), "insert": ("media",),
    "overlay": ("media", "placement"), "background": ("media",),
    "zoom": (), "caption_gap": (), "music_level": ("level",),
    "transition": ("look",),
    "sound": ("sound",),
    "grade_compare": (),
    "moment": ("style",),
}


def _element(kind: str) -> dict[str, Any]:
    if kind in {"transition", "sound"}:
        return _obj({
            "id": _ID, "kind": {"const": kind}, "lane": {"const": "agent"},
            "start": _ANCHOR, **_ELEMENT_FIELDS[kind],
        }, ("id", "kind", "start", *_ELEMENT_REQUIRED[kind]))
    endings = [{"required": ["end"]}, {"required": ["hold_s"]}]
    if kind in {"title", "chapter", "graphic"}:
        endings.append({"not": {"anyOf": endings.copy()}})
    constraints = {}
    if kind == "overlay":
        constraints["allOf"] = [{
            "if": {"properties": {"placement": {"const": "box"}}, "required": ["placement"]},
            "then": {"required": ["box"]},
        }, {
            "if": {"properties": {"placement": _enum("headroom", "side_left", "side_right",
                                                   "tall_behind")},
                   "required": ["placement"]},
            "then": {"not": {"required": ["box"]}},
        }]
    if kind == "zoom":
        endings.append({
            "properties": {"move": {"const": "opening"}}, "required": ["move"],
            "not": {"anyOf": [{"required": ["end"]}, {"required": ["hold_s"]}]},
        })

        def forbid(*fields: str) -> dict[str, Any]:
            return {"not": {"anyOf": [{"required": [field]} for field in fields]}}

        constraints["allOf"] = [{
            "if": {"not": {"required": ["move"]}},
            "then": {"required": ["peak_scale", "center"],
                     **forbid("scale", "ease", "push_s", "keys")},
        }, {
            "if": {"required": ["move"]}, "then": forbid("peak_scale"),
        }, {
            "if": {"properties": {"move": {"const": "punch"}}, "required": ["move"]},
            "then": {"properties": {"ease": {**_enum("cut", "snap"), "default": "snap"}},
                     **forbid("keys", "push_s")},
        }, {
            "if": {"properties": {"move": {"const": "push"}}, "required": ["move"]},
            "then": {"required": ["scale"],
                     "properties": {"ease": {
                         **_enum("linear", "ease-in-out"), "default": "linear",
                     }}, **forbid("keys")},
        }, {
            "if": {"properties": {"move": {"const": "opening"}}, "required": ["move"]},
            "then": forbid("end", "hold_s", "ease", "push_s", "keys", "peak_scale"),
        }, {
            "if": {"properties": {"move": {"const": "keys"}}, "required": ["move"]},
            "then": {"required": ["keys"], **forbid("scale", "ease", "push_s")},
        }]
    return _obj({
        "id": _ID, "kind": {"const": kind},
        "lane": {"const": "agent"} if kind in {"moment", "grade_compare", "chapter"}
        else _enum("agent", "broll"),
        "start": _ANCHOR, "end": _ANCHOR, "hold_s": _POSITIVE,
        **_ELEMENT_FIELDS[kind],
    }, ("id", "kind", "start", *_ELEMENT_REQUIRED[kind]), oneOf=endings, **constraints)


def _fragment_element(kind: str) -> dict[str, Any]:
    spec = _element(kind)
    if kind not in {"insert", "overlay", "background"}:
        spec["properties"]["lane"] = {"const": "agent"}
        spec["required"].append("lane")
    return spec


SPECS: dict[str, dict[str, Any]] = {
    "text-hook": _obj({
        "schema": {"const": "yap-studio/text-hook@1"},
        "ideas": _arr(_str(minLength=1, maxLength=1000), maxItems=3, uniqueItems=True,
                      anyOf=[{"maxItems": 0}, {"minItems": 3}]),
        "chosen": _nullable(_str(minLength=1, maxLength=1000)),
        "source": _enum("typed", "idea", "note"),
        "kind": _enum("text_hook", "opening"),
    }, ("schema", "ideas", "chosen", "source"), allOf=[{
        "if": {"properties": {"chosen": {"type": "null"}}},
        "then": {"properties": {"ideas": {"minItems": 3}}},
    }]),
    "project": _obj({
        "schema": {"const": "yap-studio/project@1"},
        "name": _str(minLength=1, maxLength=80, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$"),
        "style": _nullable(_PATH), "creator": _nullable(_PATH),
        "video_type": _enum(*VIDEO_TYPES),
        "shape": {**_enum("vertical", "horizontal"), "default": "vertical"},
        "broll_sources": _arr(_enum(*BROLL_SOURCES), uniqueItems=True),
        "grade": _obj({
            "look": _enum("none", "film", "eyedropper"),
            "strength": _num(minimum=0, maximum=1),
            "color": _str(pattern=r"^#[0-9A-Fa-f]{6}$"),
            "sample": _obj({"time_s": _NONNEG, "x": _num(minimum=0, maximum=1),
                            "y": _num(minimum=0, maximum=1)}, ("time_s", "x", "y")),
        }, allOf=[{
            "if": {"properties": {"look": {"const": "eyedropper"}}, "required": ["look"]},
            "then": {"oneOf": [{"required": ["color"], "not": {"required": ["sample"]}},
                               {"required": ["sample"], "not": {"required": ["color"]}}]},
        }, {
            "if": {"required": ["sample"]},
            "then": {"required": ["look"],
                     "properties": {"look": {"const": "eyedropper"}}},
        }]),
        "takes": _arr(_obj({
            "id": _TAKE_ID, "path": _PATH, "crop_x": _num(minimum=0, maximum=1),
            "audio_stream": {"type": "integer", "minimum": 0},
            "voiceover": _BOOL,
        }, ("id", "path")), minItems=1),
        "language": {**_str(minLength=1), "default": "en"},
        "asr": _obj({"engine": _enum(*ASR_ENGINES), "model": _nullable(_TEXT)}),
        "platform": _enum("none", *PLATFORMS), "word_fixes": _map(_TEXT),
        "word_text_fixes": _map(_str(minLength=1, pattern=r"\S"), keys=_WORD_ID),
        "caption_backing": _map(_enum("shadow", "box", "none"), keys=_str(pattern=r"^caption:")),
        "audio": {"oneOf": [
            _obj({"mode": {"const": "preserve"}, "match_takes": _BOOL}, ("mode",)),
            _obj({"mode": {"const": "loudnorm"},
                  "integrated_lufs": _num(minimum=-70, maximum=-5),
                  "true_peak_dbtp": _num(minimum=-9, maximum=0),
                  "lra_lu": _num(minimum=1, maximum=50), "match_takes": _BOOL},
                 ("mode", "integrated_lufs", "true_peak_dbtp", "lra_lu")),
        ]},
        "music": _obj({
            "mode": _enum("auto", "elevenlabs", "file", "none"), "file": _nullable(_PATH),
            "description": _nullable(_TEXT), "tags": _arr(_TEXT), "generate": _BOOL,
        }),
        "rules": _obj({
            key: {"type": "integer" if isinstance(value, int) else "number",
                  "minimum": 0, "default": value}
            for key, value in RULE_DEFAULTS.items()
        }),
    }, ("schema", "name", "takes")),
    "words": _obj({
        "schema": {"const": "yap-studio/words@1"}, "take": _TAKE_ID,
        "take_sha256": _HEX64, "time_base": {"const": "source"}, "audio_offset_s": _num(),
        "engine": _enum(*ASR_ENGINES), "model": _nullable(_TEXT),
        "transcript_sha256": _HEX64, "words": _arr(_WORD),
    }, ("schema", "take", "take_sha256", "time_base", "audio_offset_s", "engine",
        "model", "transcript_sha256", "words")),
    "edit": _obj({
        "schema": {"const": "yap-studio/edit@1"}, "words_sha256": _HASHES,
        "segments": _arr(_obj({
            "id": _ID, "take": _TAKE_ID, "from": _ANCHOR, "to": _ANCHOR,
            "text": _TEXT, "note": _TEXT,
        }, ("id", "take", "from", "to"))), "end_hold_s": _nullable(_NONNEG),
    }, ("schema", "words_sha256", "segments")),
    "removals": _obj({
        "schema": {"const": "yap-studio/removals@1"},
        "removals": _arr(_obj({
            "id": _ID, "take": _TAKE_ID, "start": _NONNEG, "end": _NONNEG,
            "kind": _enum("silence", "filler", "stutter", "restart", "manual"),
            "words": _arr(_WORD_ID), "status": _enum("proposed", "applied", "rejected"),
            "by": _TEXT, "listen": _BOOL, "evidence": _OPEN, "reason": _TEXT,
        }, ("id", "take", "start", "end", "kind", "status"))),
    }, ("schema", "removals")),
    "cut-words": _obj({
        "schema": {"const": "yap-studio/cut-words@1"}, "time_base": {"const": "output"},
        "words_sha256": _HASHES, "words": _arr(_obj({
            **_WORD["properties"], "segment": _ID,
        }, (*_WORD["required"], "segment"))),
    }, ("schema", "time_base", "words_sha256", "words")),
    "timeline": _obj({
        "schema": {"const": "yap-studio/timeline@1"}, "words_sha256": _HASHES,
        "elements": _arr({"oneOf": [_element(kind) for kind in _ELEMENT_FIELDS]}),
    }, ("schema", "words_sha256", "elements")),
    "timeline-fragment": _obj({
        "schema": {"const": "yap-studio/timeline@1"},
        "elements": _arr({"oneOf": [_fragment_element(kind) for kind in _ELEMENT_FIELDS]}),
    }, ("schema", "elements")),
}


def json_schema(name: str) -> dict[str, Any]:
    """Return an independent draft 2020-12 schema for a named contract."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"urn:yap-studio:{name}:1", **copy.deepcopy(SPECS[name]),
    }


def _validate(value: Any, spec: dict[str, Any], filename: str, location: str) -> None:
    def fail(detail: str) -> None:
        raise ProjectFileError(filename, location or "$", detail)

    if "not" in spec:
        try:
            _validate(value, spec["not"], filename, location)
        except ProjectFileError:
            pass
        else:
            fail("disallowed shape")
    for keyword in ("anyOf", "oneOf"):
        if keyword in spec:
            failures = []
            matched = 0
            for choice in spec[keyword]:
                try:
                    _validate(value, choice, filename, location)
                    matched += 1
                except ProjectFileError as exc:
                    failures.append(exc)
            if not matched:
                # Follow the discriminant so errors retain their precise field path.
                for choice, error in zip(spec[keyword], failures, strict=True):
                    props = choice.get("properties", {})
                    if isinstance(value, dict) and any(
                        key in value and ("const" not in rule or value[key] == rule["const"])
                        for key, rule in props.items() if key in {"kind", "word", "time"}
                    ):
                        raise error
                if failures and all("required" in choice for choice in spec[keyword]):
                    fields = " or ".join(" and ".join(choice["required"])
                                         for choice in spec[keyword])
                    fail(f"expected exactly one of {fields}")
                if len(failures) == 1:
                    raise failures[0]
                fail("value does not match an allowed shape")
            if keyword == "oneOf" and matched != 1:
                fail("expected exactly one allowed shape")
    for choice in spec.get("allOf", []):
        _validate(value, choice, filename, location)
    if "if" in spec:
        try:
            _validate(value, spec["if"], filename, location)
        except ProjectFileError:
            pass
        else:
            _validate(value, spec.get("then", {}), filename, location)
    typ = spec.get("type")
    numeric = (isinstance(value, (int, float)) and not isinstance(value, bool)
               and (isinstance(value, int) or math.isfinite(value)))
    valid = {
        "object": isinstance(value, dict), "array": isinstance(value, list),
        "string": isinstance(value, str), "boolean": isinstance(value, bool),
        "null": value is None,
        "integer": numeric and value == int(value),
        "number": numeric,
    }
    if typ and not valid[typ]:
        fail(f"expected {typ}")
    if "const" in spec and value != spec["const"]:
        fail(f"expected {spec['const']!r}")
    if "enum" in spec and value not in spec["enum"]:
        fail(f"expected one of {spec['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        for key, accepts in (
            ("minimum", lambda v, bound: v >= bound),
            ("maximum", lambda v, bound: v <= bound),
            ("exclusiveMinimum", lambda v, bound: v > bound),
        ):
            if key in spec and not accepts(value, spec[key]):
                fail(f"outside allowed range ({key}: {spec[key]})")
    if isinstance(value, str):
        if not spec.get("minLength", 0) <= len(value) <= spec.get("maxLength", math.inf):
            fail("length outside allowed range")
        if "pattern" in spec and re.search(spec["pattern"], value) is None:
            fail("expected a word id like t1:12" if spec == _WORD_ID else "invalid format")
    if isinstance(value, dict):
        for key in spec.get("required", []):
            if key not in value:
                raise ProjectFileError(filename, f"{location}.{key}".lstrip("."), "required")
        props = spec.get("properties", {})
        for key, child in value.items():
            here = f"{location}.{key}".lstrip(".")
            if "propertyNames" in spec:
                _validate(key, spec["propertyNames"], filename, here)
            rule = props.get(key, spec.get("additionalProperties", True))
            if rule is False:
                raise ProjectFileError(filename, here, "unknown field")
            if isinstance(rule, dict):
                _validate(child, rule, filename, here)
    if isinstance(value, list):
        if spec.get("uniqueItems") and any(child in value[:index]
                                           for index, child in enumerate(value)):
            fail("items must be unique")
        if not spec.get("minItems", 0) <= len(value) <= spec.get("maxItems", math.inf):
            fail("number of items outside allowed range")
        prefix = spec.get("prefixItems", [])
        for index, child in enumerate(value):
            rule = prefix[index] if index < len(prefix) else spec.get("items", True)
            if rule is False:
                fail("unexpected array item")
            if isinstance(rule, dict):
                _validate(child, rule, filename, f"{location}[{index}]")


@dataclass(frozen=True)
class Take:
    """A read-only source and optional crop/audio selection."""

    id: str
    path: str
    crop_x: float = 0.5
    audio_stream: int | None = None
    voiceover: bool = False


@dataclass(frozen=True)
class Project:
    """Project settings with defaults resolved, independent of the file bytes."""

    name: str
    takes: tuple[Take, ...]
    style: str | None = None
    creator: str | None = None
    language: str = "en"
    asr: dict[str, Any] = field(default_factory=lambda: {"engine": "auto", "model": None})
    platform: str = "none"
    word_fixes: dict[str, str] = field(default_factory=dict)
    word_text_fixes: dict[str, str] = field(default_factory=dict)
    caption_backing: dict[str, str] = field(default_factory=dict)
    music: dict[str, Any] = field(default_factory=lambda: {
        "mode": "auto", "file": None, "description": None, "tags": [], "generate": False,
    })
    rules: dict[str, int | float] = field(default_factory=lambda: dict(RULE_DEFAULTS))
    audio: dict[str, Any] = field(default_factory=lambda: dict(AUDIO_DEFAULTS))
    grade: dict[str, Any] | None = None
    video_type: str | None = None
    broll_sources: tuple[str, ...] | None = None
    shape: str = "vertical"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON with the version marker and resolved defaults."""
        data = asdict(self)
        if self.shape == "vertical":
            data.pop("shape")
        if self.grade is None:
            data.pop("grade")
        if self.video_type is None:
            data.pop("video_type")
        if self.broll_sources is None:
            data.pop("broll_sources")
        else:
            data["broll_sources"] = list(self.broll_sources)
        data["schema"] = "yap-studio/project@1"
        data["takes"] = [
            {key: value for key, value in asdict(take).items()
             if value is not None and (key != "voiceover" or value)}
            for take in self.takes
        ]
        return data


@dataclass(frozen=True)
class Word:
    """A source word on the take's container clock."""

    id: str
    text: str
    start: float
    end: float
    conf: float | None = None


@dataclass(frozen=True)
class Words:
    """Ingested words tied to both the take and transcript bytes."""

    take: str
    take_sha256: str
    time_base: str
    audio_offset_s: float
    engine: str
    model: str | None
    transcript_sha256: str
    words: tuple[Word, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned source-time words contract."""
        return {"schema": "yap-studio/words@1", **asdict(self),
                "words": [asdict(word) for word in self.words]}


@dataclass(frozen=True)
class Anchor:
    """A source word anchor or explicit time, with an optional segment guard."""

    word: str | None = None
    time: float | None = None
    text: str | None = None
    edge: str = "start"
    offset_s: float = 0.0
    segment: str | None = None
    snap: str | None = None

    def matches(self, text: str) -> bool:
        """Compare a text guard using the transcript's normalized vocabulary."""
        return self.text is None or norm_words(self.text) == norm_words(text)

    def to_dict(self) -> dict[str, Any]:
        """Return the appropriate time or word anchor shape."""
        if self.time is not None:
            return {"time": self.time}
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class Segment:
    """One story segment, in ordered source time."""

    id: str
    take: str
    from_: Anchor
    to: Anchor
    text: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the authored segment shape, including from and to anchors."""
        result = {"id": self.id, "take": self.take,
                  "from": self.from_.to_dict(), "to": self.to.to_dict()}
        for key in ("text", "note"):
            if getattr(self, key) is not None:
                result[key] = getattr(self, key)
        return result


@dataclass(frozen=True)
class Edit:
    """The agent's ordered story and optional end hold."""

    words_sha256: dict[str, str]
    segments: tuple[Segment, ...]
    end_hold_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned edit shape."""
        return {"schema": "yap-studio/edit@1", "words_sha256": dict(self.words_sha256),
                "segments": [segment.to_dict() for segment in self.segments],
                "end_hold_s": self.end_hold_s}


@dataclass(frozen=True)
class Removal:
    """A proposed, applied, or rejected source-time removal."""

    id: str
    take: str
    start: float
    end: float
    kind: str
    status: str
    words: tuple[str, ...] = ()
    by: str = "fillers"
    listen: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None


@dataclass(frozen=True)
class Removals:
    """The fillers-owned removal decisions."""

    removals: tuple[Removal, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned removal decisions."""
        return {"schema": "yap-studio/removals@1", "removals": [
            {key: value for key, value in asdict(item).items() if value is not None}
            for item in self.removals
        ]}


@dataclass(frozen=True)
class CutWord:
    """A source word occurrence with output timings and a segment qualifier."""

    id: str
    segment: str
    text: str
    start: float
    end: float
    conf: float | None = None


@dataclass(frozen=True)
class CutWords:
    """The output-time map; repeated source ids have different segments."""

    time_base: str
    words_sha256: dict[str, str]
    words: tuple[CutWord, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned output-time word occurrence map."""
        return {"schema": "yap-studio/cut-words@1", "time_base": self.time_base,
                "words_sha256": dict(self.words_sha256),
                "words": [asdict(word) for word in self.words]}


@dataclass(frozen=True)
class Element:
    """One timeline element; fields holds only the selected kind's payload."""

    id: str
    kind: str
    start: Anchor
    end: Anchor | None = None
    hold_s: float | None = None
    lane: str | None = None
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return an independent element payload with normalized anchor edges."""
        result = {"id": self.id, "kind": self.kind, "start": self.start.to_dict(),
                  **copy.deepcopy(self.fields)}
        if self.end is not None:
            result["end"] = self.end.to_dict()
        if self.hold_s is not None:
            result["hold_s"] = self.hold_s
        if self.lane is not None:
            result["lane"] = self.lane
        return result


@dataclass(frozen=True)
class TimelineLane:
    """One writer's lane, or a fragment with no words hash map."""

    elements: tuple[Element, ...]
    words_sha256: dict[str, str] | None = None
    lane: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a lane or fragment using the same timeline version marker."""
        result = {"schema": "yap-studio/timeline@1",
                  "elements": [element.to_dict() for element in self.elements]}
        if self.words_sha256 is not None:
            result["words_sha256"] = dict(self.words_sha256)
        return result


@dataclass(frozen=True)
class TextHook:
    """The creator's per-video framing line and the ideas offered for it."""

    ideas: tuple[str, ...]
    chosen: str | None
    source: str
    kind: str = "text_hook"

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned plan without mutable model fields."""
        return {"schema": "yap-studio/text-hook@1", "ideas": list(self.ideas),
                "chosen": self.chosen, "source": self.source,
                **({"kind": self.kind} if self.kind != "text_hook" else {})}


Document = Project | Words | Edit | Removals | CutWords | TimelineLane | TextHook


def _anchor(data: dict[str, Any], edge: str) -> Anchor:
    return Anchor(**{**data, "edge": data.get("edge", edge)})


def parse_anchor(data: dict[str, Any], *, edge: str = "start") -> Anchor:
    """Validate a nested timeline anchor using the same contract as element edges."""
    _validate(data, _ANCHOR, "timeline", "anchor")
    return _anchor(data, edge)


def _unique(items: list[dict[str, Any]], filename: str, location: str) -> None:
    seen = set()
    for index, item in enumerate(items):
        if item["id"] in seen:
            raise ProjectFileError(filename, f"{location}[{index}].id", "duplicate id")
        seen.add(item["id"])


def parse_document(
    name: str, data: dict[str, Any], *, filename: str | None = None, lane: str | None = None
) -> Document:
    """Validate a contract and construct frozen models without retaining caller data."""
    filename = filename or f"{name}.json"
    _validate(data, SPECS[name], filename, "")
    data = copy.deepcopy(data)
    if name == "text-hook":
        from yap_studio.text_hook import fit_text_hook

        for text in [*data["ideas"], *([data["chosen"]] if data["chosen"] else [])]:
            if not text.strip():
                raise ProjectFileError(filename, "chosen", "write some text")
            if data.get("kind", "text_hook") == "text_hook":
                fit_text_hook(text)
        if data["chosen"] is not None and data["source"] == "idea":
            if data["chosen"] not in data["ideas"]:
                raise ProjectFileError(filename, "chosen", "choose one of the offered ideas")
        return TextHook(tuple(data["ideas"]), data["chosen"], data["source"],
                        data.get("kind", "text_hook"))
    if name == "project":
        _unique(data["takes"], filename, "takes")
        values = {key: value for key, value in data.items() if key not in {"schema", "takes"}}
        if "broll_sources" in values:
            values["broll_sources"] = tuple(values["broll_sources"])
        values["asr"] = {"engine": "auto", "model": None, **values.get("asr", {})}
        values["music"] = {"mode": "auto", "file": None, "description": None,
                           "tags": [], "generate": False, **values.get("music", {})}
        values["rules"] = {**RULE_DEFAULTS, **values.get("rules", {})}
        return Project(takes=tuple(Take(**item) for item in data["takes"]), **values)
    if name in {"words", "cut-words"}:
        if name == "words":
            _unique(data["words"], filename, "words")
        occurrences = set()
        for index, word in enumerate(data["words"]):
            here = f"words[{index}]"
            if word["end"] < word["start"]:
                raise ProjectFileError(filename, f"{here}.end", "must be at least start")
            if name == "words" and word["id"].split(":")[0] != data["take"]:
                raise ProjectFileError(filename, f"{here}.id", "word belongs to another take")
            occurrence = (word["id"], word.get("segment"))
            if occurrence in occurrences:
                raise ProjectFileError(filename, f"{here}.id", "duplicate word occurrence")
            occurrences.add(occurrence)
        values = {key: value for key, value in data.items() if key not in {"schema", "words"}}
        if name == "words":
            return Words(words=tuple(Word(**item) for item in data["words"]), **values)
        return CutWords(words=tuple(CutWord(**item) for item in data["words"]), **values)
    if name == "edit":
        _unique(data["segments"], filename, "segments")
        segments = []
        for index, item in enumerate(data["segments"]):
            for key in ("from", "to"):
                word = item[key].get("word")
                if word and word.split(":")[0] != item["take"]:
                    raise ProjectFileError(filename, f"segments[{index}].{key}.word",
                                           "word belongs to another take")
            segments.append(Segment(
                id=item["id"], take=item["take"], from_=_anchor(item["from"], "start"),
                to=_anchor(item["to"], "end"), text=item.get("text"), note=item.get("note"),
            ))
        return Edit(data["words_sha256"], tuple(segments), data.get("end_hold_s"))
    if name == "removals":
        _unique(data["removals"], filename, "removals")
        removals = []
        for index, item in enumerate(data["removals"]):
            if item["end"] <= item["start"]:
                raise ProjectFileError(filename, f"removals[{index}].end", "must exceed start")
            item["words"] = tuple(item.get("words", []))
            removals.append(Removal(**item))
        return Removals(tuple(removals))
    _unique(data["elements"], filename, "elements")
    elements = []
    for index, item in enumerate(data["elements"]):
        if lane is not None and item.get("lane", lane) != lane:
            raise ProjectFileError(filename, f"elements[{index}].lane", f"expected {lane!r}")
        if lane == "broll" and item["kind"] not in {"insert", "overlay", "background"}:
            raise ProjectFileError(filename, f"elements[{index}].kind",
                                   "broll accepts inserts, overlays and backgrounds")
        fields = {key: item[key] for key in _ELEMENT_FIELDS[item["kind"]] if key in item}
        if item["kind"] == "overlay":
            fields.setdefault("fit", "cover")
            fields.setdefault("motion", "none")
        if item["kind"] == "background":
            fields.setdefault("fit", "cover")
        if item["kind"] == "transition":
            fields["frames"] = int(item.get("frames", 12))
        elements.append(Element(
            id=item["id"], kind=item["kind"], start=_anchor(item["start"], "start"),
            end=_anchor(item["end"], "end") if "end" in item else None,
            hold_s=item.get("hold_s"), lane=item.get("lane"),
            fields=fields,
        ))
    return TimelineLane(tuple(elements), data.get("words_sha256"), lane)


def _load(name: str, path: str | Path, *, lane: str | None = None) -> Document:
    source = Path(path)
    try:
        data = json.loads(source.read_bytes())
    except (OSError, ValueError, UnicodeError) as exc:
        raise ProjectFileError(source.name, "$", f"cannot read JSON: {exc}") from exc
    return parse_document(name, data, filename=source.name, lane=lane)


def load_project(path: str | Path) -> Project:
    """Read a project.json contract."""
    return cast(Project, _load("project", path))


def load_words(path: str | Path) -> Words:
    """Read an ingest words.json contract."""
    return cast(Words, _load("words", path))


def load_edit(path: str | Path) -> Edit:
    """Read the agent's edit.json contract."""
    return cast(Edit, _load("edit", path))


def load_removals(path: str | Path) -> Removals:
    """Read the fillers-owned removals.json contract."""
    return cast(Removals, _load("removals", path))


def load_cut_words(path: str | Path) -> CutWords:
    """Read cut/cut-words.json in output time."""
    return cast(CutWords, _load("cut-words", path))


def load_timeline(path: str | Path, *, lane: str | None = None) -> TimelineLane:
    """Read a lane and check explicit lane fields against its file name."""
    lane = lane or (Path(path).stem if Path(path).stem in {"agent", "broll"} else None)
    return cast(TimelineLane, _load("timeline", path, lane=lane))


def load_timeline_fragment(path: str | Path, *, lane: str = "broll") -> TimelineLane:
    """Read a writer's fragment; the B-roll lane accepts media elements."""
    return cast(TimelineLane, _load("timeline-fragment", path, lane=lane))
