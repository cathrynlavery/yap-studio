"""B-roll style defaults, schema validation, and approved taste rules."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from yap_studio.style import (
    PRESETS_DIR,
    RESOLVED_SCHEMA,
    StyleError,
    load_style,
    loader,
    resolve,
    resolve_style_data,
    validate_style_data,
)
from yap_studio.taste import (
    TasteError,
    add_proposals,
    apply_taste,
    approve,
    check_taste,
    load_taste,
    validate_body,
    validate_style_field,
)
from yap_studio.taste.learn import UNMEASURED, _proposed_style

ROOT = Path(__file__).parents[1]


def test_cutout_follows_existing_broll_fields() -> None:
    fields = list(loader._schema()["properties"]["broll"]["properties"])
    assert fields.index("cutout") > fields.index("avoid_tags")


def test_tall_framing_schema_choices_and_default() -> None:
    rule = loader._schema()["properties"]["broll"]["properties"]["tall_framing"]
    assert rule["enum"] == ["auto", "keep", "shrink"]
    assert resolve("preset:clean_minimal").data["broll"]["tall_framing"] == rule["default"]


def test_tall_framing_default_can_change_in_schema_alone(monkeypatch) -> None:
    schema = copy.deepcopy(loader._schema())
    schema["properties"]["broll"]["properties"]["tall_framing"]["default"] = "keep"
    monkeypatch.setattr(loader, "_schema", lambda: schema)
    assert resolve_style_data({"schema_version": 1})["broll"]["tall_framing"] == "keep"


BROLL_DEFAULTS = {
    "cutout": "auto",
    "green_screen_layout": "full",
    "screen_recording_placement": "full_frame",
    "clip_placement": "full_frame",
    "photo_placement": "overlay",
    "screenshot_placement": "overlay",
    "overlay_box": "headroom",
    "overlay_motion": "none",
    "still_push": "in",
    "still_push_scale": 1.08,
    "avoid_tags": [],
    "stock_clips": "off",
    "generated_pictures": "off",
    "emphasis_on": [],
    "color_match": "on",
    "tall_framing": (
        loader._schema()["properties"]["broll"]["properties"]["tall_framing"]["default"]),
}
BROLL_CHOICES = {
    "cutout": "screen",
    "green_screen_layout": "bottom",
    "screen_recording_placement": "overlay",
    "clip_placement": "overlay",
    "photo_placement": "full_frame",
    "screenshot_placement": "full_frame",
    "overlay_box": "side_right",
    "overlay_motion": "drift",
    "still_push": "out",
    "still_push_scale": 1.02,
    "avoid_tags": ["stock_clip", "busy_background"],
    "stock_clips": "ask",
    "generated_pictures": "ask",
    "emphasis_on": ["video_letters"],
    "color_match": "off",
    "tall_framing": "shrink",
}
VALID_FIELDS = [
    (key, value)
    for key in BROLL_DEFAULTS
    if key.endswith("_placement")
    for value in ("full_frame", "overlay", "split")
] + [
    ("cutout", "auto"),
    ("cutout", "screen"),
    ("cutout", "person"),
    ("photo_placement", "green_screen"),
    ("screenshot_placement", "green_screen"),
    *(("green_screen_layout", value) for value in
      ("full", "bottom", "corner_left", "corner_right")),
    ("stock_clips", "off"),
    ("stock_clips", "ask"),
    ("generated_pictures", "off"),
    ("generated_pictures", "ask"),
    ("overlay_box", "headroom"),
    ("overlay_box", "side_left"),
    ("overlay_box", "side_right"),
    ("overlay_box", "tall_behind"),
    ("overlay_motion", "none"),
    ("overlay_motion", "drift"),
    ("overlay_motion", "scroll"),
    *(("tall_framing", value) for value in ("auto", "keep", "shrink")),
    ("color_match", "off"),
    ("color_match", "on"),
    ("still_push", "in"),
    ("still_push", "out"),
    ("still_push", "none"),
    ("still_push_scale", 1.01),
    ("still_push_scale", 1.10),
    ("avoid_tags", []),
    ("avoid_tags", ["a", "0", "_", "screen_recording_2", "a" * 40]),
    ("avoid_tags", [f"tag_{index}" for index in range(20)]),
    ("emphasis_on", []),
    ("emphasis_on", ["video_letters"]),
]
INVALID_FIELDS = [
    (key, value, f"broll.{key}")
    for key in BROLL_DEFAULTS
    if key not in {"avoid_tags", "still_push_scale", "emphasis_on"}
    for value in ("unknown", None, True, 1, [], {})
] + [
    ("still_push_scale", value, "broll.still_push_scale")
    for value in (None, True, "1.08", 1, 1.001, 1.2, [], {})
] + [
    ("clip_placement", "green_screen", "broll.clip_placement"),
    ("screen_recording_placement", "green_screen", "broll.screen_recording_placement"),
] + [
    ("avoid_tags", value, "broll.avoid_tags")
    for value in ("stock_clip", None, True, 1, {}, [f"tag_{index}" for index in range(21)])
] + [
    ("avoid_tags", ["valid_tag", tag], "broll.avoid_tags[1]")
    for tag in ("", "Uppercase", "two words", "a-b", "a.b", "café", "a" * 41, "a\n",
                1, True, None, {}, [])
] + [
    ("emphasis_on", value, "broll.emphasis_on")
    for value in ("video_letters", None, True, 1, {}, ["video_letters"] * 2,
                  ["video_letters"] * 9)
] + [
    ("emphasis_on", [value], "broll.emphasis_on[0]")
    for value in ("unknown", 1, None, {})
]


def _style_sources() -> list[Path]:
    sources = [
        *PRESETS_DIR.glob("*.json"),
        *(ROOT / "docs/design/style-page/proposed-presets").glob("*.json"),
        *(ROOT / "tests").rglob("*.json"),
    ]
    styles = []
    for source in sorted(sources):
        data = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("schema_version") == 1:
            styles.append(source)
    return styles


@pytest.fixture(autouse=True)
def isolated_taste(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))


def _body(key: str, value: Any) -> dict:
    return {
        "scope": "broll",
        "condition": None,
        "sentence": f"Use {value} for {key}.",
        "machine": {"kind": "style_field", "field": f"broll.{key}", "value": value},
        "sources": [{
            "type": "comment",
            "media_sha256": "a" * 64,
            "comment_id": "broll-note",
            "label": "fix",
            "start_frame": None,
            "end_frame": None,
            "quote": "Screen recordings full frame.",
        }],
        "context": None,
    }


def test_broll_schema_is_additive_with_defaults_for_every_field() -> None:
    schema = loader._schema()
    Draft202012Validator.check_schema(schema)
    assert schema["required"] == ["schema_version"]
    assert schema["properties"]["schema_version"]["const"] == 1
    broll = schema["properties"]["broll"]
    assert broll["default"] == {}
    assert not broll["additionalProperties"]
    assert {key: rule["default"] for key, rule in broll["properties"].items()} == BROLL_DEFAULTS
    assert all(rule["description"] for rule in broll["properties"].values())


@pytest.mark.parametrize("source", _style_sources(), ids=lambda source: source.stem)
def test_existing_styles_resolve_with_broll_defaults(source: Path) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    before = copy.deepcopy(data)
    validate_style_data(data)
    resolved = load_style(source)
    assert resolved["broll"] == BROLL_DEFAULTS
    assert resolved["schema_version"] == 1
    for section, value in data.items():
        if isinstance(value, dict):
            assert all(resolved[section][key] == setting for key, setting in value.items())
        else:
            assert resolved[section] == value
    assert data == before
    assert resolve_style_data({"schema": RESOLVED_SCHEMA, **resolved}) == resolved


@pytest.mark.parametrize("section", [{}, {"broll": {}}, {"broll": BROLL_DEFAULTS}])
def test_omitted_empty_and_explicit_broll_defaults_match(section: dict) -> None:
    resolved = resolve_style_data({"schema_version": 1, **section})
    assert resolved["broll"] == BROLL_DEFAULTS
    assert resolved == resolve_style_data({"schema_version": 1})


@pytest.mark.parametrize(("key", "value"), VALID_FIELDS)
def test_broll_values_resolve_and_validate_as_taste_fields(key: str, value: Any) -> None:
    data = {"schema_version": 1, "broll": {key: value}}
    assert Draft202012Validator(loader._schema()).is_valid(data)
    assert resolve_style_data(data)["broll"] == {**BROLL_DEFAULTS, key: value}
    assert validate_style_field(f"broll.{key}", value, "proposal.machine") == "broll"
    body = _body(key, value)
    assert validate_body(body) == body


def test_explicit_broll_choices_resolve_without_sharing_arrays() -> None:
    data = {"schema_version": 1, "broll": copy.deepcopy(BROLL_CHOICES)}
    resolved = resolve_style_data(data)
    assert resolved["broll"] == BROLL_CHOICES
    resolved["broll"]["avoid_tags"].append("extra_tag")
    assert data["broll"] == BROLL_CHOICES
    defaults = resolve_style_data({"schema_version": 1})
    defaults["broll"]["avoid_tags"].append("extra_tag")
    assert resolve_style_data({"schema_version": 1})["broll"] == BROLL_DEFAULTS


@pytest.mark.parametrize(("key", "value", "location"), INVALID_FIELDS)
def test_invalid_broll_values_and_tags_are_refused(key: str, value: Any, location: str) -> None:
    data = {"schema_version": 1, "broll": {key: value}}
    assert not Draft202012Validator(loader._schema()).is_valid(data)
    for validate in (validate_style_data, resolve_style_data):
        with pytest.raises(StyleError) as caught:
            validate(data)
        assert caught.value.parameter == f"style.{location}"
    with pytest.raises(TasteError) as caught:
        validate_body(_body(key, value))
    assert caught.value.parameter == location.replace(f"broll.{key}", "proposal.machine.value")


@pytest.mark.parametrize("scope", ["cut", "captions", "cards", "pacing", "music"])
def test_broll_style_rules_require_broll_scope(scope: str) -> None:
    with pytest.raises(TasteError, match="must be 'broll'") as caught:
        validate_body({**_body("screen_recording_placement", "overlay"), "scope": scope})
    assert caught.value.parameter == "proposal.scope"


def test_broll_scope_refuses_other_sections_and_unknown_fields() -> None:
    body = _body("screen_recording_placement", "overlay")
    with pytest.raises(TasteError, match="must be 'captions'"):
        validate_body({**body, "machine": {
            "kind": "style_field", "field": "captions.position", "value": "bottom",
        }})
    for field in ("broll.unknown", "broll", "broll.avoid_tags.item"):
        with pytest.raises(TasteError, match="unknown style field"):
            validate_style_field(field, "overlay", "proposal.machine")
    with pytest.raises(StyleError, match="unknown field"):
        validate_style_data({"schema_version": 1, "broll": {"unknown": "overlay"}})
    with pytest.raises(StyleError, match="expected object"):
        validate_style_data({"schema_version": 1, "broll": []})


@pytest.mark.parametrize(("key", "value"), BROLL_CHOICES.items())
def test_approved_broll_taste_applies_to_resolved_style(key: str, value: Any) -> None:
    body = _body(key, value)
    identity = add_proposals(body)[0].id
    style = resolve("preset:clean_minimal")
    before = style.to_json()
    assert apply_taste(style).style.data == style.data
    approve(identity)
    application = apply_taste(style)
    assert application.style.data["broll"] == {**BROLL_DEFAULTS, key: value}
    assert application.applied == ({"id": identity, "field": f"broll.{key}", "value": value},)
    assert application.skipped == ()
    assert style.to_json() == before


def test_broll_fields_are_unmeasured_and_keep_base_choices_in_learning() -> None:
    missing = [row for row in UNMEASURED if row["field"].startswith("broll.")]
    assert {row["field"] for row in missing} == {f"broll.{key}" for key in BROLL_DEFAULTS}
    assert len(missing) == len(BROLL_DEFAULTS)
    assert all(row["reason"] == "Timeline choices are not measured from pixels." for row in missing)
    base = resolve_style_data({"schema_version": 1, "broll": BROLL_CHOICES})
    proposal = _proposed_style({}, base, {"kind": "schema_defaults"}, list(UNMEASURED))
    assert proposal["style"]["broll"] == BROLL_CHOICES
    for row in missing:
        assert proposal["sources"][row["field"]] == {
            "source": "base", "status": None, "reason": row["reason"],
        }


def test_broll_taste_check_reports_timeline_choices_as_unmeasured(synth_media) -> None:
    for key, value in BROLL_CHOICES.items():
        approve(add_proposals(_body(key, value))[0].id)
    taste = load_taste()
    application = apply_taste(resolve("preset:clean_minimal"), taste.snapshot())
    result = check_taste(synth_media.path("calib_vertical"), application.pin(), taste=taste)
    assert not result.coverage["ran"]
    assert len(result.rules) == len(BROLL_DEFAULTS)
    assert {row["field"] for row in result.rules} == {f"broll.{key}" for key in BROLL_DEFAULTS}
    for row in result.rules:
        assert row["status"] == "not_measured"
        assert row["measured"] is None
        assert row["frames"] == []
        assert row["reason"] == "Timeline choices are not measured from pixels."
    assert all(finding.severity == "info" for finding in result.findings)
