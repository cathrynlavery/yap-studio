"""Style validation, resolution, and creator library behavior."""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from yap_studio.errors import OutputPathError, ValidationError, YapError
from yap_studio.faces import (
    DEFAULT_CAPTION_GAP_PX,
    DEFAULT_FACE_CLEARANCE_PX,
    caption_gap_px,
    face_clearance_px,
)
from yap_studio.style import (
    PRESETS_DIR,
    RESOLVED_SCHEMA,
    SCHEMA_PATH,
    SHIPPED_PRESETS,
    StyleError,
    StyleRefusal,
    add,
    config_root,
    get_default,
    library,
    library_dir,
    list_entries,
    load_style,
    loader,
    resolve,
    resolve_style_data,
    set_default,
    validate_style_data,
    write_resolved,
)

ROOT = Path(__file__).parents[1]
PROPOSED = ROOT / "docs" / "design" / "style-page" / "proposed-presets"


@pytest.fixture(autouse=True)
def isolated_library(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))


def _file(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_minimal_file_fills_clean_minimal_settings(tmp_path: Path) -> None:
    result = load_style(_file(tmp_path / "minimal.json", {"schema_version": 1}))
    clean = resolve("preset:clean_minimal").data
    assert result["schema_version"] == 1
    assert result["name"] == "Untitled style"
    for section in ("captions", "hook", "chapters", "graphics", "broll", "voice", "music",
                    "internal", "grade"):
        assert result[section] == clean[section]
    assert result["captions"]["font"] == "Inter"
    assert result["captions"]["width_scale"] == 1
    assert result["hook"]["font"] == "Gelasio"


def test_grade_defaults_and_presets() -> None:
    assert resolve_style_data({"schema_version": 1})["grade"] == {
        "look": "none", "strength": 1, "color": None, "slider": False}
    for name in SHIPPED_PRESETS:
        assert resolve(f"preset:{name}").data["grade"]["look"] == "none"
        assert resolve(f"preset:{name}").data["grade"]["slider"] is False
    for bad in ({"look": "other"}, {"strength": -1}, {"strength": True}, {"extra": 1}):
        with pytest.raises(StyleError):
            resolve_style_data({"schema_version": 1, "grade": bad})


def test_voice_and_ducking_defaults() -> None:
    data = resolve_style_data({"schema_version": 1})
    assert data["voice"] == {"preset": "none", "repair_rumble": False,
                             "repair_proximity": "off", "repair_room": False,
                             "repair_harshness": False}
    assert data["music"]["duck_mode"] == "level"
    assert data["music"]["duck_depth_db"] == -12
    assert data["music"]["duck_merge_gap_ms"] == 250
    assert data["music"]["duck_attack_ms"] == 10
    assert data["music"]["duck_release_ms"] == 320


@pytest.mark.parametrize("preset", ["none", "clean", "broadcast", "warm"])
@pytest.mark.parametrize("proximity", ["gentle", "strong"])
def test_voice_options_resolve(preset: str, proximity: str) -> None:
    voice = {"preset": preset, "repair_rumble": True,
             "repair_proximity": proximity, "repair_room": True, "repair_harshness": True}
    assert resolve_style_data({"schema_version": 1, "voice": voice})["voice"] == voice


@pytest.mark.parametrize(("mode", "depth", "gap"), [
    ("level", -40, 0), ("words", 0, 2000), ("words", -12.5, 250),
])
def test_music_ducking_options_resolve(mode: str, depth: float, gap: int) -> None:
    music = {"duck_mode": mode, "duck_depth_db": depth, "duck_merge_gap_ms": gap}
    data = resolve_style_data({"schema_version": 1, "music": music})
    assert all(data["music"][key] == value for key, value in music.items())


@pytest.mark.parametrize(("section", "key", "value"), [
    ("voice", "preset", "unknown"),
    ("voice", "repair", ["rumble"]),
    ("voice", "repair_rumble", 1),
    ("voice", "repair_proximity", True),
    ("voice", "repair_proximity", "medium"),
    ("voice", "repair_proximity", None),
    ("voice", "repair_room", "true"),
    ("voice", "repair_harshness", None),
    ("music", "duck_mode", "unknown"),
    ("music", "duck_depth_db", -40.01),
    ("music", "duck_depth_db", 0.01),
    ("music", "duck_depth_db", True),
    ("music", "duck_depth_db", float("nan")),
    ("music", "duck_merge_gap_ms", -1),
    ("music", "duck_merge_gap_ms", 2001),
    ("music", "duck_merge_gap_ms", 1.5),
    ("music", "duck_merge_gap_ms", False),
])
def test_invalid_voice_and_ducking_fields(section: str, key: str, value: object) -> None:
    with pytest.raises(StyleError) as caught:
        validate_style_data({"schema_version": 1, section: {key: value}})
    assert caught.value.parameter == f"style.{section}.{key}"


def test_explicit_audio_defaults_have_same_style_hash(tmp_path: Path) -> None:
    omitted = resolve(_file(tmp_path / "omitted.json", {"schema_version": 1}))
    explicit = resolve(_file(tmp_path / "explicit.json", {
        "schema_version": 1, "voice": {"preset": "none", "repair_rumble": False,
                                      "repair_proximity": "off", "repair_room": False,
                                      "repair_harshness": False},
        "music": {"duck_mode": "level", "duck_depth_db": -12, "duck_merge_gap_ms": 250},
    }))
    assert explicit.sha256() == omitted.sha256()


@pytest.mark.parametrize(
    ("data", "field", "detail"),
    [
        ({"schema_version": 1, "unknown": 1}, "style.unknown", "unknown field"),
        (
            {"schema_version": 1, "captions": {"extra": 1}},
            "style.captions.extra",
            "unknown field",
        ),
        (
            {"schema_version": 1, "captions": {"size_px": "76"}},
            "style.captions.size_px",
            "expected integer",
        ),
        (
            {"schema_version": 1, "captions": {"size_px": True}},
            "style.captions.size_px",
            "expected integer",
        ),
        (
            {"schema_version": 1, "captions": {"size_px": 181}},
            "style.captions.size_px",
            "must be between 32 and 180",
        ),
        (
            {"schema_version": 1, "captions": {"width_scale": 0.59}},
            "style.captions.width_scale",
            "must be between 0.6 and 1",
        ),
        (
            {"schema_version": 1, "captions": {"width_scale": 1.01}},
            "style.captions.width_scale",
            "must be between 0.6 and 1",
        ),
        (
            {"schema_version": 1, "captions": {"width_scale": "0.8"}},
            "style.captions.width_scale",
            "expected number",
        ),
        (
            {"schema_version": 1, "captions": {"width_scale": True}},
            "style.captions.width_scale",
            "expected number",
        ),
        (
            {"schema_version": 1, "captions": {"text_color": "red"}},
            "style.captions.text_color",
            "expected a color like #1A2B3C",
        ),
        (
            {"schema_version": 1, "captions": {"text_color": "#1A2B3C\n"}},
            "style.captions.text_color",
            "expected a color",
        ),
        (
            {"schema_version": 1, "captions": {"position": "top"}},
            "style.captions.position",
            "choose one of",
        ),
        (
            {"schema_version": 1, "captions": {"highlight_style": "glow"}},
            "style.captions.highlight_style",
            "choose one of",
        ),
        ({"schema_version": 2}, "style.schema_version", "must be 1"),
        ({}, "style.schema_version", "missing required field"),
        ({"schema_version": 1, "name": ""}, "style.name", "length is outside"),
        ({"schema_version": 1, "name": "a" * 81}, "style.name", "length is outside"),
        ({"schema_version": 1, "captions": []}, "style.captions", "expected object"),
        (
            {"schema_version": 1, "captions": {"italic_stress": 1}},
            "style.captions.italic_stress",
            "expected boolean",
        ),
        ({"schema_version": 1, "music": {"level": True}}, "style.music.level", "expected number"),
    ],
)
def test_invalid_fields_have_structured_paths(data: dict, field: str, detail: str) -> None:
    with pytest.raises(StyleError) as caught:
        validate_style_data(data)
    error = caught.value
    assert isinstance(error, (ValidationError, YapError, ValueError))
    assert error.parameter == field
    assert field in str(error)
    assert field in error.to_dict()["message"]
    assert detail in error.detail
    assert field in error.to_dict()["suggested_action"]


@pytest.mark.parametrize("width_scale", [0.6, 0.8, 1])
def test_caption_width_scale_resolves_unchanged(width_scale: float) -> None:
    data = resolve_style_data({"schema_version": 1, "captions": {"width_scale": width_scale}})
    assert data["captions"]["width_scale"] == width_scale


@pytest.mark.parametrize(
    "highlight", ["pill", "gradient", "editorial", "slam", "weight_shift", "clip_wipe"]
)
def test_new_caption_highlights_resolve_unchanged(highlight: str) -> None:
    data = resolve_style_data({"schema_version": 1, "captions": {"highlight_style": highlight}})
    assert data["captions"]["highlight_style"] == highlight


def test_caption_width_scale_explicit_default_has_same_hash(tmp_path: Path) -> None:
    omitted = resolve(_file(tmp_path / "omitted.json", {"schema_version": 1}))
    explicit = resolve(
        _file(tmp_path / "explicit.json", {"schema_version": 1, "captions": {"width_scale": 1}})
    )
    assert explicit.sha256() == omitted.sha256()


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_numbers_fail(number: float) -> None:
    with pytest.raises(StyleError, match=r"style.music.level: expected a finite number"):
        resolve_style_data({"schema_version": 1, "music": {"level": number}})


@pytest.mark.parametrize(
    ("keep", "number", "detail"),
    [("minimum", -1, "at least 0"), ("maximum", 2, "at most 1")],
)
def test_single_numeric_bounds(
    keep: str, number: int, detail: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    rule = schema["properties"]["music"]["properties"]["level"]
    rule.pop("maximum" if keep == "minimum" else "minimum")
    monkeypatch.setattr(loader, "_schema", lambda: schema)
    with pytest.raises(StyleError, match=detail) as caught:
        validate_style_data({"schema_version": 1, "music": {"level": number}})
    assert caught.value.parameter == "style.music.level"
    validate_style_data({"schema_version": 1, "music": {"level": 0.5}})


@pytest.mark.parametrize("content", ["[]", "null", "{", "\udcff"])
def test_unreadable_json_is_a_style_error(tmp_path: Path, content: str) -> None:
    source = tmp_path / "bad.json"
    source.write_bytes(content.encode("utf-8", errors="surrogateescape"))
    with pytest.raises(StyleError) as caught:
        load_style(source)
    assert caught.value.parameter == str(source)


def test_missing_file_is_a_style_error(tmp_path: Path) -> None:
    source = tmp_path / "missing.json"
    with pytest.raises(StyleError, match="cannot read JSON") as caught:
        load_style(source)
    assert caught.value.parameter == str(source)


def test_resolution_copies_data_and_defaults() -> None:
    data = {"schema_version": 1, "captions": {"font": "Alata"}}
    before = copy.deepcopy(data)
    first = resolve_style_data(data)
    first["captions"]["font"] = "Inter"
    first["music"]["level"] = 0.5
    assert data == before
    second = resolve_style_data(data)
    assert second["captions"]["font"] == "Alata"
    assert second["music"]["level"] == 0.06


def test_schema_is_read_once(monkeypatch: pytest.MonkeyPatch) -> None:
    original = loader._read
    reads = []

    def record(path: Path) -> dict:
        reads.append(path)
        return original(path)

    loader._schema.cache_clear()
    monkeypatch.setattr(loader, "_read", record)
    try:
        validate_style_data({"schema_version": 1})
        resolve_style_data({"schema_version": 1})
        load_style(PRESETS_DIR / "electric.json")
        assert reads.count(SCHEMA_PATH) == 1
    finally:
        loader._schema.cache_clear()


def test_engine_presets_are_exactly_allowlisted() -> None:
    expected = {"clean_minimal", "electric", "warm_editorial", "editorial_serif", "one_word",
                "cobalt", "mono", "headline", "candy", "marker",
                "hf_bold", "hf_editorial", "hf_neutral"}
    assert set(SHIPPED_PRESETS) == expected
    assert {path.stem for path in PRESETS_DIR.glob("*.json")} == expected


@pytest.mark.parametrize("preset", SHIPPED_PRESETS)
def test_shipped_presets_validate(preset: str) -> None:
    data = json.loads((PRESETS_DIR / f"{preset}.json").read_text(encoding="utf-8"))
    validate_style_data(data)
    result = resolve(f"preset:{preset}")
    assert result.source == f"preset:{preset}"
    assert result.name == data["name"]
    assert result.data["captions"]["width_scale"] == 1
    assert result.data["graphics"]["face_clearance_px"] == 24
    assert face_clearance_px(result.data) == 24
    assert caption_gap_px(result.data) == 100


def test_default_face_clearance_matches_schema() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    rule = schema["properties"]["graphics"]["properties"]["face_clearance_px"]
    assert DEFAULT_FACE_CLEARANCE_PX == rule["default"] == 24
    style = resolve_style_data({"schema_version": 1, "graphics": {"face_clearance_px": 50}})
    assert face_clearance_px(style) == 50


def test_default_caption_gap_matches_schema() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    rule = schema["properties"]["graphics"]["properties"]["caption_gap_px"]
    assert DEFAULT_CAPTION_GAP_PX == rule["default"] == 100
    style = resolve_style_data({"schema_version": 1, "graphics": {"caption_gap_px": 60}})
    assert caption_gap_px(style) == 60


@pytest.mark.parametrize("value", [-1, 401])
def test_caption_gap_invalid_style_values(value: int, tmp_path: Path) -> None:
    source = _file(
        tmp_path / "bad-caption-gap.json",
        {"schema_version": 1, "graphics": {"caption_gap_px": value}},
    )
    with pytest.raises(StyleError) as caught:
        load_style(source)
    assert caught.value.parameter == "style.graphics.caption_gap_px"
    assert "must be between 0 and 400" in caught.value.detail


@pytest.mark.parametrize(
    ("value", "detail"),
    [
        (-1, "must be between 0 and 200"),
        (201, "must be between 0 and 200"),
        (1.5, "expected integer"),
        (True, "expected integer"),
    ],
)
def test_face_clearance_invalid_style_values(value: object, detail: str, tmp_path: Path) -> None:
    source = _file(
        tmp_path / "bad-clearance.json",
        {"schema_version": 1, "graphics": {"face_clearance_px": value}},
    )
    with pytest.raises(StyleError) as caught:
        load_style(source)
    assert caught.value.parameter == "style.graphics.face_clearance_px"
    assert "style.graphics.face_clearance_px" in str(caught.value)
    assert detail in caught.value.detail


def test_remaining_proposed_presets_validate() -> None:
    for source in sorted(PROPOSED.glob("*.json")):
        if source.name != "order.json":
            validate_style_data(json.loads(source.read_text(encoding="utf-8")))


def test_unknown_preset_lists_shipped_ids() -> None:
    with pytest.raises(StyleError) as caught:
        resolve("preset:unknown")
    assert caught.value.parameter == "style"
    for preset in SHIPPED_PRESETS:
        assert preset in str(caught.value)


def test_default_falls_back_to_clean_minimal_without_creating_config() -> None:
    assert get_default() is None
    assert list_entries() == []
    assert resolve(None) == resolve("preset:clean_minimal")
    assert not config_root().exists()


def test_import_original_json_and_default_selection(tmp_path: Path) -> None:
    data = {"schema_version": 1, "name": "My style", "captions": {"size_px": 90}}
    source = _file(tmp_path / "My Style.json", data)
    entry = add(source, make_default=True)
    assert entry.id == "my_style"
    assert entry.name == "My style"
    assert entry.path == library_dir() / "my_style.json"
    assert json.loads(entry.path.read_text(encoding="utf-8")) == data
    assert entry.is_default
    assert get_default() == "my_style"
    assert resolve(None) == resolve("my_style")
    assert resolve(Path("my_style")) == resolve("my_style")
    assert resolve(None).data == resolve_style_data(data)
    assert resolve(None).source == "library:my_style"
    assert resolve("preset:clean_minimal").name == "Clean Minimal"
    assert resolve(source).source == str(source)
    assert list_entries() == [entry]
    config = json.loads((config_root() / "config.json").read_text(encoding="utf-8"))
    assert config == {"schema": "yap-studio/config@1", "default_style": "my_style"}


def test_import_without_default_keeps_clean_minimal(tmp_path: Path) -> None:
    entry = add(_file(tmp_path / "mine.json", {"schema_version": 1, "name": "Mine"}))
    assert not entry.is_default
    assert resolve(None) == resolve("preset:clean_minimal")


def test_identical_import_is_a_noop_and_can_select_default(tmp_path: Path) -> None:
    source = _file(tmp_path / "mine.json", {"schema_version": 1})
    entry = add(source)
    before = entry.path.stat()
    assert add(source) == entry
    after = entry.path.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert add(source, make_default=True).is_default


def test_different_import_refuses_then_replaces_atomically(tmp_path: Path) -> None:
    source = _file(tmp_path / "mine.json", {"schema_version": 1, "name": "First"})
    entry = add(source, make_default=True)
    before = entry.path.read_bytes()
    changed = {"schema_version": 1, "name": "Second"}
    _file(source, changed)
    with pytest.raises(StyleRefusal) as caught:
        add(source)
    assert caught.value.code == "style_exists"
    assert caught.value.suggested_action
    assert entry.path.read_bytes() == before
    assert resolve(None).name == "First"
    updated = add(source, replace=True)
    assert updated.is_default
    assert json.loads(updated.path.read_text(encoding="utf-8")) == changed
    assert resolve(None).name == "Second"
    assert set(library_dir().iterdir()) == {updated.path, library_dir() / ".save.lock"}


def test_failed_replacement_keeps_original_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _file(tmp_path / "mine.json", {"schema_version": 1, "name": "First"})
    entry = add(source, make_default=True)
    before = entry.path.read_bytes()
    _file(source, {"schema_version": 1, "name": "Second"})
    original_replace = os.replace

    def fail_import(staged: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        if Path(target) == entry.path:
            raise OSError("replacement failed")
        original_replace(staged, target)

    monkeypatch.setattr(library.os, "replace", fail_import)
    with pytest.raises(StyleError, match="cannot save style"):
        add(source, replace=True)
    assert entry.path.read_bytes() == before
    assert resolve(None).name == "First"
    assert set(library_dir().iterdir()) == {entry.path, library_dir() / ".save.lock"}


def test_library_ids_are_slugs_with_a_length_limit(tmp_path: Path) -> None:
    source = _file(tmp_path / ("A" * 70 + " !.json"), {"schema_version": 1})
    entry = add(source)
    assert entry.id == "a" * 64
    assert resolve(entry.id).source == f"library:{entry.id}"
    invalid = _file(tmp_path / "!!!.json", {"schema_version": 1})
    with pytest.raises(StyleError, match=r"style.id"):
        add(invalid)
    assert list_entries() == [entry]


@pytest.mark.parametrize("populated", [False, True])
def test_invalid_import_leaves_library_and_default_unchanged(
    tmp_path: Path, populated: bool
) -> None:
    if populated:
        add(_file(tmp_path / "mine.json", {"schema_version": 1}), make_default=True)
    before = {path.name: path.read_bytes() for path in library_dir().glob("*.json")}
    default = get_default()
    invalid = _file(tmp_path / "invalid.json", {"schema_version": 1, "music": {"level": 2}})
    with pytest.raises(StyleError, match=r"style.music.level"):
        add(invalid, make_default=True, replace=True)
    assert {path.name: path.read_bytes() for path in library_dir().glob("*.json")} == before
    assert get_default() == default
    if not populated:
        assert not config_root().exists()


def test_presets_can_be_default_and_entries_mark_only_library_defaults(tmp_path: Path) -> None:
    add(_file(tmp_path / "z.json", {"schema_version": 1}), make_default=True)
    add(_file(tmp_path / "a.json", {"schema_version": 1}))
    assert [entry.id for entry in list_entries()] == ["a", "z"]
    assert [entry.is_default for entry in list_entries()] == [False, True]
    set_default("preset:electric")
    assert resolve(None) == resolve("preset:electric")
    assert not any(entry.is_default for entry in list_entries())


def test_default_rejects_file_paths_and_unknown_ids_without_writing(tmp_path: Path) -> None:
    source = _file(tmp_path / "source.json", {"schema_version": 1})
    set_default("preset:electric")
    before = (config_root() / "config.json").read_bytes()
    for spec in (str(source), "missing", "preset:missing"):
        with pytest.raises(StyleError):
            set_default(spec)
        assert (config_root() / "config.json").read_bytes() == before


@pytest.mark.parametrize("field", ["bleep_words_add", "bleep_words_remove"])
def test_bleep_list_items_require_two_characters(field: str, tmp_path: Path) -> None:
    source = _file(tmp_path / "bleep.json", {"schema_version": 1, "music": {field: ["x"]}})
    with pytest.raises(StyleError, match=field):
        resolve(source)
    _file(source, {"schema_version": 1, "music": {field: ["xx"]}})
    assert resolve(source).data["music"][field] == ["xx"]


def test_bleep_defaults_preserve_resolved_bytes_and_receipt_identity() -> None:
    data = resolve_style_data({"schema_version": 1})
    style = loader.ResolvedStyle(data["name"], "test", data)
    legacy = copy.deepcopy(data)
    for key in ("bleep_swears", "bleep_words_add", "bleep_words_remove"):
        legacy["music"].pop(key)
    old_text = json.dumps({"schema": RESOLVED_SCHEMA, **legacy}, indent=2,
                          sort_keys=False, ensure_ascii=False) + "\n"
    assert style.text() == old_text
    assert style.sha256() == hashlib.sha256(old_text.encode()).hexdigest()
    data["music"]["bleep_words_add"] = ["crap"]
    assert style.to_json()["music"]["bleep_words_add"] == ["crap"]
    data["music"]["bleep_swears"] = True
    assert style.to_json()["music"]["bleep_swears"] is True


def test_resolved_output_roundtrip_canonical_hash_and_repeated_write(tmp_path: Path) -> None:
    source = _file(tmp_path / "source.json", {"schema_version": 1, "name": "Café"})
    style = resolve(source)
    out = tmp_path / "resolved-style.json"
    assert write_resolved(style, out) == out
    assert write_resolved(style, out) == out
    assert out.read_text(encoding="utf-8") == style.text()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert next(iter(payload)) == "schema"
    assert payload["schema"] == RESOLVED_SCHEMA
    assert style.text() == json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    assert "Café" in style.text()
    assert style.sha256() == hashlib.sha256(out.read_bytes()).hexdigest()
    assert load_style(out) == style.data
    assert "schema" not in style.data
    payload["captions"]["font"] = "Alata"
    assert style.data["captions"]["font"] == "Inter"
    assert not list(tmp_path.glob(".yap_tmp_*"))


def test_resolved_marker_does_not_bypass_validation() -> None:
    with pytest.raises(StyleError, match=r"style.captions.size_px"):
        resolve_style_data(
            {"schema": RESOLVED_SCHEMA, "schema_version": 1, "captions": {"size_px": 1}}
        )
    with pytest.raises(StyleError, match=r"style.schema"):
        validate_style_data({"schema": "yap-studio/other@1", "schema_version": 1})


def test_resolved_output_refuses_non_engine_json(tmp_path: Path) -> None:
    out = _file(tmp_path / "other.json", {"name": "existing"})
    before = out.read_bytes()
    with pytest.raises(OutputPathError) as caught:
        write_resolved(resolve("preset:clean_minimal"), out)
    assert caught.value.code == "unsafe_path"
    assert out.read_bytes() == before


def test_library_replacement_refuses_symlinks(tmp_path: Path) -> None:
    source = _file(tmp_path / "mine.json", {"schema_version": 1, "name": "New"})
    other = _file(tmp_path / "other.json", {"schema_version": 1, "name": "Other"})
    library_dir().mkdir(parents=True)
    (library_dir() / "mine.json").symlink_to(other)
    before = other.read_bytes()
    with pytest.raises(OutputPathError, match="symlink"):
        add(source, replace=True)
    assert other.read_bytes() == before


def test_create_refuses_an_existing_style_file(tmp_path: Path) -> None:
    target = tmp_path / "look.json"
    assert library.create("Blue", out=str(target)) == target
    before = target.read_bytes()
    with pytest.raises(StyleError, match="already exists"):
        library.create("Red", preset="electric", out=str(target))
    assert target.read_bytes() == before
    assert load_style(target)["name"] == "Blue"
    assert [path.name for path in tmp_path.iterdir() if path.name != "config"] == ["look.json"]


def test_create_never_replaces_a_style_published_after_its_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "look.json"
    rival = json.dumps({"schema_version": 1, "name": "Blue"}) + "\n"
    original_validate = library.paths.validate_output

    def publish_rival(path: str | os.PathLike[str], *args: object, **kwargs: object) -> Path:
        checked = original_validate(path, *args, **kwargs)
        if Path(path) == target and not target.exists():
            target.write_text(rival, encoding="utf-8")
        return checked

    monkeypatch.setattr(library.paths, "validate_output", publish_rival)
    with pytest.raises(StyleError, match="already exists"):
        library.create("Red", preset="electric", out=str(target))
    assert target.read_text(encoding="utf-8") == rival
    assert [path.name for path in tmp_path.iterdir() if path.name != "config"] == ["look.json"]


@pytest.mark.parametrize("hard_links", [True, False])
def test_create_never_replaces_a_style_published_at_the_last_moment(
    hard_links: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "look.json"
    rival = json.dumps({"schema_version": 1, "name": "Blue"}) + "\n"
    original_link, original_replace = os.link, os.replace

    def publish_rival(destination: str | os.PathLike[str]) -> None:
        if os.path.realpath(destination) == os.path.realpath(target) and not target.exists():
            target.write_text(rival, encoding="utf-8")

    def link(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        publish_rival(destination)
        if not hard_links:
            raise OSError(errno.ENOTSUP, "hard links are not supported")
        original_link(source, destination)

    def replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        publish_rival(destination)
        original_replace(source, destination)

    # Another writer publishes just before whichever call moves the new file into place.
    # Without hard links, the exclusive-create fallback must refuse the rival's file too.
    monkeypatch.setattr(os, "link", link)
    monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(StyleError, match="already exists"):
        library.create("Red", preset="electric", out=str(target))
    assert target.read_text(encoding="utf-8") == rival
    assert [path.name for path in tmp_path.iterdir() if path.name != "config"] == ["look.json"]


def _no_link(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
    raise OSError(errno.ENOTSUP, "hard links are not supported")


def test_create_without_hard_links_renames_the_whole_file_into_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(library.os, "link", _no_link)
    target = tmp_path / "look.json"
    assert library.create("Blue", out=str(target)) == target
    assert load_style(target)["name"] == "Blue"
    assert [path.name for path in tmp_path.iterdir() if path.name != "config"] == ["look.json"]


def test_no_replace_rename_keeps_an_existing_target(tmp_path: Path) -> None:
    source = _file(tmp_path / "staged.json", {"schema_version": 1, "name": "Red"})
    target = _file(tmp_path / "look.json", {"schema_version": 1, "name": "Blue"})
    with pytest.raises(FileExistsError):
        library._rename_no_replace(source, target)
    assert load_style(target)["name"] == "Blue"
    assert load_style(source)["name"] == "Red"


def test_create_refuses_a_drive_that_cannot_publish_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_rename(source: Path, target: Path) -> None:
        raise OSError(errno.ENOTSUP, "Operation not supported")

    monkeypatch.setattr(library.os, "link", _no_link)
    monkeypatch.setattr(library, "_rename_no_replace", no_rename)
    target = tmp_path / "look.json"
    with pytest.raises(StyleError, match="cannot save style"):
        library.create("Blue", out=str(target))
    assert [path.name for path in tmp_path.iterdir() if path.name != "config"] == []


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_platform_config_locations(
    platform: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("YAP_STUDIO_CONFIG")
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(library.sys, "platform", platform)
    expected = (
        tmp_path / "Library" / "Application Support" / "yap-studio"
        if platform == "darwin"
        else tmp_path / ".local" / "share" / "yap-studio"
    )
    assert config_root() == expected
    assert library_dir() == expected / "styles"
    assert not expected.exists()


def test_config_overrides_expand_to_absolute_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("YAP_STUDIO_CONFIG", "~/settings")
    assert config_root() == tmp_path / "settings"
    monkeypatch.setenv("YAP_STUDIO_CONFIG", "relative-settings")
    assert config_root() == tmp_path / "relative-settings"
    monkeypatch.delenv("YAP_STUDIO_CONFIG")
    monkeypatch.setattr(library.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", "~/data")
    assert config_root() == tmp_path / "data" / "yap-studio"


def test_tracked_text_has_no_private_palette_codes() -> None:
    raw_codes = os.environ.get("YAP_STUDIO_HOUSE_HEX")
    if not raw_codes:
        pytest.skip("private palette codes were not supplied")
    codes = [code.strip().casefold() for code in raw_codes.split(",") if code.strip()]
    if not codes or any(re.fullmatch(r"#[0-9a-f]{6}", code) is None for code in codes):
        pytest.fail("private palette check requires comma-separated hex colors")
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30,
        check=True,
    ).stdout.split(b"\0")
    offenders = []
    for raw_path in tracked:
        if not raw_path:
            continue
        relative = raw_path.decode(errors="surrogateescape")
        path = ROOT / relative
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            continue
        for number, line in enumerate(data.decode(errors="replace").splitlines(), 1):
            if any(code in line.casefold() for code in codes):
                offenders.append(f"{relative}:{number}")
    if offenders:
        pytest.fail("private palette references found:\n" + "\n".join(offenders))


def test_moves_defaults_and_presets() -> None:
    defaults = {"zoom_mode": "none", "punch_scale": 1.5,
                "transitions": ["whip_pan", "cinematic_zoom", "light_leak"],
                "jump_cuts": "hard", "opening_push": False}
    assert resolve_style_data({"schema_version": 1})["moves"] == defaults
    for preset in SHIPPED_PRESETS:
        assert resolve(f"preset:{preset}").data["moves"] == defaults


@pytest.mark.parametrize("mode", ["none", "punches", "punches_and_push"])
def test_moves_options(mode: str) -> None:
    moves = {"zoom_mode": mode, "punch_scale": 2, "transitions": ["light_leak", "whip_pan"],
             "jump_cuts": "hard", "opening_push": True}
    assert resolve_style_data({"schema_version": 1, "moves": moves})["moves"] == moves


@pytest.mark.parametrize("looks", [
    [], ["whip_pan"], ["light_leak", "cinematic_zoom"],
    ["whip_pan", "cinematic_zoom", "light_leak"],
])
def test_moves_transitions_options(looks: list[str]) -> None:
    data = resolve_style_data({"schema_version": 1, "moves": {"transitions": looks}})
    assert data["moves"]["transitions"] == looks


def test_moves_transition_looks_match_engine_and_timeline_schema() -> None:
    from yap_studio.project.model import _ELEMENT_FIELDS
    from yap_studio.stages.transitions import LOOKS

    schema = json.loads(SCHEMA_PATH.read_bytes())
    looks = schema["properties"]["moves"]["properties"]["transitions"]["items"]["enum"]
    assert looks == _ELEMENT_FIELDS["transition"]["look"]["enum"]
    assert set(looks) == LOOKS


@pytest.mark.parametrize("fields,detail", [
    ({"punch_scale": 1.75}, "choose one of"),
    ({"unknown": 1}, "unknown field"),
    ({"transitions": ["dissolve"]}, "choose one of"),
    ({"transitions": ["whip_pan", "whip_pan"]}, "choose each transition look only once"),
    ({"transitions": "whip_pan"}, "expected array"),
])
def test_moves_refuses_invalid_fields(fields: dict, detail: str) -> None:
    with pytest.raises(StyleError, match=detail):
        validate_style_data({"schema_version": 1, "moves": fields})


@pytest.mark.parametrize("mode", ["hard", "punch", "smooth", "step_in"])
def test_jump_cuts_options(mode: str) -> None:
    assert resolve_style_data({"schema_version": 1, "moves": {"jump_cuts": mode}})[
        "moves"]["jump_cuts"] == mode


def test_jump_cuts_refuses_unknown() -> None:
    with pytest.raises(StyleError):
        resolve_style_data({"schema_version": 1, "moves": {"jump_cuts": "unknown"}})
