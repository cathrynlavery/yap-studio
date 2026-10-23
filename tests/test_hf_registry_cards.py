"""Registry cards share slot rules, geometry, placement and receipt identity."""

from __future__ import annotations

import copy
import json
import re
import shutil
from dataclasses import replace
from html.parser import HTMLParser
from types import SimpleNamespace

import numpy as np
import pytest
import synth
from test_hf_registry import CORE
from test_hf_registry import synthetic as synthetic
from test_planner_anchors import synthetic_words
from test_planner_cards import requests, synthetic_faces
from test_stage_captions import layer_project as layer_project
from test_stage_captions import set_elements
from test_stage_graphics import graphic, turn_on_card_kind

from yap_studio import cards
from yap_studio.errors import DependencyError, ValidationError, YapError
from yap_studio.faces.model import Box
from yap_studio.hf_registry import catalog
from yap_studio.planner import cards as planner
from yap_studio.project.status import ProjectContext
from yap_studio.render import hyperframes
from yap_studio.stages import assemble as assemble_stage
from yap_studio.stages.graphics import graphics_snapshot
from yap_studio.style import cards_on, resolve
from yap_studio.style.loader import StyleError, check_card_kinds_on

KIND = "hf:sample"
SOURCE = (
    '<!doctype html><html data-composition-id="sample" '
    "data-composition-variables='[{\"id\":\"text\",\"type\":\"string\"},"
    "{\"id\":\"progress\",\"type\":\"number\"}]'><head>"
    f'<script src="{CORE}"></script></head><body><template>'
    '<div id="root" data-composition-id="sample"><style>'
    '#root{position:absolute;inset:0;font-family:Inter,sans-serif}'
    '.mark{position:absolute;left:400px;top:300px;width:200px;height:100px;'
    'background:var(--brand,red)}</style><div class="mark"></div>'
    '<script>window.__timelines=window.__timelines||{};'
    'window.__timelines.sample=gsap.timeline({paused:true});</script>'
    '</div></template></body></html>'
)


def duration_source(*, template: bool, duration: int = 1) -> str:
    """A whole-item clip plus unrelated timings and attribute-like script text."""
    html_attributes = (f"data-composition-id='sample' data-composition-duration = '{duration}'"
                       if template else "")
    return (
        f"<!doctype html><html {html_attributes}><head><meta charset='utf-8'>"
        f'<script src="{CORE}"></script></head><body>' + ("<template>" if template else "") +
        f'<div id="root" data-composition-id="sample" DATA-DURATION = "{duration}" '
        f"data-note='data-duration=\"{duration}\"' "
        "data-composition-variables='[{\"id\":\"text\",\"type\":\"string\"},"
        "{\"id\":\"progress\",\"type\":\"number\"}]'>"
        '<style>.clip{position:absolute;inset:0}.mark{position:absolute;'
        'left:40px;top:40px;width:200px;height:100px;background:red}</style>'
        '<div id="sample-clip" class="clip" data-start="0" '
        f'data-duration={duration} data-whole-clip data-track-index="0">'
        '<div class="mark"></div><i data-duration="0.25"></i>'
        f'<i data-duration="{duration}"></i></div>'
        f'<div class="clip" data-start="0.25" data-duration="{duration}"></div>'
        '<script>(function(){var root=document.getElementById("root");'
        f'var authored=\'<div data-duration="{duration}"></div>\';'
        'var durationValue=Number(root.dataset.duration);'
        f'var D=Number.isFinite(durationValue)&&durationValue>0?durationValue:{duration};'
        'window.__timelines=window.__timelines||{};'
        'window.__timelines.sample=gsap.timeline({paused:true})'
        '.to(root.querySelector(".mark"),{opacity:0,duration:0.1},D-0.1);'
        '})();</script></div>' + ("</template>" if template else "") + '</body></html>'
    )


def card_fields():
    return {
        "status": "on",
        "kind": {"category": "Words", "role": "titles", "label": "Sample", "use": "words"},
        "mount": "fill", "readability": "none", "min_panel": [600, 200],
        "slots": {
            "text": {"type": "string", "required": True, "min_length": 1, "max_length": 18},
            "percent": {"type": "integer", "required": False, "minimum": 0,
                        "maximum": 100, "default": 37},
        },
        "variables": {"text": "{text}", "progress": "{percent}"},
        "preview": {"text": "Ship it now", "percent": 37},
        "tokens": {"--brand": "accent", "--accent": "accent", "--fg": "ink",
                   "--surface": "paper"}, "min_s": 1.5,
    }


@pytest.fixture
def registry_card(synthetic):
    registry_catalog, install, _, renderer, add, _ = synthetic
    add(source=SOURCE)
    registry_catalog.items()["sample"].update(card_fields())
    install.install_items(install=renderer)
    return synthetic


@pytest.fixture
def card_style():
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_kinds_on"] = [KIND]
    style.data["graphics"].update(card_width_px=960, accent_color="#123456")
    style.data["hook"]["font"] = "Gelasio"
    style.data["internal"].update(key_text_top_px=180, key_text_bottom_px=540)
    return style


def test_catalog_kind_fields():
    from yap_studio.hf_registry import cards as registry

    assert registry.kinds() == ("hf:char-slam-explode", "hf:conic-progress-ring")
    assert "gooey-split" not in catalog.items()
    assert registry.is_kind("hf:sample") and not registry.is_kind("stat")
    assert registry.item_name("hf:char-slam-explode") == "char-slam-explode"
    with pytest.raises(ValidationError):
        registry.item_name("stat")
    metadata = registry.template_metadata("hf:char-slam-explode")
    assert metadata["id"] == "hf:char-slam-explode" and metadata["version"] == 1
    assert metadata["derived_from"][0]["item"] == "char-slam-explode"
    assert metadata["slots"] == catalog.item("char-slam-explode")["slots"]
    assert metadata["box"]["width"] == "graphics.card_width_px + 80"
    style = resolve("preset:clean_minimal")
    assert not any(registry.is_kind(kind) for kind in cards_on(style))
    style.data["graphics"]["card_kinds_on"] = list(reversed(registry.kinds()))
    assert cards_on(style)[-2:] == registry.kinds()


def test_catalog_preview_slots():
    from yap_studio.hf_registry import cards as registry

    expected = {"char-slam-explode": {"text": "Ship it now"},
                "conic-progress-ring": {"percent": 72}}
    for entry in catalog.items().values():
        if entry["status"] == "on":
            cards._slots(entry["preview"], entry["slots"])
    for name, slots in expected.items():
        assert registry.preview_slots(f"hf:{name}") == slots
        returned = registry.preview_slots(f"hf:{name}")
        returned.clear()
        assert registry.preview_slots(f"hf:{name}") == slots


@pytest.mark.parametrize(("status", "readability", "valid"), [
    ("on", None, False), ("on", "shadow", False), ("on", [], False),
    ("on", "halo", True), ("on", "none", True), ("pilot", None, True),
])
def test_catalog_readability(tmp_path, monkeypatch, status, readability, valid):
    data = copy.deepcopy(catalog.load_catalog())
    entry = data["items"]["char-slam-explode"]
    entry["status"] = status
    entry.pop("readability", None)
    if readability is not None:
        entry["readability"] = readability
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(data))
    monkeypatch.setattr(catalog, "CATALOG_PATH", source)
    catalog.load_catalog.cache_clear()
    try:
        if valid:
            assert catalog.load_catalog()["items"]["char-slam-explode"] == entry
        else:
            with pytest.raises(ValidationError, match="readability"):
                catalog.load_catalog()
    finally:
        catalog.load_catalog.cache_clear()


def test_catalog_accepts_accent_tint(tmp_path, monkeypatch):
    data = copy.deepcopy(catalog.load_catalog())
    data["items"]["conic-progress-ring"]["tokens"]["--surface"] = "accent_tint"
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(data))
    monkeypatch.setattr(catalog, "CATALOG_PATH", source)
    catalog.load_catalog.cache_clear()
    try:
        assert catalog.load_catalog() == data
    finally:
        catalog.load_catalog.cache_clear()


def test_pilot_palette_and_readability():
    common = {"--brand": "accent", "--accent": "accent", "--fg": "paper"}
    assert catalog.item("char-slam-explode")["tokens"] == {
        **common, "--accent-2": "accent", "--cse-accent": "accent",
    }
    assert catalog.item("conic-progress-ring")["tokens"] == {**common, "--surface": "accent_tint"}
    assert all(entry["readability"] == "halo" for entry in catalog.items().values()
               if entry["status"] == "on")


@pytest.mark.parametrize("name", ["char-slam-explode", "conic-progress-ring"])
def test_pilot_plan_has_clear_backdrop(synthetic, card_style, name):
    registry_catalog, registry_install, _, renderer, add, _ = synthetic
    pilot = json.loads(catalog.CATALOG_PATH.read_text())["items"][name]
    declarations = json.dumps([{"id": key, "type": "string"} for key in pilot["variables"]])
    source = re.sub(r"data-composition-variables='[^']*'",
                    f"data-composition-variables='{declarations}'", SOURCE.replace("sample", name))
    add(name=name, source=source)
    registry_catalog.item(name).update({key: value for key, value in pilot.items()
                                       if key not in {"type", "upstream_path", "main", "files"}})
    registry_install.install_items(install=renderer)
    plan = cards.card_layer_plan(f"hf:{name}", pilot["preview"], card_style)
    assert plan.registry["tokens"]["--bg"] == "transparent"
    assert "--radius" not in plan.registry["tokens"]


def test_catalog_backdrop_mapping_overrides_host_default(registry_card, card_style, tmp_path):
    from yap_studio.hf_registry import cards as registry

    plan = cards.card_layer_plan(KIND, {"text": "Words"}, card_style)
    assert plan.registry["tokens"]["--bg"] == "transparent"
    registry_card[0].item("sample")["tokens"]["--bg"] = "paper"
    plan = cards.card_layer_plan(KIND, {"text": "Words"}, card_style)
    color = card_style.data["graphics"]["paper_color"]
    assert plan.registry["tokens"]["--bg"] == color
    host = registry.prepare(plan, tmp_path / "work")
    mount = next(attrs for _, attrs in Elements((host / "index.html").read_text()).elements
                 if attrs.get("id") == "hf-mount")
    assert f"--bg:{color}" in mount["style"]
    assert "--bg:transparent" not in mount["style"]


def test_colour_coverage_accepts_host_backdrop_but_refuses_unmapped_ink():
    entry = card_fields()
    catalog.validate_color_tokens("sample", entry, '<style>.clip{background:var(--bg, #090d16);'
                                  'border-radius:var(--radius, 16px)}</style>')
    with pytest.raises(ValidationError, match="unmapped colour token.*--ink-x"):
        catalog.validate_color_tokens("sample", entry,
                                      '<style>.tile{color:var(--ink-x, #ffffff)}</style>')


@pytest.mark.parametrize("brand_accent", [None, "#F34E3F"])
def test_accent_tint_resolves_from_creator_palette(registry_card, card_style, brand_accent):
    registry_card[0].item("sample")["tokens"]["--surface"] = "accent_tint"
    if brand_accent:
        card_style.data.setdefault("brand", {})["accent_color"] = brand_accent
    plan = cards.card_layer_plan(KIND, {"text": "Words"}, card_style)
    accent = brand_accent or card_style.data["graphics"]["accent_color"]
    assert plan.registry["tokens"]["--surface"] == cards.glow_colors(accent)[0]


@pytest.mark.parametrize("readability", ["halo", "none"])
def test_readability_filter_is_host_only(registry_card, card_style, tmp_path, readability):
    from yap_studio.hf_registry import cards as registry

    entry = registry_card[0].item("sample")
    entry["readability"] = readability
    plan = cards.card_layer_plan(KIND, {"text": "Words"}, card_style)
    host = registry.prepare(plan, tmp_path / "work")
    mount = next(attrs for _, attrs in Elements((host / "index.html").read_text()).elements
                 if attrs.get("id") == "hf-mount")
    if readability == "halo":
        chain = re.search(r"filter:([^;]+)", mount["style"])
        assert chain is not None
        assert 2 <= chain[1].count("drop-shadow(") <= 3
        assert "#000" in chain[1]
        assert "rgba(0,0,0," in chain[1]
    else:
        assert "filter:" not in mount["style"]
    control = tmp_path / "control"
    registry_card[2].localize("sample", control, install=registry_card[3],
                             fonts_by_role=plan.registry["fonts_by_role"], role="titles",
                             duration_s=hyperframes.layer_duration_s(plan.frames, plan.fps))
    for relative in entry["files"]:
        assert (host / relative).read_bytes() == (control / relative).read_bytes()
        assert (registry_card[1].item_files("sample") / relative).read_text() == SOURCE


@pytest.mark.parametrize("fallback", ["#123", "#12345678", "rgb(1, 2, 3)",
                                      "hsl(20 30% 40%)", "rebeccapurple", "transparent",
                                      "var(--surprise, #123456)"])
def test_card_refuses_unmapped_colour_token(synthetic, card_style, fallback):
    registry_catalog, registry_install, _, renderer, add, _ = synthetic
    add(source=SOURCE.replace("var(--brand,red)", f"var(--surprise, {fallback})"))
    entry = registry_catalog.item("sample")
    entry.update(card_fields())
    registry_install.install_items(install=renderer)
    with pytest.raises(ValidationError, match="unmapped colour token.*--surprise"):
        cards.card_layer_plan(KIND, {"text": "Words"}, card_style)


def test_colour_coverage_allows_fonts_and_other_fallbacks(registry_card, card_style):
    # Font fallbacks and non-colour variables must not become palette tokens.
    from yap_studio.hf_registry import catalog as registry_catalog

    registry_catalog.validate_color_tokens("sample", card_fields(), SOURCE + (
        '<style>p{font-family:var(--font-display, Geist, sans-serif);'
        'font-family:var(--font-body, Black, sans-serif);'
        'width:var(--width, 20px);display:var(--display, block)}</style>'
    ))


@pytest.mark.parametrize("name", ["char-slam-explode", "conic-progress-ring"])
def test_installed_pilot_colour_tokens(name, _shared_cache_root, monkeypatch):
    from yap_studio.hf_registry import install

    with monkeypatch.context() as gate:
        gate.setenv("YAP_STUDIO_CACHE", str(_shared_cache_root))
        source = install.installed(name)
    if source is None:
        pytest.skip(f"{name} is not installed in the gate cache; run yap-studio registry install")
    entry = catalog.item(name)
    catalog.validate_color_tokens(name, entry, (source / entry["main"]).read_text())


@pytest.mark.parametrize(("name", "preview"), [
    ("char-slam-explode", None), ("char-slam-explode", []),
    ("char-slam-explode", {}), ("char-slam-explode", {"text": "x" * 13}),
    ("char-slam-explode", {"text": 123}),
    ("char-slam-explode", {"text": "Ship it now", "unknown": "value"}),
    ("conic-progress-ring", {"percent": 101}),
    ("conic-progress-ring", {"percent": True}),
])
def test_catalog_validates_preview(tmp_path, monkeypatch, name, preview):
    data = copy.deepcopy(catalog.load_catalog())
    data["items"][name]["preview"] = preview
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(data))
    monkeypatch.setattr(catalog, "CATALOG_PATH", source)
    catalog.load_catalog.cache_clear()
    try:
        with pytest.raises(ValidationError, match="preview"):
            catalog.load_catalog()
    finally:
        catalog.load_catalog.cache_clear()


@pytest.mark.parametrize("status", ["on", "pilot"])
def test_catalog_requires_preview_only_for_on(tmp_path, monkeypatch, status):
    data = copy.deepcopy(catalog.load_catalog())
    entry = data["items"]["char-slam-explode"]
    entry["status"] = status
    entry.pop("preview", None)
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(data))
    monkeypatch.setattr(catalog, "CATALOG_PATH", source)
    catalog.load_catalog.cache_clear()
    try:
        if status == "pilot":
            assert catalog.load_catalog()["items"]["char-slam-explode"]["status"] == "pilot"
        else:
            with pytest.raises(ValidationError, match="preview"):
                catalog.load_catalog()
    finally:
        catalog.load_catalog.cache_clear()


def test_catalog_measured_text_contract():
    expected = {
        "char-slam-explode": (31, [640, 200], {"text": "Ship it now!"}),
        "conic-progress-ring": (49, [400, 220], {"percent": 100}),
    }
    for entry in catalog.items().values():
        if entry["status"] != "on":
            continue
        assert entry["measured"]["min_text_px"] >= cards.MIN_TEXT_PX
        assert entry["measured"]["at"]["panel"] == entry["min_panel"]
        slots = entry["measured"]["at"]["slots"]
        for slot, rule in entry["slots"].items():
            value = slots[slot]
            if rule["type"] == "string":
                assert len(value) == rule["max_length"]
            elif rule["type"] == "integer":
                assert value == rule["maximum"]
    for name, (minimum, panel, slots) in expected.items():
        assert catalog.item(name)["measured"] == {
            "min_text_px": minimum, "at": {"panel": panel, "slots": slots},
        }


@pytest.mark.parametrize("measured", [
    None, {}, {"min_text_px": 31},
    {"min_text_px": 29, "at": {"panel": [640, 200], "slots": {"text": "Ship it now!"}}},
    {"min_text_px": True, "at": {"panel": [640, 200], "slots": {"text": "Ship it now!"}}},
    {"min_text_px": float("nan"),
     "at": {"panel": [640, 200], "slots": {"text": "Ship it now!"}}},
    {"min_text_px": "31", "at": {"panel": [640, 200], "slots": {"text": "Ship it now!"}}},
    {"min_text_px": 31, "at": []},
    {"min_text_px": 31, "at": {"panel": [600, 200], "slots": {"text": "Ship it now!"}}},
    {"min_text_px": 31, "at": {"panel": [640, 200], "slots": {"text": "Short"}}},
    {"min_text_px": 31, "at": {"panel": [640, 200], "slots": {"text": "x" * 13}}},
    {"min_text_px": 31, "at": {"panel": [640, 200], "slots": {}}},
    {"min_text_px": 31, "at": {"panel": [640, 200], "slots": []}},
    {"min_text_px": 31, "at": {"panel": [640, 200]}},
])
def test_catalog_requires_valid_measurement(tmp_path, monkeypatch, measured):
    data = copy.deepcopy(catalog.load_catalog())
    entry = data["items"]["char-slam-explode"]
    entry.update(min_panel=[640, 200], measured=measured)
    entry["slots"]["text"]["max_length"] = 12
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(data))
    monkeypatch.setattr(catalog, "CATALOG_PATH", source)
    catalog.load_catalog.cache_clear()
    try:
        with pytest.raises(ValidationError, match="measured"):
            catalog.load_catalog()
    finally:
        catalog.load_catalog.cache_clear()


def test_catalog_requires_measurement_at_maximum_number(tmp_path, monkeypatch):
    data = copy.deepcopy(catalog.load_catalog())
    data["items"]["conic-progress-ring"]["measured"] = {
        "min_text_px": 49, "at": {"panel": [400, 220], "slots": {"percent": 99}},
    }
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(data))
    monkeypatch.setattr(catalog, "CATALOG_PATH", source)
    catalog.load_catalog.cache_clear()
    try:
        with pytest.raises(ValidationError, match="measured"):
            catalog.load_catalog()
    finally:
        catalog.load_catalog.cache_clear()


@pytest.mark.parametrize(("status", "min_text_px", "valid"), [
    ("on", None, False), ("pilot", None, True), ("on", 30, True), ("on", 30.5, True),
])
def test_measurement_required_only_for_on_items(tmp_path, monkeypatch, status, min_text_px, valid):
    data = copy.deepcopy(catalog.load_catalog())
    entry = data["items"]["char-slam-explode"]
    entry["status"] = status
    if min_text_px is None:
        entry.pop("measured")
    else:
        entry["measured"]["min_text_px"] = min_text_px
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(data))
    monkeypatch.setattr(catalog, "CATALOG_PATH", source)
    catalog.load_catalog.cache_clear()
    try:
        if valid:
            assert catalog.load_catalog()["items"]["char-slam-explode"]["status"] == status
        else:
            with pytest.raises(ValidationError, match="measured"):
                catalog.load_catalog()
    finally:
        catalog.load_catalog.cache_clear()


def test_headline_minimum_panel_guards_refit(synthetic, card_style):
    headline = json.loads(catalog.CATALOG_PATH.read_text())["items"]["char-slam-explode"]
    registry_catalog, install, _, renderer, add, _ = synthetic
    add(source=SOURCE)
    registry_catalog.items()["sample"].update({key: headline[key] for key in (
        "status", "kind", "mount", "min_panel", "slots", "variables", "tokens", "min_s", "measured",
    )})
    registry_catalog.items()["sample"]["variables"] = {"text": "{text}"}
    install.install_items(install=renderer)
    slots = {"text": "Ship it now!"}
    plan = planner.fitted_card(KIND, slots, card_style, frames=60, width=720)
    assert plan.width - 2 * cards.PAD_X == 640
    with pytest.raises(ValidationError, match="640.*200"):
        planner.fitted_card(KIND, slots, card_style, frames=60, width=716)
    with pytest.raises(ValidationError, match="1..12 characters"):
        cards.card_layer_plan(KIND, {"text": "x" * 13}, card_style)


@pytest.mark.parametrize(("field", "value"), [
    ("status", "bad"), ("status", []), ("kind", None),
    ("kind", {"category": "bad", "role": "cards"}),
    ("kind", {"category": "Words", "role": "bad", "label": "a", "use": "b"}),
    ("mount", "bad"), ("mount", {}), ("mount", "scale"), ("min_panel", [0, 200]),
    ("min_panel", [600]), ("min_s", 0.4), ("min_s", True),
    ("variables", {"text": "{unknown}"}), ("variables", {"text": "{text.foo}"}),
    ("tokens", {"brand": "accent"}), ("tokens", {"--brand": "unknown"}),
    ("tokens", {"--brand": {}}), ("variables", {"text": "{text!r}"}),
    ("variables", {"text": "{"}), ("min_s", float("nan")),
    ("slots", {"text": {"type": "string", "required": True, "max_length": -1}}),
    ("slots", {"text": {"type": "integer", "required": True,
                         "minimum": 100, "maximum": 0}}),
    ("slots", {"text": {"type": "enum", "required": False, "options": []}}),
    ("scale_max", 3),
])
def test_catalog_rejects_card_fields(tmp_path, monkeypatch, field, value):
    data = copy.deepcopy(catalog.load_catalog())
    entry = data["items"]["char-slam-explode"]
    if field == "scale_max":
        entry.update(mount="scale", stage=[1920, 1080], content=[0, 0, 1920, 1080])
    entry[field] = value
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(data))
    monkeypatch.setattr(catalog, "CATALOG_PATH", source)
    catalog.load_catalog.cache_clear()
    try:
        with pytest.raises(ValidationError):
            catalog.load_catalog()
    finally:
        catalog.load_catalog.cache_clear()


def test_catalog_scale_enum_and_id_limit(tmp_path, monkeypatch):
    data = copy.deepcopy(catalog.load_catalog())
    entry = data["items"]["char-slam-explode"]
    entry.update(mount="scale", stage=[1920, 1080], content=[200, 100, 1200, 600], scale_max=2)
    entry["slots"]["spread"] = {"type": "enum", "required": False,
                                  "options": ["tight", "wild"], "default": "tight"}
    source = tmp_path / "catalog.json"
    monkeypatch.setattr(catalog, "CATALOG_PATH", source)
    for valid, changes in [(True, {}), (False, {"content": [1800, 0, 200, 100]}),
                           (False, {"scale_max": 0})]:
        source.write_text(json.dumps(
            {**data, "items": {"char-slam-explode": {**entry, **changes}}}))
        catalog.load_catalog.cache_clear()
        try:
            if valid:
                assert catalog.load_catalog()["items"]["char-slam-explode"]["mount"] == "scale"
            else:
                with pytest.raises(ValidationError):
                    catalog.load_catalog()
        finally:
            catalog.load_catalog.cache_clear()
    name = "a" * 62
    entry.update(upstream_path=f"registry/components/{name}/{name}.html")
    source.write_text(json.dumps({**data, "items": {name: entry}}))
    try:
        with pytest.raises(ValidationError, match="64"):
            catalog.load_catalog()
    finally:
        catalog.load_catalog.cache_clear()


class Elements(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.elements = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


@pytest.mark.parametrize("template", [False, True])
@pytest.mark.parametrize(("original_duration", "frames"), [(1, 90), (4, 60)])
def test_prepared_item_duration_edits_are_exact(
    synthetic, card_style, tmp_path, template, original_duration, frames,
):
    registry_catalog, registry_install, localizer, renderer, add, _ = synthetic
    add(source=duration_source(template=template, duration=original_duration))
    entry = registry_catalog.item("sample")
    entry.update(card_fields())
    registry_install.install_items(install=renderer)
    cached = registry_install.item_files("sample") / entry["main"]
    original_bytes = cached.read_bytes()
    control = tmp_path / "control"
    localizer.localize("sample", control, install=renderer,
                       fonts_by_role={"titles": card_style.data["hook"]["font"]}, role="titles")
    before = (control / entry["main"]).read_text()
    plan = cards.card_layer_plan(KIND, {"text": "Words"}, card_style, frames=frames)
    composition = cards.prepare_card(plan, tmp_path / "work")
    duration = hyperframes.layer_duration_s(frames, plan.fps)
    expected = before.replace(f'DATA-DURATION = "{original_duration}"',
                              f'DATA-DURATION = "{duration}"')
    expected = expected.replace(f'data-duration={original_duration} data-whole-clip',
                                f'data-duration={duration} data-whole-clip')
    expected = expected.replace(f"data-composition-duration = '{original_duration}'",
                                f"data-composition-duration = '{duration}'")
    prepared = (composition / entry["main"]).read_text()
    assert prepared == expected
    assert cached.read_bytes() == original_bytes
    host = Elements((composition / "index.html").read_text())
    mount = next(attrs for _, attrs in host.elements
                 if attrs.get("data-composition-id") == "sample")
    assert mount["data-hf-authored-id"] == "root"
    assert hyperframes.preflight(composition, frames, plan.fps) > 0


@pytest.mark.parametrize("item_root_id", ["root", "hf-host"])
def test_host_root_id_is_distinct_from_item_root(
    synthetic, card_style, tmp_path, item_root_id,
):
    registry_catalog, registry_install, _, renderer, add, _ = synthetic
    add(source=SOURCE.replace('id="root"', f'id="{item_root_id}"'))
    registry_catalog.item("sample").update(card_fields())
    registry_install.install_items(install=renderer)
    plan = cards.card_layer_plan(KIND, {"text": "Words"}, card_style)
    composition = cards.prepare_card(plan, tmp_path / "work")
    elements = Elements((composition / "index.html").read_text()).elements
    host = next(attrs for _, attrs in elements
                if attrs.get("data-composition-id") == "hf-host")
    mount = next(attrs for _, attrs in elements
                 if attrs.get("data-composition-id") == "sample")
    assert host["id"] != item_root_id
    assert mount["data-hf-authored-id"] == item_root_id


@pytest.mark.parametrize("text", [
    "https://a.co", "HTTP://a.co", "ftp://a.co", "data://a.co", "a+b://a.co",
    "//a.co/x", "www.a.co", "data:a", "a.co", "url(a)", "url(//a.co)",
    "é🙂'\"&<>\\/://",
])
def test_creator_text_is_json_data_in_host(registry_card, card_style, tmp_path, text):
    plan = cards.card_layer_plan(KIND, {"text": text}, card_style, frames=60)
    composition = cards.prepare_card(plan, tmp_path / "work")
    elements = Elements((composition / "index.html").read_text()).elements
    mount = next(attrs for _, attrs in elements if attrs.get("data-composition-id") == "sample")
    values = json.loads(mount["data-variable-values"])
    assert values == plan.variables
    assert values["text"].encode("utf-8") == text.encode("utf-8")
    assert hyperframes.preflight(composition, plan.frames, plan.fps) > 0


def test_plan_host_fill_and_render_dispatch(registry_card, card_style, tmp_path, monkeypatch):
    from yap_studio.hf_registry import cards as registry

    plan = cards.card_layer_plan(KIND, {"text": "It's <fine>"}, card_style, frames=60)
    builtin = cards.card_layer_plan("stat", {"end": 37}, card_style, frames=60)
    assert (plan.x, plan.y, plan.width, plan.height) == (
        builtin.x, builtin.y, builtin.width, builtin.height)
    assert plan.fonts == ("Gelasio",)
    assert plan.variables == {"text": "It's <fine>", "progress": 37.0}
    assert isinstance(plan.variables["progress"], float)
    host = registry.prepare(plan, tmp_path / "work")
    source = (host / "index.html").read_text()
    roots = [attrs for _, attrs in Elements(source).elements if "data-composition-id" in attrs]
    assert [r["data-composition-id"] for r in roots] == ["hf-host", "sample"]
    mount = roots[1]
    assert json.loads(mount["data-variable-values"]) == plan.variables
    assert "&#39;" in source
    assert 'left:40px;top:24px;width:960px;height:272px' in mount["style"]
    assert "--brand:#123456" in mount["style"]
    assert "--fg:" + card_style.data["graphics"]["ink_color"] in mount["style"]
    assert 'src="gsap.min.js"' in source and "data-yap-fonts" in source
    assert 'window.__timelines["hf-host"]' in source
    assert "paused:true" in source and "opacity:0" in source and "Math.min(0.3" in source
    assert hyperframes.preflight(host, plan.frames, plan.fps) > 0
    calls = []
    monkeypatch.setattr(hyperframes, "render_layer", lambda *args, **kwargs: calls.append(kwargs))
    cards.render_card(plan, tmp_path / "again", tmp_path / "out.mov")
    assert calls[0]["variables"] == {}
    assert calls[0]["browser_gpu"] == hyperframes.CARD_BROWSER_GPU
    assert cards.video_slots(KIND) == {}
    monkeypatch.setattr(hyperframes, "ensure_installed", lambda **kw: pytest.fail(
        "an explicit renderer must not trigger an installation lookup"))
    cards.render_card(plan, tmp_path / "explicit", tmp_path / "out.mov",
                      install=registry_card[3])


@pytest.mark.parametrize("scale_max,expected", [(2, (0.8, -120, -160)),
                                              (0.25, (0.25, 320, 60))])
def test_scale_geometry(registry_card, card_style, tmp_path, scale_max, expected):
    registry_card[0].items()["sample"].update(
        mount="scale", stage=[1920, 1080], content=[200, 300, 1200, 200], scale_max=scale_max)
    plan = cards.card_layer_plan(KIND, {"text": "Words"}, card_style)
    host = cards.prepare_card(plan, tmp_path)
    mount = next(a for _, a in Elements((host / "index.html").read_text()).elements
                 if a.get("data-composition-src"))
    s, tx, ty = expected
    assert mount["data-width"] == "1920" and mount["data-height"] == "1080"
    assert f"translate({tx:g}px,{ty:g}px) scale({s:g})" in mount["style"]
    assert "transform-origin:0 0" in mount["style"]


@pytest.mark.parametrize("slots", [{}, {"text": "x" * 19}, {"text": "a", "percent": 101},
                                  {"text": "a", "percent": True}, {"text": "a", "bad": 1}])
def test_slot_errors_match_cards(registry_card, card_style, slots):
    with pytest.raises(ValidationError) as builtin:
        cards._slots(slots, card_fields()["slots"])
    with pytest.raises(ValidationError) as hf:
        cards.card_layer_plan(KIND, slots, card_style)
    assert hf.value.message == builtin.value.message


@pytest.mark.parametrize("frames", [30, 44])
def test_minimum_and_panel_refusals(registry_card, card_style, frames):
    card_style.data["graphics"]["hold_s"] = 1
    with pytest.raises(ValidationError, match=re.escape(
        "a hf:sample card needs at least 1.5 s (45 frames at 30 fps); "
        "raise graphics.hold_s or pass more frames")):
        cards.card_layer_plan(KIND, {"text": "Words"}, card_style, frames=frames)
    with pytest.raises(ValidationError, match="600.*200"):
        planner.fitted_card(KIND, {"text": "Words"}, card_style, frames=60, width=600)


@pytest.mark.parametrize(("min_s", "fps", "expected"), [(1.5, 30, 45), (1.5, 24, 36), (2, 30, 60)])
def test_registry_default_preview_extends_short_hold(
    registry_card, card_style, min_s, fps, expected,
):
    registry_card[0].item("sample")["min_s"] = min_s
    card_style.data["graphics"]["hold_s"] = 1
    assert cards.default_preview_frames(KIND, {"text": "Words"}, card_style, fps=fps) == expected


def test_opt_in_order_and_pilot(registry_card, card_style):
    from yap_studio.taste.model import TasteError, validate_style_field

    assert not cards.card_kind_off(card_style, KIND)
    off = replace(card_style, data=copy.deepcopy(card_style.data))
    off.data["graphics"].pop("card_kinds_on")
    assert cards.card_kind_off(off, KIND)
    assert "has not turned on" in cards.card_kind_off_detail(off, KIND)
    check_card_kinds_on([KIND], "kinds")
    assert validate_style_field("graphics.card_kinds_on", [KIND], "rule") == "graphics"
    for kind in ("hf:unknown", "hf:pilot"):
        registry_card[0].items()["pilot"] = {"status": "pilot"}
        with pytest.raises(StyleError, match="hf:sample"):
            check_card_kinds_on([kind], "kinds")
        with pytest.raises(TasteError, match="hf:sample"):
            validate_style_field("graphics.card_kinds_on", [kind], "rule")
        with pytest.raises(ValidationError):
            cards.card_layer_plan(kind, {"text": "Words"}, card_style)
    assert cards_on(card_style)[-1] == KIND


@pytest.mark.parametrize("roots,valid", [
    ('<div data-composition-id="host" {meta}><div data-composition-id="mount"></div></div>', True),
    ('<div data-composition-id="host" {meta}></div><div data-composition-id="other"></div>', False),
    ('<div data-composition-id="host" {meta}></div><aside>'
     '<div data-composition-id="mount"></div></aside>', False),
])
def test_top_level_roots(tmp_path, roots, valid):
    meta = 'data-no-timeline data-width="100" data-height="100" data-duration="2"'
    (tmp_path / "index.html").write_text(roots.format(meta=meta))
    if valid:
        assert hyperframes.preflight(tmp_path, 60) == 2
    else:
        with pytest.raises(YapError, match="exactly one"):
            hyperframes.preflight(tmp_path, 60)


def test_plan_cards_and_missing_install(registry_card, card_style):
    data = requests(template=KIND, slots={"text": "Ship it"}, hold_s=2)
    data["cards"][0].pop("end")
    result = planner.plan_cards(data, synthetic_words(), card_style,
                                frames=900, faces=synthetic_faces(box=None))
    assert result.cards[0].template == KIND and result.cards[0].render_key
    layer = cards.card_layer_plan(KIND, {"text": "Ship it"}, card_style)
    before = planner._render_key(layer)
    registry_card[0].items()["sample"]["tokens"]["--extra"] = "accent"
    assert planner._render_key(layer) != before
    shutil.rmtree(registry_card[1].item_dir("sample"))
    with pytest.raises(DependencyError) as missing:
        cards.card_layer_plan(KIND, {"text": "Ship it"}, card_style)
    assert missing.value.code == "registry_item_missing"
    assert missing.value.suggested_action


@pytest.mark.parametrize("display_record", [True, False])
def test_bleep_registry_card_mask_and_check(registry_card, layer_project, tmp_path, display_record):
    turn_on_card_kind(layer_project, KIND)
    style_path = layer_project / "style.json"
    style = json.loads(style_path.read_text())
    style["music"] = {"bleep_swears": True}
    style_path.write_text(json.dumps(style))
    set_elements(layer_project, graphic(template=KIND, slots={"text": "Fuck"}, hold_s=1.5))
    context = ProjectContext(layer_project)
    card, = graphics_snapshot(context).graphics
    assert card.plan.variables["text"] == "F***"
    placement = card.placement()
    assert placement["display_text"] == ["F***"]
    assert not (cards.template_dir(KIND) / "template.json").exists()
    if not display_record:
        placement.pop("display_text")
    synth.float_audio(layer_project / "cut/voice.wav", np.zeros(3 * synth.RATE, dtype=np.float32))
    snapshot = SimpleNamespace(
        small_files={"cut/cut-words.json": (layer_project / "cut/cut-words.json").read_bytes(),
                     "project.json": (layer_project / "project.json").read_bytes(),
                     "captions/phrases.json": b'{"phrases": []}'},
        title_layers=[placement], timeline=context.resolved_timeline(),
    )
    result, = assemble_stage._bleep_check(
        context, snapshot, SimpleNamespace(bleep_voice=None), layer_project, tmp_path / "mix")
    assert result["pass"] is display_record
    assert result["audible"] == []
    assert result["unmasked"] == ([] if display_record else [
        {"time_s": .2, "where": "card agent-tag", "masked": "F***"}])


def test_snapshot_receipts_and_live(registry_card, layer_project, tmp_path, monkeypatch):
    from yap_studio.gallery.live import live_layers, sample_words
    from yap_studio.stages.graphics import _render_card

    turn_on_card_kind(layer_project, KIND)
    set_elements(layer_project, graphic(template=KIND, slots={"text": "Ship it"}, hold_s=1.5))
    context = ProjectContext(layer_project)
    reads = []
    read = context.read
    monkeypatch.setattr(context, "read", lambda name: (reads.append(name), read(name))[1])
    snapshot, = graphics_snapshot(context).graphics
    assert str(catalog.CATALOG_PATH) in reads
    assert snapshot.plan.template == KIND
    assert snapshot.assets.metadata
    old = snapshot.digest
    calls = []
    monkeypatch.setattr(cards, "render_card", lambda *args, **kwargs: calls.append(args[0]))
    _render_card(snapshot, tmp_path / "render", tmp_path / "out.mov")
    assert calls[0].template == KIND
    registry_card[0].items()["sample"]["variables"]["text"] = "Hi {text}"
    with pytest.raises(ValidationError, match="template or font changed"):
        _render_card(snapshot, tmp_path / "stale", tmp_path / "out.mov")
    assert graphics_snapshot(ProjectContext(layer_project)).graphics[0].digest != old
    style = context.render_style()
    live, = live_layers(style, sample_words("Ship it today"), tmp_path,
                        card=KIND, slots={"text": "Ship it"}, only="card")
    assert live.variables == {} and (live.composition / "index.html").is_file()


@pytest.mark.parametrize(("face", "expected"), [
    (Box(458, 629, 396, 396), "top"),
    (Box(458, 100, 396, 300), "lower"),
    (Box(0, 0, 1080, 1920), "blocked"),
])
def test_shared_placement_and_blocked_refusals(registry_card, card_style, face, expected):
    data = requests(template=KIND, slots={"text": "Ship it"})
    result = planner.plan_cards(data, synthetic_words(), card_style,
                                frames=900, faces=synthetic_faces(face))
    placed = result.cards[0]
    if expected == "blocked":
        assert placed.clearance["status"] == "blocked"
        assert result.status == "needs_review"
    else:
        assert placed.clearance["status"] == "clear"
        assert placed.clearance["position"] == expected
        assert placed.clearance["gap_px"] >= placed.clearance["margin_px"]


def test_minimum_panel_skips_side_corridor_and_caption_gap(registry_card, card_style):
    layer = cards.card_layer_plan(KIND, {"text": "Words"}, card_style)
    face = Box(458, 100, 396, 300)
    placed, _, clearance = planner.place_card(
        layer, {"text": "Words"}, card_style, [(face,)], caption_top=1000)
    assert clearance["position"] == "lower" and placed.width == layer.width
    assert placed.y + placed.height <= 1000 - card_style.data["graphics"]["caption_gap_px"]
    _, _, blocked = planner.place_card(
        layer, {"text": "Words"}, card_style, [(face,)], caption_top=400)
    assert blocked["status"] == "blocked"
    # A catalogued item with a smaller minimum can use the same side corridor.
    registry_card[0].items()["sample"]["min_panel"] = [200, 200]
    _, _, side = planner.place_card(
        layer, {"text": "Words"}, card_style, [(face,)], caption_top=1000)
    assert side["position"] == "side"


def test_graphics_snapshot_refuses_blocked_card(registry_card, layer_project, monkeypatch):
    from yap_studio.stages import graphics as stage

    turn_on_card_kind(layer_project, KIND)
    set_elements(layer_project, graphic(template=KIND, slots={"text": "Words"}, hold_s=1.5))
    monkeypatch.setattr(stage, "picture_faces", lambda *args: synthetic_faces(
        Box(0, 0, 1080, 1920)))
    with pytest.raises(ValidationError, match="no clear top, readable side or lower") as blocked:
        graphics_snapshot(ProjectContext(layer_project))
    assert blocked.value.code and blocked.value.suggested_action
