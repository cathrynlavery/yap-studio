"""Offline registry classification and draft catalog contracts."""

from __future__ import annotations

import copy
import hashlib
import html
import importlib.util
import io
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import numpy as np
import pytest
from test_hf_registry import synthetic as synthetic

from yap_studio import cards, ffmpeg, paths
from yap_studio.errors import DependencyError, ValidationError, YapError
from yap_studio.hf_registry import cards as registry_cards
from yap_studio.hf_registry import catalog, install, localize, triage
from yap_studio.render.hyperframes import HYPERFRAMES_VERSION, RendererInstall
from yap_studio.style import resolve

pytestmark = pytest.mark.usefixtures("_no_real_node")
FIXTURE = Path(__file__).parent / "fixtures" / "hf-registry"


@pytest.fixture
def registry():
    result = {}
    for item in json.loads((FIXTURE / "registry.json").read_text())["items"]:
        name = item["name"]
        folder = FIXTURE / (item["type"].removeprefix("hyperframes:") + "s") / name
        manifest = json.loads((folder / "registry-item.json").read_text())
        files, source = None, None
        if item["type"] != "hyperframes:example":
            files = {
                file["target"]: hashlib.sha256((folder / file["path"]).read_bytes()).hexdigest()
                for file in manifest["files"]
            }
            main = next(file for file in manifest["files"]
                        if file["type"] in ("hyperframes:snippet", "hyperframes:composition"))
            source = (folder / main["path"]).read_text()
        result[name] = manifest, files, source
    assert len(result) == 8
    return result


def classify(name="sample", variables=(), *, source=None, tags=(), **kwargs):
    from yap_studio.hf_registry.triage import classify as classify_item

    main = f"compositions/components/{name}.html"
    manifest = {"type": "hyperframes:component", "title": "Sample", "tags": list(tags),
                "files": [{"type": "hyperframes:snippet", "target": main}]}
    if source is None:
        declaration = html.escape(json.dumps(list(variables)), quote=True)
        source = (f'<div data-composition-id="sample" '
                  f'data-composition-variables="{declaration}"></div>')
    return classify_item(name, manifest, {main: "0" * 64}, source, **kwargs)


@pytest.fixture
def records(registry):
    from yap_studio.hf_registry.triage import classify

    return {name: classify(name, *args) for name, args in registry.items()}


def test_fixture_titles_and_cards(records):
    title = records["titlecard-calm"]
    assert title["use"] == "title"
    assert title["reason"] == "Title: text slots headline, kicker."
    entry = title["entry"]
    assert entry["mount"] == "fill"
    assert entry["preview"] == {"headline": "Less, but better", "kicker": "DESIGN PRINCIPLE"}
    assert entry["variables"] == {"headline": "{headline}", "kicker": "{kicker}"}
    assert entry["kind"] == {"role": "titles", "category": "Words", "label": "Titlecard Calm",
                             "use": "text up to 16 characters; text up to 16 characters"}
    assert all(rule == {"type": "string", "required": True, "min_length": 1, "max_length": 16}
               for rule in entry["slots"].values())
    ring = records["conic-progress-ring"]
    assert ring["use"] == "card"
    assert ring["reason"] == "Card: number slot progress; text slot label."
    entry = ring["entry"]
    assert entry["kind"]["category"] == "Numbers"
    assert entry["slots"] == {
        "progress": {"type": "integer", "required": True, "minimum": 0, "maximum": 100},
        "label": {"type": "string", "required": True, "min_length": 1, "max_length": 12},
    }
    assert entry["preview"] == {"progress": 100, "label": "100"}
    assert entry["tokens"] == {"--brand": "accent", "--surface": "accent_tint", "--fg": "paper"}
    push = records["push-in"]
    assert push["use"] == "card"
    assert push["entry"]["kind"]["category"] == "Words"
    assert list(push["entry"]["slots"]) == ["text"]
    assert "--bg" not in push["entry"]["tokens"]


def test_fixture_caption_and_exclusions(records):
    caption = records["caption-kinetic-slam"]
    assert caption["use"] == "caption"
    assert caption["reason"] == (
        "Caption look with its own sample words; using it for captions needs a caption port."
    )
    entry = caption["entry"]
    assert entry["mount"] == "scale"
    assert entry["stage"] == [1920, 1080]
    assert entry["content"] == [0, 0, 1920, 1080]
    assert entry["scale_max"] == 1
    assert entry["slots"] == {}
    assert entry["kind"]["role"] == "titles"
    assert entry["kind"]["use"] == "a caption look with its own sample words"
    for name, reason in {
        "lt-color-block": "No slot for the creator's words or numbers",
        "rgb-glitch-text": "A paste-in snippet without its own composition",
        "cursor-zoom-follow": "Needs pictures: image.",
        "decision-tree": "An example project; HyperFrames installs it with init, not add.",
    }.items():
        record = records[name]
        assert record["use"] == "none" and record["reason"].startswith(reason)
        assert set(record) == {"type", "use", "reason"}


def test_usable_drafts_fit_catalog_contract(registry, records):
    from yap_studio.hf_registry.triage import MIN_S, STYLE

    style = resolve(STYLE).data
    panel = [style["graphics"]["card_width_px"],
             style["internal"]["key_text_bottom_px"] - style["internal"]["key_text_top_px"]
             - cards.PAD_TOP - cards.PAD_BOTTOM]
    for name, record in records.items():
        if record["use"] == "none":
            continue
        entry = record["entry"]
        assert record["check"] == {"status": "pending", "detail": ""}
        assert entry["min_panel"] == panel
        assert entry["min_s"] == MIN_S == 1.5
        assert entry["files"] == registry[name][1]
        assert entry["main"] in entry["files"]
        assert entry["upstream_path"] == f"registry/{record['type']}s/{name}/{name}.html"
        assert "status" not in entry and "measured" not in entry
        limits = {slot: ("x" * rule["max_length"] if rule["type"] == "string" else
                         rule["maximum"] if rule["type"] == "integer" else rule["default"])
                  for slot, rule in entry["slots"].items()}
        catalog.validate_color_tokens(name, entry, registry[name][2])
        catalog._card_metadata(name, {**entry, "status": "on", "measured": {
            "min_text_px": 30, "at": {"panel": panel, "slots": limits}}})


@pytest.mark.parametrize("source", [
    '<script>window.sample = () => {};</script>',
    ' \n<!-- <div data-composition-id="fake"></div> -->\n<script>sample();</script> ',
    '<script data-composition-variables=\'[{"id":"image","type":"image"}]\'>sample();</script>',
])
def test_script_only_helper(source):
    record = classify(source=source)
    assert record["use"] == "helper"
    assert record["reason"] == "A script-only snippet that built-in templates can call."
    assert set(record["entry"]) == {"type", "upstream_path", "main", "files"}
    assert record["check"] == {"status": "pending", "detail": ""}


@pytest.mark.parametrize("source", [
    '<script src="local.js"></script>', '<script>a();</script><script>b();</script>',
    '<div></div><script>a();</script>', 'words<script>a();</script>',
    '<!doctype html><script>a();</script>', '<script>a();', '',
])
def test_other_snippets_are_not_helpers(source):
    assert classify(source=source)["use"] == "none"


def test_early_exclusions(registry):
    from yap_studio.hf_registry.triage import classify

    manifest, files, source = registry["titlecard-calm"]
    assert classify("sample", {**manifest, "deprecated": True}, files, source,
                    error="failure")["reason"] == "Deprecated upstream."
    error = "x" * 250
    record = classify("sample", manifest, None, None, error=error)
    assert record["reason"] == "HyperFrames add failed: " + error[:200]
    assert record["use"] == "none"
    manifest = {**manifest, "type": "hyperframes:example", "deprecated": True}
    assert classify("sample", manifest, None, None, error="failure")["reason"].startswith(
        "An example project;")


@pytest.mark.parametrize("declaration", ["{", "{}", "null", '[{"id": "text"}]'])
def test_invalid_variable_declarations(declaration):
    record = classify(source=f"<div data-composition-id='sample' "
                      f"data-composition-variables='{declaration}'></div>")
    assert record == {"type": "component", "use": "none",
                      "reason": "Its variable declarations are not valid JSON."}


def test_slots_use_all_declarations_and_skip_comments_and_duplicates():
    declarations = [
        {"id": "boardText", "type": "string", "default": "Synthetic text"},
        {"id": "board-text", "type": "string", "default": "Duplicate"},
        {"id": "from-text", "type": "string", "default": ""},
        {"id": "9invalid", "type": "string"},
        {"id": "layout", "type": "string", "role": "layout"},
    ]
    declaration = html.escape(json.dumps(declarations), quote=True)
    source = ("<!-- <div data-composition-variables='broken'></div> -->"
              f'<html data-composition-variables="{declaration}">'
              '<div data-composition-id="sample" data-composition-variables=\'['
              '{"id":"tail","type":"string","default":"End"}]\'></div></html>')
    entry = classify(source=source)["entry"]
    assert list(entry["slots"]) == ["board_text", "from_text", "tail"]
    assert entry["variables"] == {"boardText": "{board_text}", "from-text": "{from_text}",
                                  "tail": "{tail}"}
    assert entry["preview"] == {"board_text": "Synthetic text", "from_text": "Sample text",
                                "tail": "End"}


def test_roleless_number_is_not_a_slot():
    record = classify(variables=[{"id": "stars", "type": "number", "default": 100}])
    assert record["use"] == "none"
    assert record["reason"].startswith("No slot")


def test_roleless_string_remains_a_slot_without_number_knobs():
    entry = classify(variables=[
        {"id": "boardText", "type": "string", "default": "Synthetic title"},
        {"id": "stars", "type": "number", "default": 100},
    ])["entry"]
    assert list(entry["slots"]) == ["board_text"]
    assert entry["preview"] == {"board_text": "Synthetic title"}
    assert entry["variables"] == {"boardText": "{board_text}"}
    assert entry["kind"]["category"] == "Words"


@pytest.mark.parametrize(("default", "step"), [(1.5, 1), (2, 0.5), (True, 1)])
def test_non_integral_numbers_are_not_slots(default, step):
    record = classify(variables=[{"id": "amount", "type": "number", "role": "content",
                                  "default": default, "step": step}])
    assert record["use"] == "none" and record["reason"].startswith("No slot")


def test_integer_bounds_clamp_and_enum_defaults():
    entry = classify(variables=[
        {"id": "amount", "type": "number", "role": "content", "default": 120.0,
         "min": 10.0, "max": 100.0},
        {"id": "below", "type": "number", "role": "content", "default": -5, "min": 0, "max": 10},
        {"id": "fallback", "type": "number", "role": "content", "default": 200,
         "min": 0.5, "max": 1.5},
        {"id": "spread", "type": "enum", "role": "content", "default": "missing",
         "options": ["tight", {"value": "wild", "label": "Wild"}]},
        {"id": "tone", "type": "enum", "role": "content", "default": "wild",
         "options": ["tight", "wild"]},
        {"id": "unassigned", "type": "enum", "options": ["one", "two"]},
    ])["entry"]
    assert entry["slots"]["amount"] == {"type": "integer", "required": True,
                                        "minimum": 10, "maximum": 100}
    assert entry["slots"]["fallback"]["maximum"] == 200
    assert entry["preview"] == {"amount": 100, "below": 0, "fallback": 200,
                                "spread": "tight", "tone": "wild"}
    assert entry["slots"]["spread"] == {"type": "enum", "required": False,
                                        "options": ["tight", "wild"], "default": "tight"}
    assert "a number from 10 to 100" in entry["kind"]["use"]
    assert "one of tight, wild" in entry["kind"]["use"]


@pytest.mark.parametrize(("tags", "use", "category"), [
    (["typography"], "title", "Words"), (["headline", "lower-third"], "card", "Words"),
    (["chat"], "card", "Chat"), (["code-animation"], "card", "Editor"),
    (["caption-style"], "caption", "Words"),
])
def test_tag_classification(tags, use, category):
    record = classify(variables=[{"id": "text", "type": "string"}], tags=tags)
    assert record["use"] == use and record["entry"]["kind"]["category"] == category
    assert list(record["entry"]["slots"]) == ["text"]


def test_number_precedes_title_tags_and_picture_slots_precede_captions():
    assert classify(variables=[{"id": "count", "type": "number", "role": "content", "default": 3}],
                    tags=["headline"])["entry"]["kind"]["category"] == "Numbers"
    record = classify(variables=[{"id": "photo", "type": "image"},
                                 {"id": "clip", "type": "video"}], tags=["caption-style"])
    assert record["reason"] == "Needs pictures: photo, clip."


def test_first_root_dimensions_and_token_sources():
    source = ('<div data-composition-id="first" data-width="800" data-height="600" '
              'data-composition-variables=\'[{"id":"text","type":"string"}]\'>'
              '<div data-composition-id="second" data-width="100" data-height="100"></div>'
              '<style>.x{color:var(--BRAND-panel,red);border:var(--border,blue);'
              'background:var(--bg,black);fill:var(--unknown,white);'
              'font-family:var(--font,Black)}</style></div>')
    entry = classify(source=source)["entry"]
    assert entry["stage"] == [800, 600]
    assert entry["tokens"] == {"--BRAND-panel": "accent", "--border": "accent_tint",
                                "--unknown": "paper"}
    assert classify(source=source.replace('data-width="800"', 'data-width="800.5"'))[
        "entry"]["mount"] == "fill"


@pytest.mark.parametrize("dimension", ["data-width", "data-height"])
def test_missing_dimension_value_uses_fill(dimension):
    source = ('<div data-composition-id="sample" data-width="800" data-height="600" '
              'data-composition-variables=\'[{"id":"text","type":"string"}]\'></div>')
    value = "800" if dimension == "data-width" else "600"
    entry = classify(source=source.replace(f'{dimension}="{value}"', dimension))["entry"]
    assert entry["mount"] == "fill"
    assert not {"stage", "content", "scale_max"} & entry.keys()


def test_color_tokens_ignore_font_family_and_find_nested_fallbacks():
    source = ('<style>.x{font-family:var(--font,Black);FONT-FAMILY:var(--other,red);'
              'color:var(--outer,var(--nested,red));background:var(--bg,#123456);'
              'fill:var(--color,oklch(50% 0.2 30));width:var(--width,10px)}</style>')
    assert catalog.color_tokens(source) == {"--nested", "--bg", "--color"}
    with pytest.raises(ValidationError) as raised:
        catalog.validate_color_tokens("sample", {"tokens": {}}, source)
    assert raised.value.detail == "sample: unmapped colour token: --color, --nested"


def test_trial_overlays_and_restores_catalog(records):
    shipped = copy.deepcopy(catalog.load_catalog())
    entry = records["titlecard-calm"]["entry"]
    with catalog.trial({"titlecard-calm": entry}):
        assert catalog.item("titlecard-calm") is entry
        assert catalog.load_catalog() == shipped
    assert "titlecard-calm" not in catalog.items()
    enabled = {**entry, "status": "on"}
    with catalog.trial({"titlecard-calm": enabled, "conic-progress-ring": enabled}):
        assert "hf:titlecard-calm" in registry_cards.kinds()
        assert catalog.card_catalog()["items"]["titlecard-calm"] is enabled
        assert catalog.item("conic-progress-ring") is enabled
        with catalog.trial({"titlecard-calm": entry}):
            assert catalog.item("titlecard-calm") is entry
            assert catalog.item("conic-progress-ring") is enabled
        assert catalog.item("titlecard-calm") is enabled
        with pytest.raises(RuntimeError), catalog.trial({"nested": entry}):
            raise RuntimeError("synthetic failure")
        assert "nested" not in catalog.items()
    assert catalog.items() == shipped["items"]
    with pytest.raises(RuntimeError), catalog.trial({"titlecard-calm": entry}):
        raise RuntimeError("synthetic failure")
    assert catalog.items() == shipped["items"]


@pytest.fixture
def triage_document(records):
    from yap_studio.hf_registry.triage import document

    for name, entry in catalog.load_catalog()["items"].items():
        records.setdefault(name, {"type": entry["type"], "use": "none",
                                  "reason": "Synthetic exclusion."})
    return document(records, commit=catalog.load_catalog()["registry"]["commit"],
                    version=HYPERFRAMES_VERSION)


@pytest.fixture
def triage_path(tmp_path, monkeypatch):
    target = tmp_path / "triage.json"
    monkeypatch.setattr(catalog, "TRIAGE_PATH", target)
    catalog.load_triage.cache_clear()
    try:
        yield target
    finally:
        catalog.load_triage.cache_clear()


def test_document_and_dumps_sort_and_round_trip(records):
    from yap_studio.hf_registry.triage import STYLE, document, dumps

    doc = document(dict(reversed(list(records.items()))), commit="a" * 40, version="synthetic")
    assert doc == {"schema": catalog.TRIAGE_SCHEMA, "hyperframes_version": "synthetic",
                   "commit": "a" * 40, "style": STYLE, "items": records}
    assert list(doc["items"]) == sorted(records)
    encoded = dumps(doc)
    assert json.loads(encoded) == doc and encoded.endswith("\n")
    lines = encoded.splitlines()
    assert lines[0] == "{" and lines[1].startswith('  "schema": ')
    item_lines = [line for line in lines if line.startswith('    "')]
    assert len(item_lines) == len(records)
    for name, line in zip(sorted(records), item_lines, strict=True):
        assert line == "    " + json.dumps(name) + ": " + json.dumps(
            records[name], sort_keys=True, separators=(",", ":")) + (
                "," if name != sorted(records)[-1] else "")
    assert list(json.loads(dumps({**doc, "items": dict(reversed(list(records.items())))}))[
        "items"]) == sorted(records)


def test_load_triage_valid_and_cached(triage_path, triage_document):
    triage_path.write_text(json.dumps(triage_document))
    assert catalog.load_triage() == triage_document
    triage_path.write_text("broken")
    assert catalog.load_triage() == triage_document


@pytest.mark.parametrize(("keys", "value", "message"), [
    ((), [], "expected an object"),
    (("schema",), "wrong", "schema"),
    (("hyperframes_version",), "0.0.0", "a HyperFrames bump must refresh the triage"),
    (("commit",), "a" * 40, "refresh it with scripts/bump_hyperframes.py --registry"),
    (("style",), " ", "style"), (("style",), 1, "style"),
    (("items",), {}, "items"), (("items",), [], "items"),
    (("items", "Bad_name"), {}, "invalid item name"),
    (("items", "titlecard-calm"), [], "record must be an object"),
    (("items", "titlecard-calm", "type"), "other", "type"),
    (("items", "titlecard-calm", "use"), "other", "use"),
    (("items", "titlecard-calm", "reason"), " ", "reason"),
    (("items", "titlecard-calm", "entry"), None, "entry"),
    (("items", "titlecard-calm", "entry", "type"), "block", "entry.type"),
    (("items", "titlecard-calm", "entry", "upstream_path"), "wrong", "upstream_path"),
    (("items", "titlecard-calm", "entry", "files"), {}, "files"),
    (("items", "titlecard-calm", "entry", "files"), {"safe.html": "bad"}, "64 hex"),
    (("items", "titlecard-calm", "entry", "main"), "missing.html", "main"),
    (("items", "titlecard-calm", "entry", "main"), [], "main"),
    (("items", "titlecard-calm", "check"), None, "check"),
    (("items", "titlecard-calm", "check", "status"), "unknown", "check"),
    (("items", "titlecard-calm", "check", "detail"), None, "check"),
    (("items", "lt-color-block", "entry"), {}, "must not have entry"),
    (("items", "lt-color-block", "check"), {}, "must not have check"),
])
def test_load_triage_rejects_invalid_metadata(triage_path, triage_document, keys, value, message):
    if keys:
        target = triage_document
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    else:
        triage_document = value
    triage_path.write_text(json.dumps(triage_document))
    with pytest.raises(ValidationError, match=re.escape(message)) as raised:
        catalog.load_triage()
    assert raised.value.parameter == "HyperFrames registry triage"


@pytest.mark.parametrize("unsafe", ["../escape.html", "/absolute.html", "a/./b", "a//b",
                                    "a\\b", "a:b", ""])
def test_load_triage_rejects_unsafe_files(triage_path, triage_document, unsafe):
    triage_document["items"]["titlecard-calm"]["entry"]["files"] = {unsafe: "0" * 64}
    triage_path.write_text(json.dumps(triage_document))
    with pytest.raises(ValidationError, match="unsafe file path"):
        catalog.load_triage()


@pytest.mark.parametrize("field", ["entry", "check"])
def test_load_triage_requires_usable_metadata(triage_path, triage_document, field):
    del triage_document["items"]["titlecard-calm"][field]
    triage_path.write_text(json.dumps(triage_document))
    with pytest.raises(ValidationError, match=field):
        catalog.load_triage()


@pytest.mark.parametrize("status", ["pass", "fail", "pending"])
def test_load_triage_accepts_check_statuses(triage_path, triage_document, status):
    triage_document["items"]["titlecard-calm"]["check"]["status"] = status
    triage_path.write_text(json.dumps(triage_document))
    assert catalog.load_triage() == triage_document


def test_load_triage_requires_every_shipped_item(triage_path, triage_document):
    del triage_document["items"]["echo-smear"]
    triage_path.write_text(json.dumps(triage_document))
    with pytest.raises(ValidationError, match="echo-smear"):
        catalog.load_triage()


@pytest.mark.parametrize("content", [None, "broken", b"\xff"])
def test_load_triage_unreadable(triage_path, content):
    if content is not None:
        triage_path.write_bytes(content.encode() if isinstance(content, str) else content)
    with pytest.raises(ValidationError, match="cannot read metadata") as raised:
        catalog.load_triage()
    assert raised.value.parameter == "HyperFrames registry triage"


@pytest.fixture
def registry_add(tmp_path):
    audit = tmp_path / "adds.jsonl"
    control = tmp_path / "add-control.json"
    control.write_text("{}")
    executable = tmp_path / "fake-hyperframes"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, shutil, sys\nfrom pathlib import Path\n"
        f"fixture = Path({str(FIXTURE.resolve())!r})\n"
        f"control = json.loads(Path({str(control)!r}).read_text())\n"
        "name = sys.argv[2]\nroot = Path(sys.argv[4])\n"
        "assert sys.argv[1:] == ['add', name, '--dir', str(root), '--json', '--no-clipboard']\n"
        "assert Path(os.environ['HOME']).is_dir()\n"
        "assert Path(os.environ['HYPERFRAMES_HISTORY_ROOT']).is_dir()\n"
        f"assert json.loads((root / 'hyperframes.json').read_text())['registry'] == "
        f"{catalog.registry_url()!r}\n"
        f"with Path({str(audit)!r}).open('a') as stream:\n"
        "    stream.write(json.dumps({'name': name, 'staging': str(root), "
        "'history': os.environ['HYPERFRAMES_HISTORY_ROOT']}) + '\\n')\n"
        "assert name != 'decision-tree'\n"
        "mode = control.get('mode') if control.get('name') == name else None\n"
        "if mode == 'error':\n"
        "    print(json.dumps({'ok': False, 'error': 'synthetic add failed'}))\n"
        "    sys.exit(1)\n"
        "item = next(item for item in json.loads((fixture / 'registry.json').read_text())"
        "['items'] if item['name'] == name)\n"
        "folder = fixture / (item['type'].removeprefix('hyperframes:') + 's') / name\n"
        "written = []\n"
        "for file in json.loads((folder / 'registry-item.json').read_text())['files']:\n"
        "    target = root / file['target']\n"
        "    target.parent.mkdir(parents=True, exist_ok=True)\n"
        "    shutil.copyfile(folder / file['path'], target)\n"
        "    written.append(str(target))\n"
        "if mode == 'outside': written.append(str(root.parent / 'outside.html'))\n"
        "if mode == 'relative': written = ['relative.html']\n"
        "if mode == 'symlink':\n"
        "    target.unlink()\n"
        "    target.symlink_to(folder / file['path'])\n"
        "if mode == 'missing': written = []\n"
        "print(json.dumps({'ok': True, 'written': written}))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    renderer = RendererInstall(HYPERFRAMES_VERSION, tmp_path / "renderer", executable,
                              None, "v26.1.0")

    def fetch(url):
        prefix = catalog.registry_url() + "/"
        assert url.startswith(prefix)
        return (FIXTURE / url.removeprefix(prefix)).read_bytes()

    return renderer, fetch, audit, control


def test_survey_stages_every_non_example_and_matches_classification(
    registry_add, records, tmp_path,
):
    renderer, fetch, audit, _ = registry_add
    work = tmp_path / "survey"
    result = triage.survey(catalog.registry_url(), executable=renderer.executable, work=work,
                           home=paths.validate_output_dir(install.registry_home()), fetch=fetch)
    assert result == records
    assert len(result) == 8 and list(result) == sorted(result)
    calls = [json.loads(line) for line in audit.read_text().splitlines()]
    assert {call["name"] for call in calls} == records.keys() - {"decision-tree"}
    assert len(calls) == len({call["staging"] for call in calls}) == 7
    assert len({call["history"] for call in calls}) == 7
    assert not list(work.iterdir())
    assert all(not Path(call["history"]).exists() for call in calls)


@pytest.mark.parametrize(("mode", "reason"), [
    ("error", "synthetic add failed"),
    ("outside", "HyperFrames add wrote outside its staging folder."),
    ("relative", "HyperFrames add wrote outside its staging folder."),
    ("symlink", "HyperFrames add wrote outside its staging folder."),
    ("missing", "HyperFrames add did not write its main file."),
])
def test_survey_add_failures_are_item_local(registry_add, tmp_path, mode, reason):
    renderer, fetch, _, control = registry_add
    control.write_text(json.dumps({"name": "conic-progress-ring", "mode": mode}))
    result = triage.survey(catalog.registry_url(), executable=renderer.executable,
                           work=tmp_path / "survey",
                           home=paths.validate_output_dir(install.registry_home()), fetch=fetch)
    assert result["conic-progress-ring"]["use"] == "none"
    assert reason in result["conic-progress-ring"]["reason"]
    assert "entry" not in result["conic-progress-ring"]
    assert result["titlecard-calm"]["use"] == "title"


@pytest.mark.parametrize("failure", [URLError("offline"), TimeoutError("timed out"),
                                    HTTPError("https://example.invalid", 404, "missing", {}, None)])
@pytest.mark.parametrize("target", ["registry.json", "examples/decision-tree/registry-item.json"])
def test_survey_fetch_failure_stops_without_classifying(
    registry_add, tmp_path, monkeypatch, failure, target,
):
    renderer, fetch, audit, _ = registry_add

    def urlopen(url, *, timeout):
        assert timeout == 60
        if url == catalog.registry_url() + "/" + target:
            raise failure
        return io.BytesIO(fetch(url))

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    with pytest.raises(DependencyError) as raised:
        triage.survey(catalog.registry_url(), executable=renderer.executable, work=tmp_path,
                      home=paths.validate_output_dir(install.registry_home()))
    assert raised.value.code == "registry_triage_fetch"
    assert raised.value.suggested_action == "Check the network and run the triage again."
    assert not audit.exists()


@pytest.mark.parametrize("status", ["pass", "fail"])
def test_merge_carries_completed_checks_for_identical_usable_entries(records, status):
    previous = copy.deepcopy(records)
    previous["titlecard-calm"]["check"] = {"status": status, "detail": "Earlier result."}
    previous["titlecard-calm"]["reason"] = "Earlier wording."
    assert triage.merge(previous, records) is records
    assert records["titlecard-calm"]["check"] == previous["titlecard-calm"]["check"]
    assert records["decision-tree"] == previous["decision-tree"]


@pytest.mark.parametrize("change", ["entry", "use", "pending", "new", "no-previous"])
def test_merge_leaves_new_or_changed_items_pending(records, change):
    previous = copy.deepcopy(records)
    old = previous["titlecard-calm"]
    old["check"] = {"status": "pass", "detail": "Earlier result."}
    if change == "entry":
        old["entry"]["preview"]["headline"] = "Changed"
    elif change == "use":
        old["use"] = "card"
    elif change == "pending":
        old["check"] = {"status": "pending", "detail": "Earlier pending."}
    elif change == "new":
        del previous["titlecard-calm"]
    else:
        previous = None
    expected = copy.deepcopy(records)
    assert triage.merge(previous, records) == expected


@pytest.mark.parametrize(("drawn", "opaque", "padding", "status", "detail"), [
    (0, 0, False, "fail", "Nothing drew in the card panel at 1.8 s."),
    (99, 0, False, "fail", "Nothing drew in the card panel at 1.8 s."),
    (100, 0, False, "pass", "Draws on 1% of the panel."),
    (1250, 0, False, "pass", "Draws on 12% of the panel."),
    (10000, 10000, False, "fail", "An opaque backdrop covers the card band."),
    (10000, 9000, False, "fail", "An opaque backdrop covers the card band."),
    (10000, 8999, False, "pass", "Draws on 100% of the panel."),
    (0, 0, True, "fail", "Nothing drew in the card panel at 1.8 s."),
])
def test_verdict_counts_only_panel_alpha(drawn, opaque, padding, status, detail):
    rgba = np.zeros((cards.PAD_TOP + 100 + cards.PAD_BOTTOM, 2 * cards.PAD_X + 100, 4),
                    dtype=np.uint8)
    if padding:
        rgba[:] = 255
    panel = rgba[cards.PAD_TOP:-cards.PAD_BOTTOM, cards.PAD_X:-cards.PAD_X]
    alpha = np.zeros(10000, dtype=np.uint8)
    alpha[:drawn] = 249
    alpha[:opaque] = 250
    panel[:, :, 3] = alpha.reshape(100, 100)
    assert triage.verdict(rgba) == (status, detail)


@pytest.mark.parametrize("failure", [YapError("Synthetic render failed.\nSecond line."),
                                    OSError("x" * 400 + "\nSecond line."), ValueError("")])
def test_check_render_failure_continues_and_cleans_work(
    registry_add, records, tmp_path, monkeypatch, capsys, failure,
):
    renderer, _, audit, _ = registry_add
    selected = {name: records[name] for name in ("titlecard-calm", "decision-tree",
                                                "conic-progress-ring")}
    original = copy.deepcopy(catalog.items())
    calls = []

    def render(plan, work_dir, out, *, install):
        assert install is renderer and plan.frames == 90
        name = plan.template.removeprefix("hf:")
        assert catalog.item(name)["status"] == "on"
        work_dir.mkdir(parents=True)
        (work_dir / "partial.html").write_text("synthetic")
        out.write_bytes(b"partial")
        calls.append((name, work_dir, out))
        if name == "conic-progress-ring":
            raise failure
        assert not calls[0][1].exists() and not calls[0][2].exists()
        raise YapError("Next item attempted.")

    monkeypatch.setattr(cards, "render_card", render)
    result = triage.check(selected, install=renderer, work=tmp_path / "checks")
    message = failure.message if isinstance(failure, YapError) else str(failure)
    assert result is selected
    assert selected["conic-progress-ring"]["check"] == {
        "status": "fail", "detail": message.split("\n", 1)[0][:300]}
    assert selected["titlecard-calm"]["check"] == {"status": "fail",
                                                  "detail": "Next item attempted."}
    assert selected["decision-tree"] == records["decision-tree"]
    assert [call[0] for call in calls] == ["conic-progress-ring", "titlecard-calm"]
    assert all(not folder.exists() and not movie.exists() for _, folder, movie in calls)
    assert len(audit.read_text().splitlines()) == 2
    assert catalog.items() == original
    assert capsys.readouterr().err.splitlines() == ["conic-progress-ring card fail",
                                                  "titlecard-calm title fail"]


def test_check_install_failure_continues(registry_add, records, tmp_path, monkeypatch):
    renderer, _, audit, control = registry_add
    control.write_text(json.dumps({"name": "conic-progress-ring", "mode": "error"}))
    selected = {name: records[name] for name in ("conic-progress-ring", "titlecard-calm")}

    def render(*args, **kwargs):
        raise YapError("Next item attempted.")

    monkeypatch.setattr(cards, "render_card", render)
    triage.check(selected, install=renderer, work=tmp_path / "checks")
    assert "synthetic add failed" in selected["conic-progress-ring"]["check"]["detail"]
    assert selected["titlecard-calm"]["check"]["detail"] == "Next item attempted."
    assert [json.loads(line)["name"] for line in audit.read_text().splitlines()] == [
        "conic-progress-ring", "titlecard-calm"]


def test_check_helper_localizes_offline_and_reuses_installed_item(synthetic, tmp_path, monkeypatch):
    _, registry_install, _, renderer, add, audit = synthetic
    source = "<script>window.sampleHelper = () => {};</script>"
    main = add("sample-helper", source=source)
    record = classify("sample-helper", source=source)
    record["entry"]["main"] = main
    record["entry"]["files"] = catalog.item("sample-helper")["files"]
    calls = []
    original = localize.localize

    def localized(name, dest, *, install):
        result = original(name, dest, install=install)
        assert "window.sampleHelper" in (dest / result.main).read_text()
        calls.append(dest)
        return result

    monkeypatch.setattr(localize, "localize", localized)
    with catalog.trial({"sample-helper": {**record["entry"], "status": "helper"}}):
        registry_install.install_items(["sample-helper"], install=renderer)
    selected = {"sample-helper": record}
    assert triage.check(selected, install=renderer, work=tmp_path / "checks") is selected
    assert record["check"] == {"status": "pass", "detail": "Its script runs offline."}
    assert len(calls) == 1 and not calls[0].exists()
    assert len(audit.read_text().splitlines()) == 1


@pytest.mark.parametrize("keep_still", [False, True])
def test_check_reads_selected_frame_and_keeps_only_requested_stills(
    registry_add, records, tmp_path, monkeypatch, capsys, keep_still,
):
    renderer, _, audit, _ = registry_add
    selected = {name: records[name] for name in ("caption-kinetic-slam", "titlecard-calm")}
    drafts = {name: {**record["entry"], "status": "on"} for name, record in selected.items()}
    with catalog.trial(drafts):
        install.install_items(list(drafts), install=renderer)
    calls = []

    def render(plan, work_dir, out, *, install):
        assert install is renderer and plan.frames == 90
        name = plan.template.removeprefix("hf:")
        entry = selected[name]["entry"]
        assert plan.variables == {key: value.format_map(entry["preview"])
                                  for key, value in entry["variables"].items()}
        work_dir.mkdir(parents=True)
        out.write_bytes(b"synthetic movie")
        calls.append((plan, work_dir, out))

    def extract(movie, still, frame):
        assert movie == calls[-1][2] and frame == 54
        still.write_bytes(b"synthetic still")

    def read_pixels(args, *, timeout_s, binary):
        assert binary and timeout_s > 0
        assert args[args.index("-pix_fmt") + 1] == "rgba"
        assert args[args.index("-f") + 1] == "rawvideo" and args[-1] == "pipe:1"
        plan = calls[-1][0]
        rgba = np.zeros((plan.height, plan.width, 4), dtype=np.uint8)
        rgba[cards.PAD_TOP:cards.PAD_TOP + 20, cards.PAD_X:cards.PAD_X + 20, 3] = 255
        return SimpleNamespace(stdout=rgba.tobytes())

    from yap_studio import templates_command

    monkeypatch.setattr(cards, "render_card", render)
    monkeypatch.setattr(templates_command, "_extract_still", extract)
    monkeypatch.setattr(ffmpeg, "run", read_pixels)
    work = tmp_path / "checks"
    stills = tmp_path / "stills" if keep_still else None
    triage.check(selected, install=renderer, work=work, stills=stills)
    assert all(record["check"]["status"] == "pass" for record in selected.values())
    assert len(audit.read_text().splitlines()) == 2
    assert len(calls) == 2 and not list(work.iterdir())
    if stills:
        assert sorted(file.name for file in stills.iterdir()) == [name + ".png" for name in drafts]
    assert capsys.readouterr().err.splitlines() == ["caption-kinetic-slam caption pass",
                                                  "titlecard-calm title pass"]


@pytest.fixture
def triage_script(monkeypatch):
    source = Path(__file__).parents[1] / "scripts" / "triage_registry.py"
    spec = importlib.util.spec_from_file_location("triage_registry_fixture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(source)])
    return module


def test_script_writes_valid_document_and_summary(
    triage_script, triage_path, triage_document, tmp_path, monkeypatch, capsys,
):
    script = triage_script
    records = triage_document["items"]
    renderer = SimpleNamespace(executable=tmp_path / "renderer")
    calls = []

    def ensure(*, notice):
        notice("Synthetic renderer ready.")
        return renderer

    def survey(url, *, executable, work, home):
        assert url == catalog.registry_url() and executable == renderer.executable
        assert work.is_dir() and home == install.registry_home() and home.is_dir()
        calls.append(work)
        return records

    def check(items, *, install, work, stills):
        assert items is records and install is renderer and work == calls[0]
        assert stills == tmp_path / "stills"
        items["titlecard-calm"]["check"] = {"status": "pass", "detail": "Synthetic pass."}
        items["conic-progress-ring"]["check"] = {"status": "fail", "detail": "Synthetic fail."}
        return items

    monkeypatch.setattr(script, "TRIAGE", triage_path)
    monkeypatch.setattr(script.hyperframes, "ensure_installed", ensure)
    monkeypatch.setattr(script, "survey", survey)
    monkeypatch.setattr(script, "check", check)
    monkeypatch.setattr(sys, "argv", ["triage_registry.py", "--stills", str(tmp_path / "stills")])
    assert script.main() == 0
    assert catalog.load_triage() == triage_document
    output = capsys.readouterr()
    summary = json.loads(output.out)
    assert summary.keys() == {"items", "usable", "passed", "failed", "seconds"}
    assert summary["items"] == len(records) and summary["usable"] == 4
    assert summary["passed"] == summary["failed"] == 1 and summary["seconds"] >= 0
    assert output.err == "Synthetic renderer ready.\n"
    assert not calls[0].exists()


def test_script_reports_errors_without_overwriting(triage_script, triage_path, monkeypatch, capsys):
    triage_path.write_text("Earlier triage.")

    def ensure(**kwargs):
        raise DependencyError("Synthetic renderer unavailable.", suggested_action="Try again.")

    monkeypatch.setattr(triage_script, "TRIAGE", triage_path)
    monkeypatch.setattr(triage_script.hyperframes, "ensure_installed", ensure)
    assert triage_script.main() == 1
    assert triage_path.read_text() == "Earlier triage."
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "Synthetic renderer unavailable.\nTry again.\n"
