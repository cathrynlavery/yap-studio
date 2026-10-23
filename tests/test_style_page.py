"""The style page stays synchronized with engine styles and bundled fonts."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from yap_studio.style import SHIPPED_PRESETS
from yap_studio.style.loader import resolve_style_data
from yap_studio.taste.learn import UNMEASURED, _proposed_style

ROOT = Path(__file__).parents[1]
PAGE = ROOT / "docs" / "design" / "style-page"
BUNDLED_FONTS = {
    "Instrument Serif",
    "Gelasio",
    "Inter",
    "Geist",
    "Alata",
    "Archivo Black",
    "Anton SC",
    "Titan One",
    "Permanent Marker",
    "Space Mono",
}


def test_generated_style_data_is_current(tmp_path: Path) -> None:
    style_dir = tmp_path / "style"
    shutil.copytree(ROOT / "src/yap_studio/style", style_dir)
    schema_file = style_dir / "schema-v1.json"
    schema = json.loads(schema_file.read_text())
    schema_file.write_text(json.dumps(schema))
    result = subprocess.run(
        [sys.executable, str(PAGE / "sync_data.py"), str(style_dir), "--check"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "matches the source files" in result.stdout


def test_page_reads_learned_broll_without_rendering(tmp_path: Path) -> None:
    broll = {
        "cutout": "screen",
        "green_screen_layout": "full",
        "screen_recording_placement": "overlay",
        "clip_placement": "overlay",
        "photo_placement": "full_frame",
        "screenshot_placement": "full_frame",
        "overlay_box": "side_right",
        "overlay_motion": "none",
        "tall_framing": "auto",
        "still_push": "in",
        "still_push_scale": 1.08,
        "avoid_tags": ["stock", "busy_background"],
        "stock_clips": "off",
        "generated_pictures": "off",
        "emphasis_on": [],
        "color_match": "on",
    }
    proposed = _proposed_style(
        {"captions.max_words_per_phrase": {"status": "proposed", "value": 5}},
        resolve_style_data({"schema_version": 1, "broll": broll}),
        {"kind": "schema_defaults", "name": "Schema defaults"},
        list(UNMEASURED),
    )
    assert proposed["style"]["broll"] == broll
    for key in broll:
        assert proposed["sources"][f"broll.{key}"] == {
            "source": "base",
            "status": None,
            "reason": "Timeline choices are not measured from pixels.",
        }
    # The page lane regenerates the style page prototype's data copy
    # when the option gets a screen. Remove this exclusion then.
    proposed["style"]["broll"].pop("generated_pictures")
    proposed["sources"].pop("broll.generated_pictures")
    measurements = tmp_path / "measurements.json"
    measurements.write_text(json.dumps({
        "schema": "yap-studio/taste-measurements@1", "proposed_style": proposed,
    }), encoding="utf-8")
    # The same Node reader command used by test_taste_learn._assert_final.
    script = """
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const [page, measurements] = process.argv.slice(1);
const context = {window: {}};
vm.runInNewContext(fs.readFileSync(path.join(page, 'style-data.js'), 'utf8'), context);
const {readLearned} = require(path.join(page, 'learned.js'));
const result = readLearned(JSON.parse(fs.readFileSync(measurements, 'utf8')),
                           context.window.YAP_STYLE_DATA.schema);
console.log(JSON.stringify({measured: result.rows.measured.map(row => row.field),
                           broll_emphasis: result.rows.base.find(
                             row => row.field === 'broll.emphasis_on'),
                           font_reason: result.rows.base.find(
                             row => row.field === 'captions.font').reason}));
"""
    opened = subprocess.run(
        ["node", "-e", script, str(PAGE), str(measurements)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False,
    )
    assert opened.returncode == 0, opened.stderr
    reviewed = json.loads(opened.stdout)
    assert "captions.max_words_per_phrase" in reviewed["measured"]
    assert reviewed["font_reason"] == proposed["sources"]["captions.font"]["reason"]
    assert reviewed["broll_emphasis"]["value"] == []
    assert reviewed["broll_emphasis"]["label"] == "broll.emphasis_on"


def test_page_font_choices_equal_bundled_set() -> None:
    app = (PAGE / "app.js").read_text(encoding="utf-8")
    table = re.search(r"const FONTS = \{(.*?)\n  \};", app, flags=re.DOTALL)
    assert table is not None
    keys = re.findall(r"^\s*(?:'([^']+)'|([A-Za-z_]+)):\s*\{", table[1], flags=re.MULTILINE)
    assert {quoted or bare for quoted, bare in keys} == BUNDLED_FONTS
    assert "expected a color like #1A2B3C" in app


def test_page_has_separate_captions_and_graphics_font_choices() -> None:
    app = (PAGE / "app.js").read_text(encoding="utf-8")
    assert "function withCaptionsFont(style, font)" in app
    assert "function withGraphicsFont(style, font)" in app
    assert "segmented('Fonts'," in app
    assert "Same font for everything" in app
    assert "Different font for graphics" in app
    assert "Instrument Serif captions, Geist titles and cards" in app


def test_page_data_uses_only_bundled_fonts_and_allowlisted_shipped_presets() -> None:
    text = (PAGE / "style-data.js").read_text(encoding="utf-8")
    data = json.loads(text.split("window.YAP_STYLE_DATA = ", 1)[1].removesuffix(";\n"))
    assert {preset["id"] for preset in data["presets"] if not preset["proposed"]} == set(
        SHIPPED_PRESETS
    )
    for preset in data["presets"]:
        for section in ("captions", "hook", "graphics"):
            font = preset["style"].get(section, {}).get("font")
            if font is not None:
                assert font in BUNDLED_FONTS


def test_page_loads_looks_before_app_as_a_deferred_classic_script() -> None:
    html = (PAGE / "index.html").read_text(encoding="utf-8")
    scripts = re.findall(r"<script\b[^>]*>", html)
    looks = next(tag for tag in scripts if 'src="looks.js"' in tag)
    app = next(tag for tag in scripts if 'src="app.js"' in tag)
    assert "defer" in looks
    assert 'type="module"' not in looks
    assert scripts.index(looks) < scripts.index(app)


def test_page_highlight_choices_come_from_schema_helpers() -> None:
    app = (PAGE / "app.js").read_text(encoding="utf-8")
    assert "window.YapLooks.highlightLooks(SCHEMA)" in app
    assert "const KINDS" not in app
    assert re.search(r"const\s+\w+\s*=\s*\[\s*\{\s*id:\s*['\"]moving_block", app) is None


def test_preview_caption_ink_matches_engine_for_imports_and_brand_accents() -> None:
    sources = [
        {
            "schema_version": 1,
            "brand": {"accent_color": "#777777"} if branded else {},
            "captions": {
                "highlight_style": highlight,
                "highlight_color": surface,
                "highlight_text_color": requested,
                "background_mode": mode,
                "background_color": surface,
                "background_opacity": opacity,
                "text_color": requested,
            },
        }
        for highlight in ("none", "highlight", "pill_karaoke")
        for surface, requested in (
            ("#777777", "#111111"),  # Both candidates miss 4.5; use pure black.
            ("#888888", "#FFFFFF"),
            ("#FFFFFF", "#333333"),  # Acceptable requested ink is not always the best.
            ("#FFFFFF", "#000000"),  # Preserve requested pure black when it wins.
            ("#000000", "#FFFFFF"),
        )
        for mode, opacity in (("none", 1), ("box", 0), ("box", 1), ("band", 0.01))
        for branded in (False, True)
    ]
    script = r"""
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const page = process.argv[1];
const sandbox = {window: {YapLooks: require(path.join(page, 'looks.js'))},
  document: {querySelector: () => null}, matchMedia: () => ({matches: true})};
vm.runInNewContext(fs.readFileSync(path.join(page, 'style-data.js'), 'utf8'), sandbox);
const app = fs.readFileSync(path.join(page, 'app.js'), 'utf8').replace(
  /\n  start\(\);\n\}\)\(\);\s*$/,
  '\n  window.resolvePreview = style => effectiveStyle(merge(style, SCHEMA));\n})();');
vm.runInNewContext(app, sandbox);
const sources = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(sources.map(source => {
  const before = JSON.stringify(source);
  const {text_color, highlight_text_color} = sandbox.window.resolvePreview(source).captions;
  if (JSON.stringify(source) !== before) throw new Error('Preview mutated the source style');
  return {text_color, highlight_text_color};
})));
"""
    result = subprocess.run(
        ["node", "-e", script, str(PAGE)], input=json.dumps(sources),
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    for source, preview in zip(sources, json.loads(result.stdout), strict=True):
        captions = resolve_style_data(source)["captions"]
        assert preview == {key: captions[key] for key in preview}, source


@pytest.mark.parametrize("helper,sections", [
    ("withFont", ("captions", "hook", "graphics")),
    ("withCaptionsFont", ("captions",)),
    ("withGraphicsFont", ("hook", "graphics")),
])
def test_page_font_helpers_follow_the_prototype_brand_rule(
    helper: str, sections: tuple[str, ...],
) -> None:
    sources = [{
        "name": "Synthetic look",
        "captions": {"font": "Inter", "letter_spacing_px": 2, "width_scale": 0.85},
        "hook": {"font": "Gelasio", "style": "quiet_serif"},
        "graphics": {"font": "Alata", "style": "paper"},
        "brand": {"captions_font": "Creator Serif" if branded else "",
                  "titles_font": "Creator Sans" if branded else "",
                  "cards_font": "Creator Cards" if branded else "", "accent_color": "#123456"},
    } for branded in (True, False)]
    script = r"""
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const [page, helper] = process.argv.slice(1);
const sandbox = {window: {YapLooks: require(path.join(page, 'looks.js'))},
  document: {querySelector: () => null}, matchMedia: () => ({matches: true})};
vm.runInNewContext(fs.readFileSync(path.join(page, 'style-data.js'), 'utf8'), sandbox);
const app = fs.readFileSync(path.join(page, 'app.js'), 'utf8').replace(
  /\n  start\(\);\n\}\)\(\);\s*$/,
  '\n  window.fontHelpers = {withFont, withCaptionsFont, withGraphicsFont};\n})();');
vm.runInNewContext(app, sandbox);
const sources = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(sources.map(source => {
  const before = JSON.stringify(source);
  const result = sandbox.window.fontHelpers[helper](source, 'Geist');
  if (JSON.stringify(source) !== before) throw new Error('Font helper mutated the source style');
  return result;
})));
"""
    result = subprocess.run(
        ["node", "-e", script, str(PAGE), helper], input=json.dumps(sources),
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    brand_fields = {"captions": "captions_font", "hook": "titles_font", "graphics": "cards_font"}
    for source, changed in zip(sources, json.loads(result.stdout), strict=True):
        expected = deepcopy(source)
        for section in sections:
            field = brand_fields[section]
            if expected["brand"][field]:
                expected["brand"][field] = "Geist"
            else:
                expected[section]["font"] = "Geist"
        if "captions" in sections:
            expected["captions"]["letter_spacing_px"] = -0.5
        assert changed == expected


def test_page_letter_width_belongs_to_the_font_group() -> None:
    app = (PAGE / "app.js").read_text(encoding="utf-8")
    groups = re.search(r"const GROUP_PATHS = \{(.*?)\n  \};", app, flags=re.DOTALL)
    assert groups is not None
    font = re.search(r"font:\s*\[(.*?)\]", groups[1], flags=re.DOTALL)
    assert font is not None
    assert "'captions.width_scale'" in font[1]
    assert "window.YapLooks.widthChoices(SCHEMA)" in app
    assert "window.YapLooks.nearestWidth" in app
