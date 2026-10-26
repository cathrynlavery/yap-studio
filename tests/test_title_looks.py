"""Registry title look plans and opt-in real renderer coverage."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from yap_studio import ffmpeg, paths, text_rules
from yap_studio.captions import CaptionWord, kinetic_display_times, kinetic_word_times
from yap_studio.errors import ValidationError
from yap_studio.render import hyperframes
from yap_studio.render.text_metrics import text_width_em
from yap_studio.stages.captions import prepare_layer_plan, render_layer_plan, title_layer_plan
from yap_studio.style.loader import ResolvedStyle, resolve_style_data
from yap_studio.text_rules import apply_title_rules

ROOT = Path(__file__).parents[1]
TEMPLATES = ROOT / "src/yap_studio/templates"
LOOKS = {
    "calm_card": "titlecard-calm",
    "lockup": "titlecard-lockup",
    "write_on": "hw-write-title",
    "headline_slam": "headline-slam",
    "shard_reveal": "glass-shard-title",
    "textured": "texture-mask-text",
    "shimmer": "text-shimmer",
    "stagger": "text-stagger",
    "morph": "morph-text",
    "typewriter": "typewriter",
}


def _style(look: str, *, size: int = 94, position: str = "headroom") -> ResolvedStyle:
    data = resolve_style_data({
        "schema_version": 1,
        "name": "Synthetic title",
        "hook": {"style": look, "size_px": size, "duration_s": 1, "position": position},
    })
    return ResolvedStyle(data["name"], "test", data)


@pytest.mark.parametrize("plain,key,expected", [
    ("go", "go", [False, True]),
    ("going", "go", [False, True]),
    ("Yap 25", "yap twenty five", [False, False, True, True]),
    ("go go go go", "go go", [False] * 4 + [True] * 2),
    ("go", "", [False]),
])
def test_kinetic_plan_marks_only_the_rewritten_key_words(plain, key, expected) -> None:
    vocabulary = ["Yap"]
    plan = title_layer_plan(text_rules.apply_text_rules(plain, vocabulary),
                            text_rules.apply_text_rules(key, vocabulary), _style("kinetic"))
    display = json.loads(plan.variables["kinetic_lines"])
    assert [accent for line in display for accent in line["accent"]] == expected
    assert all(len(line["accent"]) == len(line["text"].split()) for line in display)


@pytest.mark.parametrize("look", LOOKS)
@pytest.mark.parametrize("lines", [
    ["Make it clear"],
    ["A better opening", "starts right here"],
    ["A better opening", "starts with one clear", "idea for everyone"],
])
def test_title_look_plan_fits_vertical_frame(look: str, lines: list[str]) -> None:
    plan = title_layer_plan("", "clear", _style(look), lines=lines)
    assert plan.width <= 1080
    assert 0 <= plan.x and plan.x + plan.width <= 1080
    assert 0 <= plan.y and plan.y + plan.height <= 1920
    assert plan.variables["lines"] == json.dumps(lines)
    assert plan.variables["font_family"]
    assert plan.variables["accent_color"].startswith("#")


@pytest.mark.parametrize("look", LOOKS)
def test_title_look_center_and_tall_refusal(look: str) -> None:
    plan = title_layer_plan("Short", "title", _style(look, position="center"))
    assert 0 <= plan.y and plan.y + plan.height <= 1920
    with pytest.raises(ValidationError, match="too tall"):
        title_layer_plan("long " * 150, "", _style(look, size=180))


def test_title_look_sources_are_local_and_attributed() -> None:
    html = (TEMPLATES / "title/template.html").read_text()
    metadata = json.loads((TEMPLATES / "title/template.json").read_text())
    provenance = json.loads((TEMPLATES / "PROVENANCE.json").read_text())
    header = html.split("<!--", 1)[1].split("-->", 1)[0]
    assert "http://" not in html and "https://" not in html
    assert {option["value"] for option in next(
        item for item in metadata["variables"] if item["id"] == "look"
    )["options"]} >= set(LOOKS)
    for look, item in LOOKS.items():
        entries = [entry for entry in provenance["files"] if entry["item"] == item
                   and entry["yap_path"] == "src/yap_studio/templates/title/template.html"]
        assert len(entries) == 1, look
        entry = entries[0]
        assert {"item": item, "path": entry["upstream_path"]} in metadata["derived_from"]
        assert (f'Derived from HyperFrames registry item "{item}"\n'
                f'  ({entry["upstream_path"]} at commit {provenance["upstream"]["commit"]})'
                in header)


@pytest.mark.parametrize("lines", [
    ["Make it clear"], ["A better opening", "starts right here"],
    ["A better opening", "starts with one clear", "idea for everyone"],
])
@pytest.mark.parametrize("position", ["headroom", "top", "center"])
def test_kinetic_plan_fits_and_refuses_tall_titles(lines: list[str], position: str) -> None:
    style = _style("kinetic", position=position)
    plan = title_layer_plan("", "clear", style, lines=lines)
    assert plan.width == 1080 and plan.x == 0
    assert 0 <= plan.y and plan.y + plan.height <= 1920
    assert plan.variables["word_times"] == "[]"
    with pytest.raises(ValidationError, match="shorten the kinetic title"):
        title_layer_plan("long " * 150, "", style)


@pytest.mark.parametrize("source,expected", [
    (["Go"], ["Go"]), (["Go make"], ["Go make"]),
    (["Go make it"], ["Go make it"]), (["Go make it clear"], ["Go make it clear"]),
    (["we can do it right now"], ["we can do", "it right now"]),
    (["we can do it yes we can go"], ["we can do it", "yes we can go"]),
    (["we can do", "it"], ["we can do it"]),
])
def test_kinetic_display_line_breaks(source: list[str], expected: list[str]) -> None:
    plan = title_layer_plan("", "", _style("kinetic"), lines=source)
    display = json.loads(plan.variables["kinetic_lines"])
    assert [line["text"] for line in display] == expected
    assert " ".join(line["text"] for line in display).split() == " ".join(source).split()


def test_kinetic_line_sizes_fill_width_and_ignore_hook_size() -> None:
    lines = ["Go", "Make it work"]
    small = title_layer_plan("", "", _style("kinetic", size=40), lines=lines)
    large = title_layer_plan("", "", _style("kinetic", size=180), lines=lines)
    assert small == large
    display = json.loads(small.variables["kinetic_lines"])
    assert display[0]["size_px"] == 260
    assert display[0]["size_px"] > display[1]["size_px"]
    assert all(110 <= line["size_px"] <= 260 for line in display)
    assert small.height == (sum(math.ceil(line["size_px"] * .98) for line in display)
                            + 2 * small.variables["padding_px"])
    assert abs(small.y + small.height / 2 - 520) <= .5


def test_kinetic_bundled_font_sizes_fill_planned_width() -> None:
    style = _style("kinetic")
    style.data["hook"]["font"] = "Instrument Serif"
    plan = title_layer_plan("", "", style, lines=["How many R's", "are in strawberry"])
    for line in json.loads(plan.variables["kinetic_lines"]):
        width = text_width_em(line["text"], "Instrument Serif", 400) * line["size_px"]
        assert .9 * 940 <= width <= 940


def test_kinetic_unbundled_font_sizes_stay_within_planned_width() -> None:
    family = "Creator Display"
    metrics = json.loads((ROOT / "src/yap_studio/render/text_metrics.json").read_text())
    assert family not in metrics["fonts"]
    style = _style("kinetic")
    style.data["hook"]["font"] = family
    plan = title_layer_plan("", "", style, lines=["Go", "Go now"])
    display = json.loads(plan.variables["kinetic_lines"])
    assert display
    widths = [text_width_em(line["text"], family, 400) * line["size_px"] for line in display]
    assert max(widths) <= 940
    assert all(110 <= line["size_px"] <= 260 for line in display)


def test_kinetic_template_uses_synchronous_plan_sizes() -> None:
    html = (TEMPLATES / "title/template.html").read_text()
    kinetic = html.split("else if (v.look === 'kinetic') {", 1)[1].split(
        "else if (v.look === 'stagger') {", 1)[0]
    assert "copy.style.fontSize = `${size_px}px`" in kinetic
    assert "document.fonts.load" not in kinetic
    assert "__kineticReady" not in kinetic
    assert "document.fonts.ready.then(" in kinetic


@pytest.mark.parametrize("shadow_opacity", [0, .65])
def test_kinetic_padding_reserves_punch_and_caption_shadow(shadow_opacity: float) -> None:
    style = _style("kinetic")
    style.data["captions"].update(shadow_opacity=shadow_opacity, shadow_blur_px=12,
                                  shadow_x_px=-2, shadow_y_px=4)
    plan = title_layer_plan("", "", style, lines=["How many R's", "are in strawberry"])
    display = json.loads(plan.variables["kinetic_lines"])
    heights = [math.ceil(line["size_px"] * .98) for line in display]
    shadow_room = 3 * 12 + 4 if shadow_opacity else 0
    padding = math.ceil(max(heights) * .25 + shadow_room + 12)
    assert plan.variables["padding_px"] == padding
    assert plan.height == sum(heights) + 2 * padding


@pytest.mark.parametrize("lines,limit", [(["go " * 17], 16), (["W" * 100], 1)])
def test_kinetic_refuses_words_that_cannot_fit_at_floor(lines: list[str], limit: int) -> None:
    with pytest.raises(ValidationError, match=f"shorten the kinetic title to {limit} words"):
        title_layer_plan("", "", _style("kinetic"), lines=lines)


@pytest.mark.parametrize("look", ["kinetic", "quiet_serif"])
def test_title_passes_word_times_and_caption_ink(look: str) -> None:
    style = _style(look)
    style.data["captions"].update(text_color="#EEDDBB", shadow_opacity=0.65,
                                  shadow_blur_px=12, shadow_x_px=-2, shadow_y_px=4)
    plan = title_layer_plan("Small", "steps", style, word_times=[0.1, 0.5])
    assert json.loads(plan.variables["word_times"]) == [0.1, 0.5]
    assert plan.variables["ink_color"] == "#EEDDBB"
    for key in ("shadow_opacity", "shadow_blur_px", "shadow_x_px", "shadow_y_px"):
        assert plan.variables[key] == style.data["captions"][key]


def test_kinetic_template_declares_look_and_plan_variables() -> None:
    html = (TEMPLATES / "title/template.html").read_text()
    metadata = json.loads((TEMPLATES / "title/template.json").read_text())
    declared = json.loads(html.split("data-composition-variables='", 1)[1].split("'", 1)[0])
    assert declared == metadata["variables"]
    look = next(item for item in declared if item["id"] == "look")
    assert {"value": "kinetic", "label": "Kinetic hook"} in look["options"]
    assert metadata["box"]["width"]["kinetic"] == 1080
    plan = title_layer_plan("Small", "steps", _style("kinetic"))
    assert set(plan.variables) <= {item["id"] for item in declared}


@pytest.mark.parametrize("lines,texts,fixes,expected", [
    (["Small", "steps"], ["Small", "steps"], {}, [0.1, 0.5]),
    (["SMALL! steps?"], ["small,", "STEPS."], {}, [0.1, 0.5]),
    (["Small steps"], ["Small", "um", "steps"], {}, [0.1, 0.9]),
    (["Simple steps"], ["Small", "steps"], {"small": "Simple"}, [0.1, 0.5]),
    (["Start now"], ["Small", "steps"], {"small steps": "Start now"}, [0.1, 0.1]),
    (["A.P.I.2 works"], ["A.P.I.2", "works"], {}, [0.1, 0.5]),
    (["Plan & ship"], ["plan", "and", "ship"], {}, [0.1, 0.1, 0.9]),
    (["- Small steps"], ["Small", "steps"], {}, [0.1, 0.1, 0.5]),
    (["+ 🛠", "Small & steps -"], ["Small", "steps"], {},
     [0.1, 0.1, 0.1, 0.1, 0.5, 0.5]),
])
def test_kinetic_matches_display_words_in_order(lines: list[str], texts: list[str],
                                               fixes: dict, expected: list[float]) -> None:
    words = [CaptionWord(str(i), text, 1.1 + i * 0.4, 1.2 + i * 0.4)
             for i, text in enumerate(texts)]
    assert kinetic_word_times(lines, words, 1, 3, fixes) == expected


@pytest.mark.parametrize("title,spoken,expected", [
    ("Take 25 steps", "take twenty five steps", [0.1, 0.5, 1.3]),
    ("$5", "five dollars", [0.1]),
    ("50%", "fifty percent", [0.1]),
    ("2.5 times", "two point five times", [0.1, 1.3]),
    ("2026", "twenty twenty six", [0.1]),
    ("twenty five", "twenty five", [0.1, 0.5]),
    ("Take twenty five steps", "take 25 steps", [0.1, 0.5, 0.5, 0.9]),
    ("Take twenty-five steps", "take 25 steps", [0.1, 0.5, 0.9]),
    ("gonna ship", "going to ship", [0.1, 0.9]),
    ("going to ship", "gonna ship", [0.1, 0.1, 0.5]),
])
def test_kinetic_matches_normalized_title_tokens(
    title: str, spoken: str, expected: list[float],
) -> None:
    words = [CaptionWord(str(i), text, 1.1 + i * 0.4, 1.2 + i * 0.4)
             for i, text in enumerate(spoken.split())]
    assert kinetic_word_times([title], words, 1, 3, {}) == expected


def test_kinetic_rules_match_the_whole_written_title() -> None:
    words = [CaptionWord("0", "25", 1.1, 1.2)]
    assert kinetic_word_times(["twenty", "five"], words, 1, 3, {}) == [0.1, 0.1]


def test_kinetic_rules_follow_brand_source_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    def aliases(vocabulary):
        assert vocabulary == ["SandPiper"]
        return {"sand piper": "SandPiper"}

    monkeypatch.setattr(text_rules, "vocabulary_fixes", aliases)
    words = [CaptionWord(str(i), text, 1.1 + i * 0.4, 1.2 + i * 0.4)
             for i, text in enumerate("Use sand piper now".split())]
    assert kinetic_word_times(["Use SandPiper now"], words, 1, 3, {},
                              vocabulary=["SandPiper"]) == [0.1, 0.5, 1.3]


@pytest.mark.parametrize("title,spoken,missing", [
    ("2026 missing", "twenty twenty six", "2026"),
    ("API2 works", "A.P.I.2 works", "API2"),
])
def test_kinetic_rules_failure_preserves_first_raw_refusal(
    title: str, spoken: str, missing: str,
) -> None:
    words = [CaptionWord(str(i), text, 1.1 + i * 0.4, 1.2 + i * 0.4)
             for i, text in enumerate(spoken.split())]
    with pytest.raises(ValidationError) as caught:
        kinetic_word_times([title], words, 1, 3, {}, element_id="agent-title")
    assert caught.value.parameter == "title.agent-title.kinetic"
    assert caught.value.detail == (f'"{missing}" is not spoken between 1 s and 3 s; anchor the '
                                   'title where the phrase is said or use the spoken words')


@pytest.mark.parametrize("spoken", ["take twenty six steps", "take twenty um five steps"])
def test_kinetic_refuses_unspoken_or_interrupted_number(spoken: str) -> None:
    words = [CaptionWord(str(i), text, 1.1 + i * 0.3, 1.2 + i * 0.3)
             for i, text in enumerate(spoken.split())]
    with pytest.raises(ValidationError) as caught:
        kinetic_word_times(["Take 25 steps"], words, 1, 3, {}, element_id="agent-title")
    assert caught.value.parameter == "title.agent-title.kinetic"
    assert caught.value.detail == ('"25" is not spoken between 1 s and 3 s; anchor the '
                                   'title where the phrase is said or use the spoken words')


@pytest.mark.parametrize("lines", [["&"], ["- +", "🛠"]])
def test_kinetic_refuses_title_without_spoken_words(lines: list[str]) -> None:
    words = [CaptionWord("0", "Small", 1.1, 1.2)]
    with pytest.raises(ValidationError) as caught:
        kinetic_word_times(lines, words, 1, 3, {}, element_id="agent-title")
    assert caught.value.parameter == "title.agent-title.kinetic"
    assert caught.value.detail == "use spoken words in the kinetic title"


def test_kinetic_matches_only_starts_inside_window_and_snaps_frames() -> None:
    words = [CaptionWord("0", "Small", 0.98, 1.1),
             CaptionWord("1", "Small", 1.114, 1.2),
             CaptionWord("2", "steps", 1.519, 1.6),
             CaptionWord("3", "outside", 3, 3.1)]
    assert kinetic_word_times(["Small steps"], words, 1, 3, {}) == [3 / 30, 16 / 30]
    with pytest.raises(ValidationError, match='"outside" is not spoken'):
        kinetic_word_times(["outside"], words, 1, 3, {})


@pytest.mark.parametrize("start_s", [7 / 30, 0.249])
@pytest.mark.parametrize("title,spoken", [
    ("Small steps", [("Small", 0.22), ("steps", 0.6)]),
    ("25 steps", [("twenty", 0.22), ("five", 0.4), ("steps", 0.6)]),
])
def test_kinetic_keeps_words_that_snap_to_the_window_start(
    title: str, spoken: list[tuple[str, float]], start_s: float,
) -> None:
    words = [CaptionWord(str(i), text, start, start + 0.1)
             for i, (text, start) in enumerate(spoken)]
    assert kinetic_word_times([title], words, start_s, 2, {}) == [0, 11 / 30]


@pytest.mark.parametrize("start", [0.21, 0.99])
def test_kinetic_excludes_words_outside_the_window_frames(start: float) -> None:
    words = [CaptionWord("0", "Outside", start, start + 0.1)]
    with pytest.raises(ValidationError, match='"Outside" is not spoken'):
        kinetic_word_times(["Outside"], words, 7 / 30, 1, {})


def test_kinetic_refuses_missing_or_reordered_words_with_fix() -> None:
    words = [CaptionWord("0", "steps", 1.1, 1.2), CaptionWord("1", "Small", 1.5, 1.6)]
    with pytest.raises(ValidationError) as caught:
        kinetic_word_times(["Small steps"], words, 1, 3, {}, element_id="agent-title")
    assert caught.value.parameter == "title.agent-title.kinetic"
    assert caught.value.detail == ('"steps" is not spoken between 1 s and 3 s; anchor the '
                                   'title where the phrase is said or use the spoken words')


@pytest.mark.parametrize("end", [0.5, 2])
def test_kinetic_refuses_last_word_without_reading_hold(end: float) -> None:
    words = [CaptionWord("0", "Late", end - 0.4, end - 0.3)]
    with pytest.raises(ValidationError, match="raise hold_s or end the title later"):
        kinetic_word_times(["Late"], words, 0, end, {})


@pytest.mark.parametrize(("lines", "times", "vocabulary", "shown", "expected"), [
    pytest.param(["Small clear steps"], [0.1, 0.314159, 0.9], [],
                 ["Small clear steps"], [0.1, 0.314159, 0.9], id="unchanged"),
    pytest.param(["  Small\tclear   steps! "], [0, 0.2, 0.9], [],
                 ["Small clear steps!"], [0, 0.2, 0.9], id="whitespace"),
    pytest.param(["sandpiper works"], [0.1, 0.6], ["SandPiper"],
                 ["SandPiper works"], [0.1, 0.6], id="brand-case"),
    pytest.param(["sand piper works"], [0.1, 0.4, 0.8], ["Sand Piper"],
                 ["Sand Piper works"], [0.1, 0.4, 0.8], id="brand-same-word-count"),
    pytest.param(["twenty five steps"], [0.1, 0.3, 0.7], [],
                 ["25 steps"], [0.1, 0.7], id="number"),
    pytest.param(["save fifty percent"], [0.1, 0.3, 0.7], [],
                 ["save 50%"], [0.1, 0.3], id="percent"),
    pytest.param(["twenty five", "fifty percent", "sandpiper works"],
                 [0.1, 0.2, 0.4, 0.6, 0.8, 1.0], ["SandPiper"],
                 ["25", "50%", "SandPiper works"], [0.1, 0.4, 0.8, 1.0], id="line-offsets"),
    pytest.param(["sand piper", "twenty five steps"], [0.1, 0.3, 0.6, 0.8, 1.0],
                 ["Sand Piper"], ["Sand Piper", "25 steps"], [0.1, 0.3, 0.6, 1.0],
                 id="brand-and-number-lines"),
    pytest.param(["", "twenty five", " \t", "Small steps", ""], [0.1, 0.2, 0.5, 0.8], [],
                 ["", "25", "", "Small steps", ""], [0.1, 0.5, 0.8], id="blank-lines"),
    pytest.param([], [], [], [], [], id="no-lines"),
    pytest.param(["", " \t"], [], [], ["", ""], [], id="no-words"),
])
def test_kinetic_display_times_follow_shown_words(
    lines: list[str], times: list[float], vocabulary: list[str],
    shown: list[str], expected: list[float],
) -> None:
    original_times = list(times)
    converted, _ = apply_title_rules(lines, "", vocabulary)
    assert converted == shown
    actual = kinetic_display_times(lines, times, vocabulary)
    assert actual == expected
    assert len(actual) == sum(len(line.split()) for line in converted)
    assert times == original_times


@pytest.mark.parametrize(("spoken", "spelling", "times", "shown", "expected"), [
    ("sand piper", "SandPiper", [0.1, 0.3, 0.6, 0.9],
     "Use SandPiper now", [0.1, 0.3, 0.9]),
    ("sandpiper", "Sand Piper", [0.1, 0.3, 0.9],
     "Use Sand Piper now", [0.1, 0.3, 0.3, 0.9]),
])
def test_kinetic_display_times_follow_brand_merge_and_split(
    monkeypatch: pytest.MonkeyPatch, spoken: str, spelling: str, times: list[float],
    shown: str, expected: list[float],
) -> None:
    def alias_fixes(vocabulary: list[str]) -> dict[str, str]:
        assert vocabulary == [spelling]
        return {spoken: spelling}

    # Supply aliases at the rewrite boundary; spelling lookup currently only changes case.
    monkeypatch.setattr(text_rules, "vocabulary_fixes", alias_fixes)
    lines = [f"Use {spoken} now"]
    converted, _ = apply_title_rules(lines, "", [spelling])
    assert converted == [shown]
    actual = kinetic_display_times(lines, times, [spelling])
    assert actual == expected
    assert len(actual) == sum(len(line.split()) for line in converted)


def _browser_title(tmp_path: Path, look: str, lines: list[str] | None, *, plain: str = "",
                   accent: str = "", frames: int = 30, size: int = 70,
                   word_times: list[float] | None = None, font: str | None = None,
                   css: str = "") -> dict:
    style = _style(look, size=size)
    if font:
        style.data["hook"]["font"] = font
    plan = title_layer_plan(plain, accent, style, lines=lines, frames=frames,
                            word_times=word_times)
    composition = prepare_layer_plan(plan, tmp_path / look)
    if css:
        html = composition / "index.html"
        html.write_text(html.read_text().replace("</style>", f"{css}</style>", 1))
    install = hyperframes.ensure_installed()
    script = r"""
const fs = require('fs');
const { pathToFileURL } = require('url');
const puppeteer = require(process.argv[1]);
(async () => {
  const input = JSON.parse(fs.readFileSync(0, 'utf8'));
  const browser = await puppeteer.launch({executablePath: input.browser,
    args: ['--no-sandbox', '--disable-dev-shm-usage'], headless: true});
  try {
    const page = await browser.newPage();
    await page.setViewport({width: input.width, height: input.height});
    await page.evaluateOnNewDocument(vars => {
      window.__hyperframes = {getVariables: () => vars};
    }, input.variables);
    await page.goto(pathToFileURL(input.html).href);
    await page.evaluate(() => document.fonts.ready);
    const result = await page.evaluate(() => {
      const root = document.querySelector('[data-composition-id]');
      const hook = root.querySelector('.hook');
      const title = root.querySelector('.hook-title');
      const card = root.querySelector('.hook-card');
      const exit = Math.min(250, Number(root.dataset.duration) * 500);
      const fadeStart = Number(root.dataset.duration) * 1000 - exit;
      const frameMs = 1000 / Number(window.__hyperframes.getVariables().fps);
      const lastPreFadeFrame = Math.floor((fadeStart - .001) / frameMs) * frameMs;
      const entrance = [...hook.getAnimations({subtree: true})]
        .filter(animation => animation.effect.target !== hook &&
          !animation.effect.getKeyframes().some(frame => 'backgroundPosition' in frame) &&
          !animation.effect.target.classList.contains('shimmer-sweep'))
        .map(animation => ({target: animation.effect.target.className,
          end: animation.effect.getTiming().delay + animation.effect.getTiming().duration}));
      for (const animation of hook.getAnimations({subtree: true})) {
        animation.pause();
        animation.currentTime = lastPreFadeFrame;
      }
      const rect = node => {const r = node.getBoundingClientRect();
        return {left: r.left, right: r.right, top: r.top, bottom: r.bottom};};
      const words = [...title.querySelectorAll('.hook-word')];
      const wordAnimation = words[0]?.getAnimations()[0];
      const wordSamples = words.map((word, index) => {
        const time = JSON.parse(window.__hyperframes.getVariables().word_times)[index];
        const sample = ms => {
          for (const animation of hook.getAnimations({subtree: true})) animation.currentTime = ms;
          return {word: Number(getComputedStyle(word).opacity),
            hook: Number(getComputedStyle(hook).opacity), box: rect(word)};
        };
        return {final: sample(lastPreFadeFrame).word,
          ...(Number.isFinite(time) ? {
            before: sample(time * 1000 - frameMs), at: sample(time * 1000),
            after: sample(time * 1000 + 5 * frameMs)} : {})};
      });
      for (const animation of hook.getAnimations({subtree: true}))
        animation.currentTime = lastPreFadeFrame;
      const planned = JSON.parse(window.__hyperframes.getVariables().kinetic_lines);
      return {text: title.textContent,
        wordAccents: words.map(word => ({text: word.textContent,
          key: word.classList.contains('hook-key'), color: getComputedStyle(word).color})),
        kineticLines: [...title.querySelectorAll('.kinetic-line')].map((node, index) => {
          const copy = node.querySelector('.kinetic-text');
          const size = copy.style.fontSize;
          copy.style.fontSize = `${planned[index].size_px}px`;
          const plannedWidth = copy.getBoundingClientRect().width;
          copy.style.fontSize = size;
          return {...rect(node), content: rect(copy), plannedWidth,
            size: parseFloat(getComputedStyle(copy).fontSize)};
        }),
        wordSamples, wordKeyframes: wordAnimation?.effect.getKeyframes(),
        wordTiming: wordAnimation?.effect.getTiming(),
        ink: getComputedStyle(title).color, shadow: getComputedStyle(title).textShadow,
        background: getComputedStyle(card).backgroundColor,
        accent: title.querySelector('.hook-key') &&
          getComputedStyle(title.querySelector('.hook-key')).color,
        kicker: card.querySelector('.lockup-kicker')?.textContent ?? null,
        rootHeight: Number(root.dataset.height), title: rect(title), card: rect(card),
        titleScroll: title.scrollWidth,
        titleClient: title.clientWidth, cardScroll: card.scrollWidth,
        cardClient: card.clientWidth, fadeStart, lastPreFadeFrame, entrance,
        chars: [...title.querySelectorAll('.typed-char')].map(node => ({
          text: node.textContent, visible: getComputedStyle(node).visibility,
          rect: rect(node)})),
        lines: [...title.querySelectorAll('.typed-line')].map(rect)};
    });
    process.stdout.write(JSON.stringify(result));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
"""
    result = subprocess.run(
        ["node", "-e", script, str(install.prefix / "node_modules/puppeteer-core")],
        input=json.dumps({"browser": install.browser_path,
                          "width": plan.width, "height": plan.height,
                          "html": str(composition / "index.html"),
                          "variables": plan.variables}),
        text=True, capture_output=True, check=True, timeout=60,
    )
    return json.loads(result.stdout)


@pytest.mark.render
def test_kinetic_display_lines_fill_their_reserved_box(tmp_path: Path) -> None:
    lines = ["How many R's", "are in strawberry"]
    style = _style("kinetic")
    style.data["hook"]["font"] = "Instrument Serif"
    plan = title_layer_plan("", "", style, lines=lines, frames=60)
    reserved = json.loads(plan.variables["kinetic_lines"])
    result = _browser_title(tmp_path, "kinetic", lines, font="Instrument Serif", frames=60)
    assert len(result["kineticLines"]) == 2
    for line, planned in zip(result["kineticLines"], reserved, strict=True):
        width = line["content"]["right"] - line["content"]["left"]
        assert 960 * .8 <= width <= 960 + .1
        assert result["card"]["left"] <= line["content"]["left"]
        assert line["content"]["right"] <= result["card"]["right"] + .1
        assert 110 <= line["size"] <= 260
        assert line["size"] == pytest.approx(planned["size_px"])
        assert 0 <= line["top"] < line["bottom"] <= result["rootHeight"]


@pytest.mark.render
@pytest.mark.parametrize("spacing", [.1, 1])
def test_kinetic_shrinks_wide_font_without_crossing_floor(tmp_path: Path, spacing: float) -> None:
    lines = ["Small clear", "steps"]
    style = _style("kinetic")
    style.data["hook"]["font"] = "Instrument Serif"
    plan = title_layer_plan("", "", style, lines=lines)
    reserved = json.loads(plan.variables["kinetic_lines"])
    result = _browser_title(tmp_path, "kinetic", lines, font="Instrument Serif",
                            css=f".kinetic-text {{ letter-spacing: {spacing}em; }}")
    card_width = result["card"]["right"] - result["card"]["left"]
    line = result["kineticLines"][0]
    assert line["plannedWidth"] > card_width
    expected = max(110, reserved[0]["size_px"] * card_width / line["plannedWidth"])
    assert line["size"] == pytest.approx(expected, abs=.01)
    assert line["size"] < reserved[0]["size_px"]
    width = line["content"]["right"] - line["content"]["left"]
    if spacing == .1:
        assert expected > 110
        assert width <= card_width + .1
    else:
        assert line["size"] == 110
        assert width > card_width


@pytest.mark.render
@pytest.mark.parametrize("plain,key,expected", [
    ("go", "go", [False, True]),
    ("Yap 25", "yap twenty five", [False, False, True, True]),
])
def test_kinetic_render_accents_the_words_after_the_pipe(
    tmp_path: Path, plain: str, key: str, expected: list[bool],
) -> None:
    vocabulary = ["Yap"]
    result = _browser_title(tmp_path, "kinetic", None,
        plain=text_rules.apply_text_rules(plain, vocabulary),
        accent=text_rules.apply_text_rules(key, vocabulary))
    assert [word["key"] for word in result["wordAccents"]] == expected
    for word in result["wordAccents"]:
        assert word["color"] == (result["accent"] if word["key"] else result["ink"])


@pytest.mark.render
def test_kinetic_words_follow_absolute_times(tmp_path: Path) -> None:
    result = _browser_title(tmp_path, "kinetic", ["Small clear steps"], accent="clear",
                            frames=60, word_times=[0.1, 0.5, 0.9])
    keyframes = result["wordKeyframes"]
    assert [(frame["transform"], frame["filter"], float(frame["opacity"]))
            for frame in keyframes] == [
                ("scale(1.5)", "blur(8px)", 0),
                ("scale(0.93)", "blur(0px)", 1),
                ("scale(1)", "blur(0px)", 1),
            ]
    assert keyframes[1]["offset"] == .6
    assert result["wordTiming"]["duration"] == 260
    assert result["wordTiming"]["fill"] == "both"
    assert result["wordTiming"]["easing"] == "cubic-bezier(0.16, 1, 0.3, 1)"
    for word in result["wordSamples"]:
        assert word["before"]["word"] == 0
        assert word["at"]["word"] > 0
        assert word["after"]["word"] > 0.95 and word["after"]["hook"] == 1
        assert 0 <= word["at"]["box"]["top"]
        assert word["at"]["box"]["bottom"] <= result["rootHeight"]
    assert result["background"] == "rgba(0, 0, 0, 0)"
    assert result["shadow"] != "none" and result["accent"] != result["ink"]


@pytest.mark.render
@pytest.mark.parametrize("frames", [15, 30, 60])
def test_kinetic_fallback_finishes_before_fade(tmp_path: Path, frames: int) -> None:
    result = _browser_title(tmp_path, "kinetic", ["Small clear", "steps"], frames=frames)
    assert len(result["wordSamples"]) == 3
    assert all(word["final"] == 1 for word in result["wordSamples"])
    assert max(item["end"] for item in result["entrance"]) < result["lastPreFadeFrame"]
    assert result["card"]["bottom"] <= result["rootHeight"]


@pytest.mark.render
def test_lockup_project_lines_split_kicker_and_accent(tmp_path: Path) -> None:
    result = _browser_title(tmp_path, "lockup", ["Here's why AI", "did not know"],
                            accent="did not know")
    assert result["kicker"] == "Here's why AI"
    assert result["text"] == "did not know"
    assert result["card"]["bottom"] <= result["rootHeight"]


@pytest.mark.render
def test_lockup_without_accent_keeps_single_headline(tmp_path: Path) -> None:
    result = _browser_title(tmp_path, "lockup", ["One title", "without an accent"])
    assert result["kicker"] is None
    assert result["text"] == "One titlewithout an accent"


@pytest.mark.render
@pytest.mark.parametrize("lines", [
    ["SupercalifragilisticexpialidociousSupercalifragilisticexpialidocious"],
    ["This explicit timeline line is intentionally long enough to wrap across the panel"],
])
def test_typewriter_reveals_every_character_inside_card(tmp_path: Path,
                                                         lines: list[str]) -> None:
    result = _browser_title(tmp_path, "typewriter", lines, frames=60)
    assert "".join(char["text"] for char in result["chars"]) == "".join(lines)
    assert all(char["visible"] == "visible" for char in result["chars"])
    assert all(result["card"]["left"] <= char["rect"]["left"]
               and char["rect"]["right"] <= result["card"]["right"]
               for char in result["chars"])
    assert result["titleScroll"] <= result["titleClient"] + 1
    assert result["card"]["bottom"] <= result["rootHeight"]


@pytest.mark.render
def test_typewriter_finishes_before_short_layer_fades(tmp_path: Path) -> None:
    result = _browser_title(tmp_path, "typewriter", ["Two", "lines"], frames=15)
    assert "".join(char["text"] for char in result["chars"]) == "Twolines"
    assert all(char["visible"] == "visible" for char in result["chars"])
    assert max(item["end"] for item in result["entrance"]) < result["lastPreFadeFrame"]


@pytest.mark.render
@pytest.mark.parametrize("look", [look for look in LOOKS if look != "typewriter"])
def test_other_title_looks_fit_and_finish_before_fade(tmp_path: Path, look: str) -> None:
    result = _browser_title(tmp_path, look,
                            ["A very long title word",
                             "SupercalifragilisticexpialidociousSupercalifragilisticexpialidocious"],
                            frames=15)
    assert result["titleScroll"] <= result["titleClient"] + 1
    assert result["cardScroll"] <= result["cardClient"] + 1
    assert result["card"]["bottom"] <= result["rootHeight"]
    assert max(item["end"] for item in result["entrance"]) < result["lastPreFadeFrame"]


@pytest.mark.render
@pytest.mark.parametrize("look", [*LOOKS, "kinetic"])
def test_real_title_look_midpoint_alpha_inside_plan(tmp_path: Path, look: str) -> None:
    plan = title_layer_plan("Small", "steps", _style(look),
                            word_times=[0, .15] if look == "kinetic" else None)
    with paths.operation():
        output = render_layer_plan(plan, tmp_path, tmp_path / f"{look}.mov").path
    midpoint = plan.frames // 2
    rgba = ffmpeg.run([
        "ffmpeg", "-nostdin", "-i", str(output), "-map", "0:v:0", "-an",
        "-vf", f"select=eq(n\\,{midpoint})", "-frames:v", "1", "-pix_fmt", "rgba",
        "-f", "rawvideo", "pipe:1",
    ], timeout_s=120, binary=True).stdout
    assert len(rgba) == plan.width * plan.height * 4
    alpha = np.frombuffer(rgba, dtype=np.uint8).reshape(plan.height, plan.width, 4)[:, :, 3]
    rows, columns = np.nonzero(alpha > 2)
    assert len(rows) > 100 and len(columns) > 100
    assert 0 < rows.min() <= rows.max() < plan.height - 1
    assert 0 < columns.min() <= columns.max() < plan.width - 1
    looks_dir = os.environ.get("YAP_STUDIO_TITLE_LOOKS_DIR")
    if looks_dir:
        destination = Path(looks_dir)
        destination.mkdir(parents=True, exist_ok=True)
        ffmpeg.run([
            "ffmpeg", "-nostdin", "-y", "-i", str(output), "-map", "0:v:0", "-an",
            "-vf", f"select=eq(n\\,{midpoint})", "-frames:v", "1",
            str(destination / f"{look}.png"),
        ], timeout_s=120)


def test_only_kinetic_titles_render_without_browser_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    dummy = object()

    def render(*args, **kwargs):
        calls.append(kwargs)
        return dummy

    monkeypatch.setattr(hyperframes, "render_layer", render)
    for look in ("kinetic", "paper_title"):
        plan = title_layer_plan("Small clear", "steps", _style(look), frames=45,
                                word_times=[0, 0.2, 0.5])
        assert render_layer_plan(plan, tmp_path / look, tmp_path / f"{look}.mov") is dummy
    assert calls[0]["browser_gpu"] is False
    assert "browser_gpu" not in calls[1]


@pytest.mark.render
def test_kinetic_render_is_deterministic(tmp_path: Path) -> None:
    plan = title_layer_plan("Small clear", "steps", _style("kinetic"), frames=45,
                            word_times=[0, 0.2, 0.5])
    hashes = []
    with paths.operation():
        for index in range(2):
            output = render_layer_plan(plan, tmp_path / str(index),
                                       tmp_path / f"kinetic-{index}.mov").path
            hashes.append(ffmpeg.run([
                "ffmpeg", "-nostdin", "-i", str(output), "-map", "0:v:0", "-an",
                "-f", "framemd5", "pipe:1",
            ], timeout_s=120).stdout)
    assert hashes[0] == hashes[1]


def test_kinetic_font_ignores_caption_font() -> None:
    style = _style("kinetic")
    style.data["hook"]["font"] = "Gelasio"
    style.data["captions"]["font"] = "Inter"
    before = title_layer_plan("Small clear", "steps", style, word_times=[0, .15, .3])
    style.data["captions"]["font"] = "Instrument Serif"
    after = title_layer_plan("Small clear", "steps", style, word_times=[0, .15, .3])
    assert before == after
    assert before.variables["font_name"] == "Gelasio"
