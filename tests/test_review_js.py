"""JavaScript timeline parity and browser-page static safety checks."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from fractions import Fraction
from html.parser import HTMLParser
from pathlib import Path

import pytest

from yap_studio.review.model import frame_time_text

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src/yap_studio/review/static"
NODE = shutil.which("node")


def run_node(*args: str) -> subprocess.CompletedProcess[str]:
    """Run Node from the repository root with a bounded, noninteractive process."""
    assert NODE is not None, "Node is required for JavaScript tests"
    return subprocess.run(
        [NODE, *args],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        # The whole suite includes Chromium walks; a loaded runner needs minutes.
        timeout=600,
        check=False,
    )


def test_node_timeline_suite() -> None:
    files = sorted(
        path.relative_to(ROOT).as_posix() for path in (ROOT / "tests/js").glob("*.test.mjs")
    )
    assert files, "no JavaScript test files found"
    result = run_node("--test", *files)
    assert result.returncode == 0, result.stdout + result.stderr


def test_frame_time_text_matches_python() -> None:
    frames = list(range(0, 18000, 7))
    rates = [(30000, 1001), (24000, 1001), (25, 1)]
    module_url = (STATIC / "timeline.mjs").as_uri()
    script = (
        f"import {{frameTimeText}} from {json.dumps(module_url)};\n"
        f"const frames = {json.dumps(frames)};\n"
        f"const rates = {json.dumps(rates)};\n"
        "console.log(JSON.stringify(rates.map(([num, den]) => "
        "frames.map(frame => frameTimeText(frame, {num, den})))));"
    )
    result = run_node("--input-type=module", "-e", script)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = json.loads(result.stdout)
    expected = [
        [frame_time_text(frame, Fraction(num, den)) for frame in frames]
        for num, den in rates
    ]
    assert actual == expected


class PageParser(HTMLParser):
    """Collect tags and attributes needed for the strict static-page guard."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.inline_script = False
        self.in_script = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))
        if tag == "script":
            self.in_script = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self.in_script = False

    def handle_data(self, data: str) -> None:
        if self.in_script and data.strip():
            self.inline_script = True


@pytest.mark.parametrize("page", ["index.html", "list.html"])
def test_static_page_obeys_csp_and_escapes_user_content(page: str) -> None:
    parser = PageParser()
    parser.feed((STATIC / page).read_text())
    assert not parser.inline_script
    assert all(tag != "style" for tag, _ in parser.tags)
    for tag, attrs in parser.tags:
        assert "style" not in attrs
        assert all(not name.startswith("on") for name in attrs)
        if tag == "script":
            assert attrs.get("type") == "module"
            assert attrs.get("src", "").startswith("static/")
        if tag == "link":
            assert attrs.get("href", "").startswith("static/")
    for name in ("app.mjs", "timeline.mjs", "taste.mjs"):
        source = (STATIC / name).read_text()
        for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "eval(", "new Function"):
            assert forbidden not in source
        assert re.search(r"https?://", source) is None


def test_series_controls_are_accessible_and_modules_parse() -> None:
    parser = PageParser()
    parser.feed((STATIC / "index.html").read_text())
    tags = {attrs["id"]: (tag, attrs) for tag, attrs in parser.tags if "id" in attrs}
    assert "hidden" in tags["version-bar"][1]
    assert tags["compare-toggle"][1]["aria-expanded"] == "false"
    assert tags["compare-toggle"][1]["aria-controls"] == "compare-panel"
    assert "controls" in tags["compare-video"][1]
    assert tags["compare-version"][1]["aria-label"] == "Compare version"
    assert tags["map-status"][1]["aria-live"] == "polite"
    assert "hidden" in tags["earlier-threads"][1]
    for name in ("app.mjs", "timeline.mjs", "taste.mjs"):
        result = run_node("--check", str(STATIC / name))
        assert result.returncode == 0, result.stderr


def test_taste_section_is_accessible_and_wraps_on_phones() -> None:
    parser = PageParser()
    parser.feed((STATIC / "index.html").read_text())
    tags = {attrs["id"]: (tag, attrs) for tag, attrs in parser.tags if "id" in attrs}
    assert tags["taste-section"][0] == "section"
    assert "hidden" in tags["taste-section"][1]
    assert tags["taste-section"][1]["aria-labelledby"] == "taste-heading"
    assert tags["taste-heading"][0] == "h2"
    assert tags["taste-count"][1]["class"] == "count"
    assert tags["taste-list"][0] == "ol"
    assert tags["taste-decided"][0] == "details"
    assert tags["taste-decided"][1]["class"] == "taste-decided"
    assert "hidden" in tags["taste-decided"][1]
    assert "open" not in tags["taste-decided"][1]
    assert tags["taste-decided-count"][1]["class"] == "count"
    assert tags["taste-decided-list"][0] == "ol"
    assert tags["taste-empty"][1]["class"] == "muted"
    assert "hidden" in tags["taste-empty"][1]
    assert tags["taste-error"][1]["role"] == "alert"
    assert "hidden" in tags["taste-error"][1]
    ids = [attrs["id"] for _, attrs in parser.tags if "id" in attrs]
    assert ids.index("taste-section") < ids.index("removed-section")
    assert ids.index("taste-list") < ids.index("taste-empty") < ids.index("taste-decided")
    css = (STATIC / "app.css").read_text()
    assert re.search(r"\.taste-actions\s*\{[^}]*flex-wrap: wrap[^}]*gap:", css)
    assert re.search(r"\.taste-card p\s*\{[^}]*overflow-wrap: anywhere", css)
    assert re.search(r"\.taste-notes-heading\s*\{[^}]*display: block[^}]*margin-top: 10px", css)
    assert re.search(r"@media \(max-width: 480px\).*?\.taste-card textarea\s*"
                     r"\{[^}]*font-size: 16px", css, re.DOTALL)


def test_round_controls_are_accessible_and_wrap() -> None:
    parser = PageParser()
    parser.feed((STATIC / "index.html").read_text())
    tags = {attrs["id"]: (tag, attrs) for tag, attrs in parser.tags if "id" in attrs}
    assert "hidden" in tags["send-bar"][1]
    assert tags["send-bar"][1]["aria-label"] == "Send notes"
    assert tags["send-notes"][1]["type"] == "button"
    assert "disabled" in tags["send-notes"][1]
    assert tags["round-list"][1]["aria-live"] == "polite"
    css = (STATIC / "app.css").read_text()
    assert re.search(r"\.send-actions, \.round-row\s*\{[^}]*flex-wrap: wrap", css)
    assert re.search(r"\.send-bar button\s*\{[^}]*min-height: 44px", css)


def test_findings_section_and_markers_are_separate_and_wrap() -> None:
    parser = PageParser()
    html = (STATIC / "index.html").read_text()
    parser.feed(html)
    tags = {attrs["id"]: (tag, attrs) for tag, attrs in parser.tags if "id" in attrs}
    assert tags["findings-section"][0] == "details"
    assert "hidden" in tags["findings-section"][1]
    assert tags["findings-list"][0] == "ol"
    assert tags["findings-status"][1]["role"] == "status"
    assert "hidden" in tags["findings-status"][1]
    assert "Flagged by the checks" in html
    assert html.index('id="finding-markers"') < html.index('id="markers"')
    assert html.index('id="findings-section"') < html.index('id="comment-list"')
    css = (STATIC / "app.css").read_text()
    assert re.search(r"\.finding-heading, \.finding-actions\s*\{[^}]*flex-wrap: wrap", css)
    assert re.search(r"\.finding-actions button\s*\{[^}]*min-height: 44px", css)
    assert re.search(r"\.finding-marker\s*\{[^}]*height: 44px", css)
    assert re.search(r"\.finding-range\s*\{[^}]*height: 44px", css)
    assert re.search(r"\.finding-range::before\s*\{[^}]*border: 1px dashed", css)
    assert re.search(r"\.finding-marker::before\s*\{[^}]*transform: rotate\(45deg\)", css)


def test_mobile_touch_targets_exclude_scrubber_marker_buttons() -> None:
    css = re.sub(r"/\*.*?\*/", "", (STATIC / "app.css").read_text(), flags=re.DOTALL)
    media = re.search(r"@media\s*\(\s*max-width\s*:\s*480px\s*\)\s*\{", css)
    assert media is not None, "missing phone media query"

    start = end = media.end()
    depth = 1
    while depth and end < len(css):
        if css[end] == "{":
            depth += 1
        elif css[end] == "}":
            depth -= 1
        end += 1
    assert depth == 0, "unclosed phone media query"

    touch_target_selectors = set()
    for rule in re.finditer(r"([^{}]+)\{([^{}]*)\}", css[start : end - 1]):
        if not re.search(
            r"(?:^|;)\s*min-height\s*:\s*44px\s*(?:!important\s*)?(?:;|$)", rule[2]
        ):
            continue
        selectors = {selector.strip() for selector in rule[1].split(",")}
        assert "button" not in selectors, (
            "bare button selector enlarges marker, range-bar, and selection-handle"
        )
        touch_target_selectors.update(selectors)

    assert {
        "button:not(.marker):not(.range-bar):not(.selection-handle):not(.pin-marker)",
        ".button-link",
        "select",
        ".scrubber",
    } <= touch_target_selectors
    assert re.search(r"\.pin-marker\s*\{[^}]*width: 44px;[^}]*height: 44px;", css)
    assert re.search(r"\.pin-marker\s*\{[^}]*touch-action: none", css)
