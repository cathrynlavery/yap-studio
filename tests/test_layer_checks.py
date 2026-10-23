"""Synthetic renderer evidence, safe slot defaults and disposable copy tests."""

from __future__ import annotations

import codecs
import html
import json
import stat
from collections.abc import Callable, Mapping
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path

import pytest

from yap_studio.render import layer_checks
from yap_studio.render.hyperframes import HYPERFRAMES_VERSION, RendererInstall, preflight

pytestmark = pytest.mark.usefixtures("_no_real_node")

_CLEAN = {"ok": True, "errorCount": 0, "warningCount": 0, "infoCount": 0, "findings": []}
_SHRINK_FINDING = {
    "severity": "warning", "code": "console_warning",
    "message": layer_checks.CAPTION_SHRINK_PREFIX + json.dumps({
        "floor_px": 32, "phrases": [{"slot": 0, "size_px": 140.25, "fits": True}],
    }),
    "time": 0.0, "selector": "[data-composition-id]",
}
_BROWSER_PASSES = ("runtime", "layout", "motion", "contrast")
_FAKE_RENDERER = """import codecs, json, os, sys, time
from pathlib import Path

args = sys.argv[1:]
command = args[0]
assert command in {"lint", "check"}
assert "--json" in args
assert sys.stdin.buffer.read() == b""
assert all(os.environ[key] == "1" for key in (
    "HYPERFRAMES_NO_TELEMETRY", "HYPERFRAMES_NO_UPDATE_CHECK",
    "HYPERFRAMES_NO_AUTO_INSTALL", "HYPERFRAMES_SKIP_SKILLS"))
assert os.environ.get("CI") == "1"
comp = Path(args[-1])
files, links, contents = [], {}, {}
for directory, directories, names in os.walk(comp, followlinks=False):
    for name in directories + names:
        entry = Path(directory) / name
        relative = str(entry.relative_to(comp))
        files.append(relative)
        if entry.is_symlink():
            links[relative] = os.readlink(entry)
        elif entry.is_file():
            contents[relative] = entry.read_bytes().hex()
data = (comp / "index.html").read_bytes()
encoding = "utf-8-sig" if data.startswith(codecs.BOM_UTF8) else (
    "utf-16" if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)) else "utf-8")
with Path(os.environ["YAP_TEST_CHECK_RECORD"]).open("a") as stream:
    stream.write(json.dumps({"args": args, "directory": str(comp),
        "html": data.decode(encoding), "files": files, "links": links,
        "contents": contents,
        "history_root": os.environ.get("HYPERFRAMES_HISTORY_ROOT"),
        "history_exists": Path(os.environ["HYPERFRAMES_HISTORY_ROOT"]).is_dir()
            if os.environ.get("HYPERFRAMES_HISTORY_ROOT") else False}) + "\\n")
response = json.loads(Path(os.environ["YAP_TEST_CHECK_CONFIG"]).read_text())[command]
if response["timeout"]:
    time.sleep(60)
sys.stdout.write(response["stdout"].replace("@COPY@", str(comp)))
sys.stderr.write(response["stderr"].replace("@COPY@", str(comp)))
sys.exit(response["exit"])
"""


def _clean_check() -> dict:
    return {
        "ok": True, "strict": False, "browserSkipped": False,
        "lint": dict(_CLEAN), "runtime": dict(_CLEAN),
        "layout": {**_CLEAN, "samples": [0.25, 0.75]},
        "motion": {**_CLEAN, "enabled": True},
        "contrast": {**_CLEAN, "enabled": True, "checked": 1, "passed": 1},
    }


@pytest.fixture
def fake_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., RendererInstall]:
    """Build an executable that records its copy and returns configurable results."""
    config = tmp_path / "responses.json"
    monkeypatch.setenv("YAP_TEST_CHECK_CONFIG", str(config))
    monkeypatch.setenv("YAP_TEST_CHECK_RECORD", str(tmp_path / "calls.jsonl"))

    def build(
        *, lint: dict | None = None, check: dict | None = None,
        lint_output: str | None = None, check_output: str | None = None,
        lint_stderr: str = "", check_stderr: str = "",
        lint_exit: int = 0, check_exit: int = 0, timeout_command: str | None = None,
    ) -> RendererInstall:
        config.write_text(json.dumps({
            "lint": {
                "stdout": json.dumps(_CLEAN if lint is None else lint)
                if lint_output is None else lint_output,
                "stderr": lint_stderr, "exit": lint_exit, "timeout": timeout_command == "lint",
            },
            "check": {
                "stdout": json.dumps(_clean_check() if check is None else check)
                if check_output is None else check_output,
                "stderr": check_stderr, "exit": check_exit, "timeout": timeout_command == "check",
            },
        }), encoding="utf-8")
        executable = tmp_path / "hyperframes"
        executable.write_text("#!/usr/bin/env python3\n" + _FAKE_RENDERER, encoding="utf-8")
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        return RendererInstall(HYPERFRAMES_VERSION, tmp_path, executable, None, "v22.0.0")

    return build


def _composition(tmp_path: Path, source: str | None = None) -> Path:
    comp = tmp_path / "title"
    comp.mkdir()
    (comp / "index.html").write_text(
        source if source is not None else
        "<!doctype html><html data-composition-variables='[]'><head></head>"
        '<body><div id="title">Title</div></body></html>',
        encoding="utf-8",
    )
    return comp


def _check(
    comp: Path, renderer: RendererInstall, *, variables: Mapping[str, object] | None = None,
    duration_s: float = 4, timeout_s: float = 10,
) -> dict:
    return layer_checks.check_layer(
        comp, {} if variables is None else variables, install=renderer,
        fps=30, duration_s=duration_s, timeout_s=timeout_s,
    )


def _calls(tmp_path: Path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]


def _assert_deleted(tmp_path: Path) -> None:
    assert not list(tmp_path.glob(".layer-check-*"))
    assert all(not Path(call["directory"]).exists() for call in _calls(tmp_path))


class _Tags(HTMLParser):
    """Collect parsed attributes and element names without executing slot text."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.variables: list[dict] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        value = dict(attrs).get("data-composition-variables")
        if value is not None:
            self.variables = json.loads(value)


def test_clean_checks_pass_and_return_json_ready_evidence(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall],
) -> None:
    result = _check(_composition(tmp_path), fake_renderer())
    assert result["schema"] == "yap-studio/layer-checks@1"
    assert result["renderer"] == {
        "name": "hyperframes", "version": HYPERFRAMES_VERSION, "workers": 1,
        "capture": "screenshot",
    }
    assert all("--no-browser-gpu" not in call["args"] for call in _calls(tmp_path))
    assert result["status"] == "passed"
    assert result["samples"] == [0.25, 0.75]
    assert set(result["passes"]) == {"lint", *_BROWSER_PASSES}
    assert all(item == {
        "status": "passed", "errors": 0, "warnings": 0,
        "findings": [], "findings_truncated": 0, "reason": None,
    } for item in result["passes"].values())
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    _assert_deleted(tmp_path)


def test_software_browser_checks_match_the_render_mode(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall],
) -> None:
    result = layer_checks.check_layer(
        _composition(tmp_path), {}, install=fake_renderer(), fps=30, duration_s=4,
        browser_gpu=False,
    )
    lint, check = _calls(tmp_path)
    assert lint["args"] == ["lint", "--json", lint["directory"]]
    assert check["args"][0] == "check"
    assert check["args"].count("--no-browser-gpu") == 1
    assert result["renderer"] == {
        "name": "hyperframes", "version": HYPERFRAMES_VERSION, "workers": 1,
        "capture": "screenshot", "browser_gpu": False,
    }
    assert result["status"] == "passed"
    _assert_deleted(tmp_path)


def test_checks_discard_separate_history_folders(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall],
) -> None:
    _check(_composition(tmp_path), fake_renderer())
    calls = _calls(tmp_path)
    assert [call["args"][0] for call in calls] == ["lint", "check"]
    roots = [Path(call["history_root"]) for call in calls]
    assert roots[0] != roots[1]
    for call, root in zip(calls, roots, strict=True):
        assert call["history_exists"]
        assert not root.is_relative_to(Path.home())
        assert not root.exists()


def test_failed_check_discards_history_folder(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall],
) -> None:
    _check(_composition(tmp_path), fake_renderer(check_exit=4, check_output="failure"))
    calls = _calls(tmp_path)
    assert len(calls) == 2 and calls[1]["args"][0] == "check"
    assert calls[1]["history_exists"]
    assert not Path(calls[1]["history_root"]).exists()


def test_layout_warning_is_mapped_and_overall_status_is_findings(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall],
) -> None:
    check = _clean_check()
    check["layout"].update(warningCount=1, findings=[{
        "severity": "warning", "code": "text-overflow", "message": "Text exceeds its box",
        "time": 0.75, "selector": "#title", "extra": "not included",
    }])
    result = _check(_composition(tmp_path), fake_renderer(check=check))
    assert result["status"] == "findings"
    assert result["passes"]["layout"] == {
        "status": "warnings", "errors": 0, "warnings": 1,
        "findings": [{"severity": "warning", "code": "text-overflow",
                      "message": "Text exceeds its box", "time": 0.75, "selector": "#title"}],
        "findings_truncated": 0, "reason": None,
    }


@pytest.mark.parametrize("warning_count", [0, 1, 2, 7])
def test_caption_shrink_is_info_before_ordinary_console_warning(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall], warning_count: int,
) -> None:
    ordinary = {"severity": "warning", "code": "console_warning", "message": "Ordinary warning",
                "time": 0.0, "selector": "[data-composition-id]"}
    check = _clean_check()
    check["runtime"].update(warningCount=warning_count, findings=[ordinary, _SHRINK_FINDING])
    result = _check(_composition(tmp_path), fake_renderer(check=check))
    runtime = result["passes"]["runtime"]
    assert runtime["findings"] == [{**_SHRINK_FINDING, "severity": "info"}, ordinary]
    assert runtime["warnings"] == max(1, warning_count - 1)
    assert runtime["errors"] == 0 and runtime["status"] == "warnings"
    assert result["status"] == "findings"


def test_caption_shrink_alone_passes(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall],
) -> None:
    check = _clean_check()
    check["runtime"].update(warningCount=1, findings=[_SHRINK_FINDING])
    result = _check(_composition(tmp_path), fake_renderer(check=check))
    runtime = result["passes"]["runtime"]
    assert runtime["findings"] == [{**_SHRINK_FINDING, "severity": "info"}]
    assert runtime["warnings"] == runtime["errors"] == 0
    assert runtime["status"] == result["status"] == "passed"


def test_caption_shrink_survives_findings_cap(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall],
) -> None:
    ordinary = [{"severity": "warning", "message": f"Finding {index}"} for index in range(30)]
    check = _clean_check()
    check["runtime"].update(warningCount=40, findings=[*ordinary, _SHRINK_FINDING])
    result = _check(_composition(tmp_path), fake_renderer(check=check))
    runtime = result["passes"]["runtime"]
    assert runtime["findings"][0] == {**_SHRINK_FINDING, "severity": "info"}
    assert [item["message"] for item in runtime["findings"][1:]] == [
        item["message"] for item in ordinary[:24]
    ]
    assert len(runtime["findings"]) == 25 and runtime["findings_truncated"] == 6
    assert runtime["warnings"] == 39 and runtime["errors"] == 0
    assert runtime["status"] == "warnings" and result["status"] == "findings"


@pytest.mark.parametrize(("code", "message"), [
    ("text-overflow", _SHRINK_FINDING["message"]),
    ("console_warning", "yap-studio caption-shrinkage {}"),
])
def test_caption_shrink_classification_requires_code_and_exact_prefix(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall], code: str, message: str,
) -> None:
    check = _clean_check()
    check["runtime"].update(warningCount=1, findings=[{
        **_SHRINK_FINDING, "code": code, "message": message,
    }])
    result = _check(_composition(tmp_path), fake_renderer(check=check))
    runtime = result["passes"]["runtime"]
    assert runtime["findings"][0]["severity"] == "warning"
    assert runtime["warnings"] == 1 and runtime["status"] == "warnings"


def test_nonzero_exits_keep_json_error_findings(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall],
) -> None:
    lint = {**_CLEAN, "ok": False, "errorCount": 1,
            "findings": [{"severity": "error", "message": "Invalid timeline"}]}
    check = _clean_check()
    check["ok"] = False
    check["runtime"] = {**lint, "errorCount": 3}
    result = _check(
        _composition(tmp_path), fake_renderer(lint=lint, check=check, lint_exit=1, check_exit=2)
    )
    assert result["status"] == "findings"
    assert result["passes"]["lint"]["status"] == "errors"
    assert result["passes"]["runtime"]["errors"] == 3
    assert result["passes"]["runtime"]["findings"][0]["message"] == "Invalid timeline"
    assert result["passes"]["layout"]["status"] == "passed"


@pytest.mark.parametrize("stderr", [
    "Error: Failed to launch the browser process!",
    "uv_os_setpriority returned EPERM",
    "browser: operation not permitted in sandbox",
])
def test_browser_blocked_keeps_lint_and_skips_all_browser_passes(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall], stderr: str,
) -> None:
    result = _check(_composition(tmp_path), fake_renderer(
        check={"ok": False, "error": "launch failed"}, check_stderr=stderr, check_exit=1,
    ))
    assert result["status"] == "incomplete"
    assert result["samples"] == []
    assert result["passes"]["lint"]["status"] == "passed"
    for name in _BROWSER_PASSES:
        assert result["passes"][name] == {
            "status": "skipped", "errors": 0, "warnings": 0, "findings": [],
            "findings_truncated": 0, "reason": "browser could not start",
        }


@pytest.mark.parametrize("command", ["lint", "check"])
@pytest.mark.parametrize("output", ["", "not JSON", "[]", '{"ok": true,}'])
def test_missing_or_invalid_json_is_skipped(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall], command: str, output: str,
) -> None:
    renderer = fake_renderer(
        **{f"{command}_output": output, f"{command}_stderr": "output failed\nmore"}
    )
    result = _check(_composition(tmp_path), renderer)
    assert result["status"] == "incomplete"
    skipped = ("lint",) if command == "lint" else _BROWSER_PASSES
    for name in skipped:
        assert result["passes"][name]["status"] == "skipped"
        assert result["passes"][name]["reason"] == f"{command} did not run: output failed"
    if command == "check":
        assert result["samples"] == []
        assert result["passes"]["lint"]["status"] == "passed"
    else:
        assert all(result["passes"][name]["status"] == "passed" for name in _BROWSER_PASSES)


def test_check_error_object_is_skipped(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall],
) -> None:
    result = _check(_composition(tmp_path), fake_renderer(
        check={"ok": False, "error": "renderer failed\nmore detail"}, check_exit=1,
    ))
    assert result["status"] == "incomplete"
    assert all(result["passes"][name]["reason"] == "check did not run: renderer failed"
               for name in _BROWSER_PASSES)


def test_browser_skipped_is_incomplete_even_with_clean_pass_objects(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall],
) -> None:
    check = {**_clean_check(), "browserSkipped": True}
    result = _check(_composition(tmp_path), fake_renderer(check=check))
    assert result["status"] == "incomplete"
    assert result["samples"] == []
    assert all(result["passes"][name]["reason"] == "browser checks skipped by the renderer"
               for name in _BROWSER_PASSES)


@pytest.mark.parametrize(("name", "changes", "reason"), [
    ("motion", {"enabled": False}, "motion check not enabled"),
    ("contrast", {"enabled": False},
     "no text on an opaque background in this layer; captions are measured on the final"),
    ("contrast", {"checked": 0},
     "no text on an opaque background in this layer; captions are measured on the final"),
])
def test_disabled_or_unmeasured_passes_are_skipped(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall],
    name: str, changes: dict, reason: str,
) -> None:
    check = _clean_check()
    check[name].update(changes)
    result = _check(_composition(tmp_path), fake_renderer(check=check))
    assert result["status"] == "incomplete"
    assert result["passes"][name]["status"] == "skipped"
    assert result["passes"][name]["reason"] == reason


def test_findings_take_precedence_over_skipped_passes(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall],
) -> None:
    result = _check(_composition(tmp_path), fake_renderer(
        lint={**_CLEAN, "warningCount": 1}, check_output="", check_stderr="unavailable",
    ))
    assert result["status"] == "findings"
    assert result["passes"]["lint"]["status"] == "warnings"
    assert result["passes"]["layout"]["status"] == "skipped"


@pytest.mark.parametrize("name", ["lint", *_BROWSER_PASSES])
def test_findings_are_capped_per_pass_with_full_counts(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall], name: str,
) -> None:
    report = {**_CLEAN, "warningCount": 40, "findings": [
        {"severity": "warning", "message": f"Finding {index}"} for index in range(31)
    ]}
    check = _clean_check()
    if name != "lint":
        check[name].update(report)
    result = _check(_composition(tmp_path), fake_renderer(
        lint=report if name == "lint" else None, check=check,
    ))
    evidence = result["passes"][name]
    assert result["status"] == "findings"
    assert evidence["warnings"] == 40
    assert evidence["findings_truncated"] == 6
    assert len(evidence["findings"]) == 25
    assert evidence["findings"][-1]["message"] == "Finding 24"


def test_findings_are_defensively_mapped_with_bounded_json_fallback(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall],
) -> None:
    check = _clean_check()
    unknown = {"unknown": "x" * 400}
    check["layout"]["findings"] = [
        {"level": "warn", "ruleId": "overflow", "description": "Too wide",
         "timeSeconds": "0.5", "elementSelector": "#text"},
        unknown, "plain finding", None,
        {"severity": None, "message": 1, "time": True, "selector": []},
    ]
    result = _check(_composition(tmp_path), fake_renderer(check=check))
    findings = result["passes"]["layout"]["findings"]
    assert findings[0] == {"severity": "warning", "code": "overflow", "message": "Too wide",
                           "time": 0.5, "selector": "#text"}
    assert findings[1]["message"] == json.dumps(unknown)[:300]
    assert findings[2]["message"] == json.dumps("plain finding")
    assert findings[3] == {"severity": "info", "code": None, "message": "null",
                           "time": None, "selector": None}
    assert findings[4]["time"] is None and findings[4]["selector"] is None
    assert result["passes"]["layout"]["warnings"] == 1
    assert result["status"] == "findings"


@pytest.mark.parametrize("quote", ["'", '"'])
def test_copy_has_real_escaped_defaults_and_preserves_caller_files(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall], quote: str,
) -> None:
    declarations = [
        {"id": "text", "default": "original"},
        {"id": "size", "default": 12},
        {"id": "visible", "default": False},
        {"id": "items", "default": []},
        {"id": "unchanged", "default": "retained"},
    ]
    attribute = html.escape(json.dumps(declarations), quote=True)
    source = (
        '<!doctype html>\n<!-- <html data-composition-variables="[]"> -->\n'
        f'<html title="data-composition-variables=decoy"\n'
        f'data-composition-variables={quote}{attribute}{quote}><head></head>'
        '<body><div id="title">Title</div></body></html>'
    )
    comp = _composition(tmp_path, source)
    (comp / "variables.json").write_text('{"old": true}')
    (comp / ".layer-old.mov").write_bytes(b"old render")
    (comp / "asset.mov").write_bytes(b"keep")
    (comp / "gsap.min.js").write_text("local timeline code")
    nested = comp / "assets"
    nested.mkdir()
    (nested / "variables.json").write_text("stale")
    (nested / ".layer-nested.mov").write_bytes(b"stale")
    (nested / "font.woff2").write_bytes(b"local font")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("do not copy")
    (comp / "linked").symlink_to(outside, target_is_directory=True)
    (comp / "linked-file").symlink_to(outside / "secret.txt")
    text = '<script id="injected">bad</script> \' " & </html>'
    variables = {"text": text, "size": 28, "visible": True,
                 "items": ["one", {"value": 2}], "undeclared": "ignore"}
    result = _check(comp, fake_renderer(), variables=variables)
    assert result["status"] == "passed"
    original = _Tags()
    original.feed(source)
    calls = _calls(tmp_path)
    assert [call["args"][0] for call in calls] == ["lint", "check"]
    for call in calls:
        parser = _Tags()
        parser.feed(call["html"])
        assert parser.tags == original.tags
        defaults = {item["id"]: item["default"] for item in parser.variables}
        expected = {key: value for key, value in variables.items() if key != "undeclared"}
        assert defaults == {**expected, "unchanged": "retained"}
        assert "&lt;script" in call["html"] and r"\&quot;injected\&quot;" in call["html"]
        assert '<script id="injected">' not in call["html"]
        assert 'data-composition-variables="' in call["html"]
        assert {"asset.mov", "gsap.min.js", "assets/font.woff2", "variables.json",
                ".layer-old.mov", "assets/variables.json", "assets/.layer-nested.mov"} <= (
            set(call["files"])
        )
        for name in ("variables.json", ".layer-old.mov", "assets/variables.json",
                     "assets/.layer-nested.mov"):
            assert bytes.fromhex(call["contents"][name]) == (comp / name).read_bytes()
        assert call["links"] == {"linked": str(outside), "linked-file": str(outside / "secret.txt")}
        assert "linked/secret.txt" not in call["files"]
        assert Path(call["directory"]).parent == comp.parent
    assert (comp / "index.html").read_text() == source
    assert (comp / "variables.json").read_text() == '{"old": true}'
    assert (comp / ".layer-old.mov").read_bytes() == b"old render"
    _assert_deleted(tmp_path)


def test_copy_preserves_referenced_layer_asset_and_caller_variables(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall],
) -> None:
    source = (
        '<!doctype html><html data-composition-variables=\'[{"id":"text",'
        '"default":"original"}]\'><body><video src=".layer-intro.mov"></video></body></html>'
    )
    comp = _composition(tmp_path, source)
    asset = b"synthetic intro clip"
    caller_variables = '{"text": "caller default"}'
    (comp / ".layer-intro.mov").write_bytes(asset)
    (comp / "variables.json").write_text(caller_variables, encoding="utf-8")
    result = _check(comp, fake_renderer(), variables={"text": "rendered text"})
    assert result["status"] == "passed"
    for call in _calls(tmp_path):
        assert '<video src=".layer-intro.mov">' in call["html"]
        assert ".layer-intro.mov" in call["files"]
        assert bytes.fromhex(call["contents"][".layer-intro.mov"]) == asset
        assert bytes.fromhex(call["contents"]["variables.json"]).decode() == caller_variables
        parser = _Tags()
        parser.feed(call["html"])
        assert parser.variables == [{"id": "text", "default": "rendered text"}]
    _assert_deleted(tmp_path)


@pytest.mark.parametrize(("bom", "encoding"), [
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
    (codecs.BOM_UTF8, "utf-8"),
])
def test_copy_rewrites_defaults_preserving_source_encoding_and_bom(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall], bom: bytes, encoding: str,
) -> None:
    declarations = [{"id": "text", "type": "string", "label": "Text", "default": "Café"}]
    attribute = html.escape(json.dumps(declarations, ensure_ascii=False), quote=True)
    source = (
        f'<html data-composition-variables="{attribute}"><body>'
        '<div data-composition-id="title" data-no-timeline data-duration="0.999" '
        'data-width="320" data-height="180">Résumé</div></body></html>'
    )
    comp = _composition(tmp_path)
    original = bom + source.encode(encoding)
    (comp / "index.html").write_bytes(original)
    assert preflight(comp, 30) == 0.999
    text = 'Café <script>slot</script> "quoted"'
    result = _check(comp, fake_renderer(), variables={"text": text})
    assert result["status"] == "passed"
    for call in _calls(tmp_path):
        copied = bytes.fromhex(call["contents"]["index.html"])
        assert copied.startswith(bom)
        decoded = copied[len(bom):].decode(encoding)
        assert decoded == call["html"]
        assert "Résumé</div>" in decoded
        parser = _Tags()
        parser.feed(decoded)
        assert parser.variables == [{**declarations[0], "default": text}]
        assert "script" not in parser.tags
    assert (comp / "index.html").read_bytes() == original
    _assert_deleted(tmp_path)


def test_composition_without_variables_attribute_is_copied_unchanged(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall],
) -> None:
    source = '<!doctype html><html><body><div id="title">Static title</div></body></html>'
    result = _check(_composition(tmp_path, source), fake_renderer(), variables={"text": "ignored"})
    assert result["status"] == "passed"
    assert all(call["html"] == source for call in _calls(tmp_path))
    _assert_deleted(tmp_path)


@pytest.mark.parametrize("command", ["lint", "check"])
def test_timeout_is_skipped_and_the_copy_is_deleted(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall], command: str,
) -> None:
    result = _check(
        _composition(tmp_path), fake_renderer(timeout_command=command), timeout_s=0.5
    )
    assert result["status"] == "incomplete"
    names = ("lint",) if command == "lint" else _BROWSER_PASSES
    for name in names:
        assert result["passes"][name]["status"] == "skipped"
        assert result["passes"][name]["reason"] == f"{command} did not run: timed out after 0.5 s"
    if command == "check":
        assert result["passes"]["lint"]["status"] == "passed"
        assert result["samples"] == []
    assert len(_calls(tmp_path)) == 2
    _assert_deleted(tmp_path)


def test_exec_failure_skips_all_passes_and_deletes_the_copy(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall],
) -> None:
    renderer = replace(fake_renderer(), executable=tmp_path / "missing-renderer")
    result = _check(_composition(tmp_path), renderer)
    assert result["status"] == "incomplete"
    assert all(item["status"] == "skipped" for item in result["passes"].values())
    assert all("did not run" in item["reason"] for item in result["passes"].values())
    assert str(tmp_path) not in json.dumps(result)
    assert not list(tmp_path.glob(".layer-check-*"))


@pytest.mark.parametrize("declarations", ["not JSON", '{"text": "wrong shape"}'])
def test_invalid_declarations_skip_checks_and_clean_up(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall], declarations: str,
) -> None:
    source = f"<html data-composition-variables='{declarations}'></html>"
    result = _check(_composition(tmp_path, source), fake_renderer())
    assert result["status"] == "incomplete"
    assert all(item["status"] == "skipped" for item in result["passes"].values())
    assert not (tmp_path / "calls.jsonl").exists()
    assert not list(tmp_path.glob(".layer-check-*"))


def test_symlink_index_is_not_followed_or_modified(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall],
) -> None:
    comp = _composition(tmp_path)
    index = comp / "index.html"
    source = index.read_text()
    outside = tmp_path / "external.html"
    outside.write_text(source)
    index.unlink()
    index.symlink_to(outside)
    result = _check(comp, fake_renderer())
    assert result["status"] == "incomplete"
    assert outside.read_text() == source
    assert index.is_symlink()
    assert not list(tmp_path.glob(".layer-check-*"))


def test_absolute_copy_paths_are_removed_from_all_finding_fields(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall],
) -> None:
    check = _clean_check()
    check["layout"]["findings"] = [
        {"severity": "warning", "code": "@COPY@/rule", "message": "@COPY@/index.html: overflow",
         "selector": "@COPY@/element"},
        {"unknown": "@COPY@/index.html"},
    ]
    result = _check(_composition(tmp_path), fake_renderer(check=check))
    findings = result["passes"]["layout"]["findings"]
    assert findings[0]["message"] == "title/index.html: overflow"
    assert findings[0]["code"] == "title/rule"
    assert findings[0]["selector"] == "title/element"
    assert findings[1]["message"] == json.dumps({"unknown": "title/index.html"})
    assert str(tmp_path) not in json.dumps(result)


@pytest.mark.parametrize("command", ["lint", "check"])
def test_skipped_reasons_strip_copy_paths_before_truncation(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall], command: str,
) -> None:
    stderr = "@COPY@/index.html: " + "x" * 300 + "\nsecond line"
    result = _check(_composition(tmp_path), fake_renderer(
        **{f"{command}_output": "", f"{command}_stderr": stderr},
    ))
    name = "lint" if command == "lint" else "runtime"
    assert result["passes"][name]["reason"] == f"{command} did not run: " + (
        "title/index.html: " + "x" * 300
    )[:200]
    assert str(tmp_path) not in json.dumps(result)


def test_check_failure_object_strips_copy_path_from_reason(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall],
) -> None:
    result = _check(_composition(tmp_path), fake_renderer(
        check={"ok": False, "error": "@COPY@/index.html: check failed"}, check_exit=1,
    ))
    assert result["passes"]["runtime"]["reason"] == (
        "check did not run: title/index.html: check failed"
    )
    assert str(tmp_path) not in json.dumps(result)


@pytest.mark.parametrize(("duration_s", "expected"), [
    (0, 9), (0.033, 9), (18, 9), (18.001, 10), (119, 60), (120, 60), (1000, 60),
])
def test_check_sample_count_and_timeout_flags(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall], duration_s: float, expected: int,
) -> None:
    _check(_composition(tmp_path), fake_renderer(), duration_s=duration_s, timeout_s=12.5)
    calls = _calls(tmp_path)
    assert calls[0]["args"] == ["lint", "--json", calls[0]["directory"]]
    assert calls[1]["args"] == ["check", "--json", "--frame-check", f"--samples={expected}",
                               "--timeout=12500", calls[1]["directory"]]
    assert calls[0]["directory"] == calls[1]["directory"]


def test_missing_passes_and_nonfinite_fields_are_not_reported_as_clean(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall],
) -> None:
    check = _clean_check()
    del check["runtime"]
    check["layout"]["samples"] = [0.25, float("nan"), None, 0.75]
    check["layout"]["findings"] = [{"message": "Info", "time": float("inf")}]
    check["layout"]["errorCount"] = "invalid"
    result = _check(_composition(tmp_path), fake_renderer(check=check))
    assert result["status"] == "incomplete"
    assert result["passes"]["runtime"]["reason"] == "runtime result missing"
    assert result["passes"]["layout"]["findings"][0]["time"] is None
    assert result["samples"] == [0.25, 0.75]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("arguments", [
    {"fps": 0}, {"fps": 1.5}, {"duration_s": -1}, {"duration_s": float("inf")},
    {"timeout_s": 0}, {"timeout_s": float("nan")},
])
def test_bad_arguments_are_rejected_before_copy_or_tool_calls(
    tmp_path: Path, fake_renderer: Callable[..., RendererInstall], arguments: dict,
) -> None:
    settings = {"fps": 30, "duration_s": 4, "timeout_s": 10, **arguments}
    with pytest.raises(ValueError):
        layer_checks.check_layer(_composition(tmp_path), {}, install=fake_renderer(), **settings)
    assert not list(tmp_path.glob(".layer-check-*"))
    assert not (tmp_path / "calls.jsonl").exists()
