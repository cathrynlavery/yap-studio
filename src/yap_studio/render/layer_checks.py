"""Collect renderer evidence on a disposable composition with real slot defaults."""

from __future__ import annotations

import codecs
import html
import json
import math
import re
import shutil
import tempfile
from collections.abc import Mapping
from html.parser import HTMLParser
from pathlib import Path

from yap_studio import process
from yap_studio.errors import ProcessingError
from yap_studio.render.hyperframes import (
    HYPERFRAMES_VERSION,
    RENDER_CAPTURE,
    RENDER_WORKERS,
    RendererInstall,
    _bom_encoding,
    _renderer_blocked,
    _renderer_env,
)

_BROWSER_PASSES = ("runtime", "layout", "motion", "contrast")
_VARIABLES_ATTRIBUTE = "data-composition-variables"
_ATTRIBUTE = re.compile(
    r"""([^\s/>=]+)(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]*))?"""
)
_FINDINGS_LIMIT = 25
CAPTION_SHRINK_PREFIX = "yap-studio caption-shrink "


class _VariablesParser(HTMLParser):
    """Locate the first element declaring composition variables and its raw tag."""

    def __init__(self) -> None:
        super().__init__()
        self.start_tag: str | None = None
        self.value: str | None = None
        self.position = (1, 0)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if self.start_tag is None and _VARIABLES_ATTRIBUTE in attributes:
            self.start_tag = self.get_starttag_text()
            self.value = attributes[_VARIABLES_ATTRIBUTE]
            self.position = self.getpos()


def _set_defaults(index: Path, variables: Mapping[str, object]) -> None:
    """Rewrite only the declared defaults, preserving the rest of the HTML."""
    if index.is_symlink():
        raise OSError("composition index is a symlink")
    data = index.read_bytes()
    encoding = _bom_encoding(data) or "utf-8"
    source = data.decode(encoding, errors="strict")
    bom = b""
    if encoding == "utf-16":
        bom = data[:2]
        encoding = "utf-16-le" if bom == codecs.BOM_UTF16_LE else "utf-16-be"
    parser = _VariablesParser()
    parser.feed(source)
    if parser.start_tag is None:
        return
    declarations = json.loads(parser.value or "")
    if not isinstance(declarations, list):
        raise ValueError("composition variables are not a list")
    for declaration in declarations:
        if isinstance(declaration, dict):
            identifier = declaration.get("id")
            if isinstance(identifier, str) and identifier in variables:
                declaration["default"] = variables[identifier]
    raw = parser.start_tag
    tag_end = re.match(r"<\s*[^\s/>]+", raw)
    for attribute in _ATTRIBUTE.finditer(raw, tag_end.end() if tag_end else 0):
        if attribute[1].lower() == _VARIABLES_ATTRIBUTE:
            escaped = html.escape(json.dumps(declarations), quote=True)
            replacement = f'{_VARIABLES_ATTRIBUTE}="{escaped}"'
            rewritten = raw[:attribute.start()] + replacement + raw[attribute.end():]
            line, column = parser.position
            offset = sum(len(part) + 1 for part in source.split("\n")[:line - 1]) + column
            updated = source[:offset] + rewritten + source[offset + len(raw):]
            index.write_bytes(bom + updated.encode(encoding))
            return


def _clean_text(value: str, copy: Path | None, folder: str) -> str:
    """Remove the disposable absolute path before storing or shortening text."""
    if copy is not None:
        for prefix in {str(copy.absolute()), str(copy.resolve())}:
            value = value.replace(prefix, folder).replace(json.dumps(prefix)[1:-1], folder)
    return value


def _skipped(reason: str) -> dict:
    """Return an explicit unavailable pass without implying success."""
    return {
        "status": "skipped", "errors": 0, "warnings": 0, "findings": [],
        "findings_truncated": 0, "reason": reason,
    }


def _number(value: object) -> float | None:
    """Accept only finite numbers, including numeric strings from upstream."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _finding(value: object, copy: Path, folder: str) -> dict:
    """Map varying upstream field names without discarding an unknown finding."""
    fields = value if isinstance(value, dict) else {}

    def text_field(*names: str) -> str | None:
        for name in names:
            item = fields.get(name)
            if isinstance(item, str) and item:
                return _clean_text(item, copy, folder)
        return None

    severity = (text_field("severity", "level") or "info").lower()
    severity = {"warn": "warning", "fatal": "error"}.get(severity, severity)
    message = text_field("message", "description", "detail", "text", "reason")
    if message is None:
        message = _clean_text(json.dumps(value), copy, folder)[:300]
    timestamp = None
    for name in ("time", "timeSeconds", "time_s", "timestamp", "t"):
        timestamp = _number(fields.get(name))
        if timestamp is not None:
            break
    return {
        "severity": severity,
        "code": text_field("code", "rule", "ruleId", "type"),
        "message": message,
        "time": timestamp,
        "selector": text_field("selector", "elementSelector", "element", "target"),
    }


def _pass(report: object, copy: Path, folder: str, name: str) -> dict:
    """Normalize a completed pass, retaining full counts and bounded findings."""
    if not isinstance(report, dict) or not any(
        key in report for key in ("ok", "errorCount", "warningCount", "findings")
    ):
        return _skipped(f"{name} result missing")
    raw_findings = report.get("findings")
    raw_findings = raw_findings if isinstance(raw_findings, list) else []
    findings = [_finding(item, copy, folder) for item in raw_findings]
    caption_shrinks = [
        item for item in findings if item["code"] == "console_warning"
        and item["message"].startswith(CAPTION_SHRINK_PREFIX)
    ]
    if caption_shrinks:
        for item in caption_shrinks:
            item["severity"] = "info"
        findings = caption_shrinks + [item for item in findings if item not in caption_shrinks]
        report = {**report, "warningCount": max(
            0, int(_number(report.get("warningCount")) or 0) - len(caption_shrinks),
        )}
    errors = max(
        int(_number(report.get("errorCount")) or 0),
        sum(item["severity"] == "error" for item in findings),
    )
    warnings = max(
        int(_number(report.get("warningCount")) or 0),
        sum(item["severity"] == "warning" for item in findings),
    )
    if report.get("ok") is False and errors == warnings == 0:
        errors = 1
    return {
        "status": "errors" if errors else "warnings" if warnings else "passed",
        "errors": errors, "warnings": warnings,
        "findings": findings[:_FINDINGS_LIMIT],
        "findings_truncated": max(0, len(findings) - _FINDINGS_LIMIT),
        "reason": None,
    }


def _run_json(
    args: list[str], install: RendererInstall, timeout_s: float, copy: Path, folder: str,
) -> tuple[dict | None, str | None]:
    """Accept JSON on any exit code; record execution and output failures plainly."""
    name = args[0]

    def failure(detail: str) -> tuple[None, str]:
        cleaned = _clean_text(detail, copy, folder).strip()
        first_line = cleaned.splitlines()[0][:200] if cleaned else "no JSON output"
        return None, f"{name} did not run: {first_line}"

    try:
        with _renderer_env() as env:
            result = process.spawn(
                [str(install.executable), *args], timeout_s=timeout_s, tool="hyperframes",
                env=env,
            )
    except ProcessingError as exc:
        if name == "check" and _renderer_blocked(exc.full_stderr):
            return None, "browser could not start"
        if exc.code == "hyperframes_exec_failed":
            return failure("renderer could not execute")
        return failure(exc.full_stderr)
    if name == "check" and _renderer_blocked(result.stderr or ""):
        return None, "browser could not start"
    try:
        report = json.loads(result.stdout)
    except (ValueError, TypeError):
        return failure(result.stderr or "no JSON output")
    if not isinstance(report, dict):
        return failure(result.stderr or "invalid JSON output")
    if "error" in report and report.get("ok") is False:
        detail = report["error"]
        return failure(detail if isinstance(detail, str) else json.dumps(detail))
    return report, None


def _result(passes: dict, samples: list[float], workers: int | str, browser_gpu: bool) -> dict:
    """Assemble versioned evidence, prioritizing findings over incomplete checks."""
    statuses = {item["status"] for item in passes.values()}
    status = (
        "findings" if statuses & {"errors", "warnings"}
        else "incomplete" if "skipped" in statuses else "passed"
    )
    return {
        "schema": "yap-studio/layer-checks@1",
        "renderer": {"name": "hyperframes", "version": HYPERFRAMES_VERSION,
                     "workers": workers, "capture": RENDER_CAPTURE,
                     **({"browser_gpu": False} if browser_gpu is False else {})},
        "status": status, "samples": samples, "passes": passes,
    }


def check_layer(
    composition_dir: Path,
    variables: Mapping[str, object],
    *,
    install: RendererInstall,
    fps: int,
    duration_s: float,
    timeout_s: float = 180,
    workers: int | str = RENDER_WORKERS,
    browser_gpu: bool = True,
) -> dict:
    """Run static and browser checks without letting their failures block a render.

    The copy declares the actual rendered values as defaults because checks do not
    accept a variables file. Disabled, missing and failed checks remain explicit.
    """
    if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
        raise ValueError("fps must be a positive integer")
    if not math.isfinite(duration_s) or duration_s < 0:
        raise ValueError("duration_s must be finite and nonnegative")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be finite and positive")
    folder = composition_dir.name or "composition"
    copy = None
    samples: list[float] = []
    passes = {name: _skipped("checks did not run") for name in ("lint", *_BROWSER_PASSES)}
    try:
        copy = Path(tempfile.mkdtemp(dir=composition_dir.parent, prefix=".layer-check-"))
        shutil.copytree(
            composition_dir, copy, symlinks=True, dirs_exist_ok=True,
            copy_function=shutil.copyfile,
        )
        _set_defaults(copy / "index.html", variables)
        lint, reason = _run_json(["lint", "--json", str(copy.absolute())], install,
                                 timeout_s, copy, folder)
        passes["lint"] = _skipped(reason) if reason else _pass(lint, copy, folder, "lint")
        count = min(60, max(9, math.ceil(duration_s / 2)))
        report, reason = _run_json(
            ["check", "--json", "--frame-check", f"--samples={count}",
             f"--timeout={math.ceil(timeout_s * 1000)}",
             *(["--no-browser-gpu"] if browser_gpu is False else []), str(copy.absolute())],
            install, timeout_s, copy, folder,
        )
        if report is not None and report.get("browserSkipped") is True:
            reason = "browser checks skipped by the renderer"
        if reason is not None:
            passes.update({name: _skipped(reason) for name in _BROWSER_PASSES})
        else:
            assert report is not None
            for name in _BROWSER_PASSES:
                passes[name] = _pass(report.get(name), copy, folder, name)
            motion = report.get("motion")
            if isinstance(motion, dict) and motion.get("enabled") is False:
                passes["motion"] = _skipped("motion check not enabled")
            contrast = report.get("contrast")
            if isinstance(contrast, dict) and (
                contrast.get("enabled") is False or contrast.get("checked") == 0
            ):
                passes["contrast"] = _skipped(
                    "no text on an opaque background in this layer; "
                    "captions are measured on the final"
                )
            layout = report.get("layout")
            raw_samples = layout.get("samples") if isinstance(layout, dict) else None
            if isinstance(raw_samples, list):
                samples = [number for item in raw_samples if (number := _number(item)) is not None]
    except (OSError, shutil.Error):
        passes = {name: _skipped("checks did not run: could not prepare composition copy")
                  for name in passes}
    except (UnicodeError, ValueError) as exc:
        reason = _clean_text(f"checks did not run: {exc}", copy, folder).splitlines()[0][:200]
        passes = {name: _skipped(reason) for name in passes}
    finally:
        if copy is not None:
            shutil.rmtree(copy, ignore_errors=True)
    return _result(passes, samples, workers, browser_gpu)
