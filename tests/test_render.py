"""Pinned renderer installation, exact frames, and alpha colour integration tests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path

import offline_renderer
import pytest

from yap_studio import ffmpeg, process, tools
from yap_studio.errors import (
    DependencyError,
    OutputPathError,
    ProcessingError,
    RendererBlockedError,
    YapError,
)
from yap_studio.render import alpha, hyperframes, layer_checks
from yap_studio.render.hyperframes import (
    GSAP_SCRIPT,
    GSAP_VERSION,
    HYPERFRAMES_VERSION,
    RENDERER_ENV,
    RendererInstall,
    ensure_installed,
    gsap_file,
    install_prefix,
    installed,
    layer_duration_s,
    preflight,
    render_layer,
    renderer_frame_count,
)

pytestmark = pytest.mark.usefixtures("_no_real_node")

_SWATCHES = (
    (16, 16, 16, 255),
    (64, 64, 64, 255),
    (240, 240, 240, 255),
    (32, 80, 224, 255),
    (200, 60, 40, 128),
)


@pytest.mark.parametrize("caller", ["layer", "checks", "beats", "studio", "inventory", "matte"])
def test_renderer_children_use_working_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, node_bins: Callable[..., Path], caller: str,
) -> None:
    from yap_studio.matte import hyperframes as matte
    from yap_studio.media import inventory
    from yap_studio.render import beats
    from yap_studio.studio import export

    broken, working = node_bins("broken", returncode=1), node_bins("working")
    original_path = os.pathsep.join([str(broken.parent), str(working.parent)])
    monkeypatch.setenv("PATH", original_path)
    audit = tmp_path / "child.json"
    executable = tmp_path / "hyperframes"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, subprocess, sys\nfrom pathlib import Path\n"
        "node = subprocess.check_output(['/usr/bin/env', 'node'], text=True).strip()\n"
        f"Path({str(audit)!r}).write_text(json.dumps([node, dict(os.environ)]))\n"
        "result = {'ok': True}\n"
        "if sys.argv[1] == 'remove-background':\n"
        "    Path(sys.argv[sys.argv.index('-o') + 1]).touch()\n"
        "if sys.argv[1] == 'beats':\n"
        "    root = Path(sys.argv[3])\n"
        "    (root / 'beats').mkdir()\n"
        "    (root / 'beats/music.wav.json').write_text(json.dumps(\n"
        "        {'version': 1, 'audio': 'music.wav', 'beats': []}))\n"
        "    result.update(file='beats/music.wav.json', count=0)\n"
        "print(json.dumps(result))\n")
    executable.chmod(0o755)
    install = RendererInstall(HYPERFRAMES_VERSION, tmp_path, executable, None, "v26.1.0")
    source = tmp_path / "source.wav"
    source.write_bytes(b"synthetic input for the fake renderer")
    if caller == "layer":
        assert json.loads(hyperframes._run_renderer(executable, ["lint"], timeout_s=5))["ok"]
    elif caller == "checks":
        result, error = layer_checks._run_json(["lint"], install, 5, tmp_path, "source")
        assert result == {"ok": True} and error is None
    elif caller == "beats":
        assert beats.detect_beats(source, duration_s=1, install=install) == {
            "bpm": None, "beats": []}
    elif caller == "studio":
        assert export._run(["lint"], install).returncode == 0
    elif caller == "inventory":
        assert inventory._resolve(tmp_path, install, []) == {"ok": True}
    else:
        matte._run(install, source, tmp_path / "output.webm", time.monotonic() + 5)
    node, env = json.loads(audit.read_text())
    assert node == str(working)
    assert env["PATH"] == str(working.parent) + os.pathsep + original_path
    assert all(env[key] == value for key, value in RENDERER_ENV.items())
    assert env.get("CI") == "1"


# The observed worker startup failure precedes Chrome launch in the agent sandbox.
_SANDBOX_STDERR = (
    "SystemError [ERR_SYSTEM_ERROR]: A system error occurred: "
    "uv_os_setpriority returned EPERM (operation not permitted)\n"
    "  code: 'ERR_SYSTEM_ERROR',\n"
    "  info: { code: 'EPERM', message: 'operation not permitted', "
    "syscall: 'uv_os_setpriority' }\n"
)
_LOCAL_SERVER_STDERR = (
    '[hyperframes] browserGpuMode probe → hardware (WebGL renderer vendor="Google Inc. (Apple)")\n'
    "\n✗  Render failed\n\n"
    "   net::ERR_INVALID_HTTP_RESPONSE at http://localhost:57764/index.html\n"
    "   Try --docker for containerized rendering\n"
)
_LOCAL_SERVER_ERRORS = (
    "ERR_INVALID_HTTP_RESPONSE", "ERR_CONNECTION_RESET", "ERR_EMPTY_RESPONSE",
)
_FAKE_RENDERER = """import hashlib, json, math, os, re, subprocess, sys
from html.parser import HTMLParser
from pathlib import Path

class Parser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "data-composition-id" in attrs:
            self.root = attrs
        elif tag == "div" and hasattr(self, "root"):
            self.fill = attrs

args = sys.argv[1:]
if args[0] in {"lint", "check"}:
    assert all(os.environ[key] == "1" for key in (
        "HYPERFRAMES_NO_TELEMETRY", "HYPERFRAMES_NO_UPDATE_CHECK",
        "HYPERFRAMES_NO_AUTO_INSTALL", "HYPERFRAMES_SKIP_SKILLS"))
    assert os.environ.get("CI") == "1"
    comp = Path(args[-1])
    log = os.environ.get("YAP_TEST_CHECK_LOG")
    if log:
        with Path(log).open("a") as stream:
            stream.write(json.dumps({"args": args, "directory": str(comp),
                "gsap_exists": (comp / "gsap.min.js").is_file(),
                "variables_exist": (comp / "variables.json").exists()}) + "\\n")
    clean = {"ok": True, "errorCount": 0, "warningCount": 0,
        "infoCount": 0, "findings": []}
    report = clean if args[0] == "lint" else {
        "ok": True, "browserSkipped": False, "lint": clean, "runtime": clean,
        "layout": {**clean, "samples": [0.025, 0.075]},
        "motion": {**clean, "enabled": True},
        "contrast": {**clean, "enabled": True, "checked": 1, "passed": 1}}
    print(json.dumps(report))
    sys.exit(0)
assert args[0] == "render"
assert args[2:4] == ["--format", "mov"]
assert "--strict-variables" in args and "--quiet" in args
assert "--no-best-effort" in args
assert all(os.environ[key] == "1" for key in (
    "HYPERFRAMES_NO_TELEMETRY", "HYPERFRAMES_NO_UPDATE_CHECK",
    "HYPERFRAMES_NO_AUTO_INSTALL", "HYPERFRAMES_SKIP_SKILLS"))
assert os.environ.get("CI") == "1"
assert sys.stdin.buffer.read() == b""
comp = Path(args[1])
variables = json.loads(Path(args[args.index("--variables-file") + 1]).read_text())
gsap = comp / "gsap.min.js"
(comp / "invoked.json").write_text(json.dumps({"args": args, "variables": variables,
    "gsap_exists": gsap.is_file(),
    "gsap_inode": gsap.stat().st_ino if gsap.is_file() else None,
    "gsap_sha256": hashlib.sha256(gsap.read_bytes()).hexdigest() if gsap.is_file() else None,
    "history_root": os.environ.get("HYPERFRAMES_HISTORY_ROOT"),
    "history_exists": Path(os.environ["HYPERFRAMES_HISTORY_ROOT"]).is_dir()
        if os.environ.get("HYPERFRAMES_HISTORY_ROOT") else False}))
(comp / ".hyperframes").mkdir(exist_ok=True)
(comp / ".hyperframes" / "render-outputs.json").write_text("{}")
counter = os.environ.get("YAP_TEST_RENDER_COUNTER")
if counter:
    counter = Path(counter)
    attempts = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(attempts))
    if attempts <= int(os.environ.get("YAP_TEST_RENDER_FAILURES", "0")):
        sys.stderr.write(os.environ["YAP_TEST_RENDER_STDERR"])
        sys.exit(1)
if "undeclared" in variables:
    sys.stderr.write("Variable validation failed: undeclared\\n")
    sys.exit(1)
parser = Parser()
parser.feed((comp / "index.html").read_text())
width, height = int(parser.root["data-width"]), int(parser.root["data-height"])
fps = int(args[args.index("--fps") + 1]) if "--fps" in args else 30
frames = max(1, math.ceil(float(parser.root["data-duration"]) * fps))
assert "background" not in parser.root["style"] and "opacity" not in parser.root["style"]
match = re.search(r"background:\\s*rgba\\(([^)]*)\\)", parser.fill["style"])
channels = [float(value.strip()) for value in match[1].split(",")]
colour = [int(value) for value in channels[:3]] + [round(channels[3] * 255)]
swatches = variables.get("swatches", [colour])
row = b"".join(bytes(swatches[min(x * len(swatches) // width, len(swatches) - 1)])
    for x in range(width))
pixels = comp / "pixels.rgba"
pixels.write_bytes(row * height)
out = args[args.index("-o") + 1]
matrix = os.environ.get("YAP_TEST_RENDER_MATRIX", "bt709")
assert matrix in {"bt601", "bt709"}
tag = "unknown" if matrix == "bt601" else "bt709"
filters = (f"format=rgba64le,scale=out_color_matrix={matrix}:out_range=tv:"
    "flags=accurate_rnd+full_chroma_int,format=yuva444p12le,"
    f"setparams=color_primaries={tag}:color_trc={tag}:colorspace={tag}:range=tv")
command = [os.environ["YAP_TEST_FFMPEG"], "-nostdin", "-hide_banner", "-loglevel", "error",
    "-y", "-f", "rawvideo", "-pixel_format", "rgba", "-video_size", f"{width}x{height}",
    "-framerate", str(fps), "-stream_loop", "-1", "-i", str(pixels), "-map", "0:v:0",
    "-an", "-vf", filters, "-frames:v", str(frames), "-c:v", "prores_ks", "-profile:v",
    "4444", "-pix_fmt", "yuva444p12le", "-threads", "1", out]
try:
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
    sys.stderr.buffer.write(result.stderr)
    sys.exit(result.returncode)
finally:
    pixels.unlink()
"""


def _composition(
    root: Path,
    *,
    frames: int = 30,
    width: int = 320,
    height: int = 180,
    duration: float | None = None,
    fps: int = 30,
) -> Path:
    root.mkdir()
    duration = layer_duration_s(frames, fps) if duration is None else duration
    (root / "index.html").write_text(
        "<!doctype html><html data-composition-variables='[]'><head><style>"
        "html,body{margin:0;background:transparent}"
        "</style></head><body>"
        f'<div data-composition-id="test" data-no-timeline data-duration="{duration}" '
        f'data-width="{width}" data-height="{height}" '
        f'style="width:{width}px;height:{height}px">'
        '<div style="width:100%;height:100%;background: rgba(200, 60, 40, 0.50196)"></div>'
        "</div></body></html>",
        encoding="utf-8",
    )
    return root


def _gsap_composition(root: Path, *, frames: int = 29, fps: int = 30) -> Path:
    comp = _composition(root, frames=frames, fps=fps)
    html = (comp / "index.html").read_text()
    html = html.replace(" data-no-timeline", "")
    html = html.replace("</head>", f'<script src="{GSAP_SCRIPT}"></script></head>')
    html = html.replace('<div style="width:100%', '<div id="box" style="width:100%')
    html = html.replace("rgba(200, 60, 40, 0.50196)", "rgba(200, 60, 40, 1);opacity:0")
    # The fade ends on the last sampled frame, rather than beyond the frame window.
    timeline = (
        "<script>window.__timelines = window.__timelines || {};"
        'window.__timelines["test"] = gsap.timeline({paused:true})'
        '.fromTo("#box", {opacity:0}, '
        f'{{opacity:1, duration:{(frames - 1) / fps}, ease:"none"}}, 0);</script>'
    )
    (comp / "index.html").write_text(html.replace("</body>", timeline + "</body>"))
    return comp


@pytest.fixture
def fake_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[str], RendererInstall]:
    """Put a faithful alpha encoder or a supplied failing renderer on PATH."""
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("YAP_TEST_FFMPEG", ffmpeg.find_tool("ffmpeg"))

    def build(body: str = _FAKE_RENDERER) -> RendererInstall:
        executable = tmp_path / "hyperframes"
        executable.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        return RendererInstall(HYPERFRAMES_VERSION, tmp_path, executable, None, "v22.0.0")

    return build


@pytest.fixture
def retry_renderer(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch,
) -> RendererInstall:
    renderer = fake_renderer()
    gsap = renderer.prefix / "node_modules" / "gsap" / "dist" / GSAP_SCRIPT
    gsap.parent.mkdir(parents=True)
    gsap.write_bytes(b"fake GSAP")
    monkeypatch.setattr(hyperframes, "GSAP_SHA256", hashlib.sha256(gsap.read_bytes()).hexdigest())
    monkeypatch.setenv("YAP_TEST_RENDER_COUNTER", str(tmp_path / "render-count"))
    monkeypatch.setenv("YAP_TEST_RENDER_FAILURES", "1")
    monkeypatch.setenv("YAP_TEST_RENDER_STDERR", _LOCAL_SERVER_STDERR)
    monkeypatch.setenv("YAP_TEST_CHECK_LOG", str(tmp_path / "check-calls.jsonl"))
    return renderer


def _decode(path: Path, *, matrix: str = "bt709") -> bytes:
    return ffmpeg.run(
        [
            "ffmpeg",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            f"scale=in_color_matrix={matrix}:in_range=tv:out_range=pc,format=rgba",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        timeout_s=60,
        binary=True,
    ).stdout


def _pixel(data: bytes, width: int, x: int, y: int) -> tuple[int, ...]:
    offset = (y * width + x) * 4
    return tuple(data[offset : offset + 4])


def _assert_pixel(actual: tuple[int, ...], expected: tuple[int, ...]) -> None:
    assert len(actual) == len(expected) == 4
    assert all(abs(a - e) <= 1 for a, e in zip(actual, expected, strict=True)), (actual, expected)


@pytest.mark.parametrize("matrix", ["bt709", "bt601"])
def test_swatch_conversion_round_trips_alpha(
    tmp_path: Path,
    fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch,
    matrix: str,
) -> None:
    if matrix == "bt601":
        monkeypatch.setenv("YAP_TEST_RENDER_MATRIX", matrix)
    comp = _composition(tmp_path / "swatches", frames=3)
    convert = alpha.convert_to_bt709
    input_facts: list[ffmpeg.MediaInfo] = []

    def inspect(src: Path, out: Path, *, frames: int, timeout_s: float | None = None) -> Path:
        info = ffmpeg.probe(src)
        input_facts.append(info)
        assert info.video_codec == "prores"
        # ffmpeg 6 leaves ProRes color_range unreported; ffmpeg 9 reports tv.
        assert info.pix_fmt == "yuva444p12le" and info.color_range in {None, "tv"}
        if matrix == "bt601":
            assert info.color_space is info.color_transfer is info.color_primaries is None
        else:
            assert info.color_space == info.color_transfer == info.color_primaries == "bt709"
        raw = _decode(src, matrix=matrix)
        for index, expected in enumerate(_SWATCHES):
            _assert_pixel(_pixel(raw, 320, index * 64 + 32, 90), expected)
        return convert(src, out, frames=frames, timeout_s=timeout_s)

    monkeypatch.setattr(alpha, "convert_to_bt709", inspect)
    out = tmp_path / "converted.mov"
    result = render_layer(
        comp, out, frames=3, variables={"swatches": list(_SWATCHES)}, install=fake_renderer()
    )
    assert len(input_facts) == 1
    assert result.path == out and result.width == 320 and result.height == 180
    assert result.render_retries == 0
    assert isinstance(result.checks, dict) and "render_retries" not in result.checks
    info = ffmpeg.probe(out)
    assert info.video_frames == result.frames == 3
    assert info.color_space == info.color_transfer == info.color_primaries == "bt709"
    assert info.color_range in {None, "tv"} and info.pix_fmt == "yuva444p12le"
    pixels = _decode(out)
    for index, expected in enumerate(_SWATCHES):
        _assert_pixel(_pixel(pixels, 320, index * 64 + 32, 90), expected)
    assert not list(comp.glob(".layer-*.mov"))
    assert not list(comp.glob(".yap_tmp_*"))


@pytest.mark.parametrize("frames", [1, 2, 3])
def test_short_layers_have_exact_frames(
    tmp_path: Path,
    fake_renderer: Callable[[str], RendererInstall],
    frames: int,
) -> None:
    comp = _composition(tmp_path / "short", frames=frames)
    result = render_layer(
        comp, tmp_path / "out.mov", frames=frames, variables={}, install=fake_renderer()
    )
    assert result.frames == ffmpeg.probe(result.path).video_frames == frames
    assert result.duration_s == layer_duration_s(frames)


def test_render_discards_history_and_new_artifacts(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
) -> None:
    comp = _composition(tmp_path / "history", frames=1)
    render_layer(comp, tmp_path / "out.mov", frames=1, variables={},
                 install=fake_renderer(), checks=False)
    invoked = json.loads((comp / "invoked.json").read_text())
    root = Path(invoked["history_root"])
    assert invoked["history_exists"]
    assert not root.is_relative_to(Path.home())
    assert not root.exists()
    assert not (comp / ".hyperframes").exists()


def test_render_preserves_existing_hyperframes_folder(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
) -> None:
    comp = _composition(tmp_path / "existing-artifacts", frames=1)
    artifact = comp / ".hyperframes" / "original.txt"
    artifact.parent.mkdir()
    artifact.write_text("keep")
    render_layer(comp, tmp_path / "out.mov", frames=1, variables={},
                 install=fake_renderer(), checks=False)
    assert artifact.read_text() == "keep"


def test_render_preserves_existing_hyperframes_symlink(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
) -> None:
    comp = _composition(tmp_path / "linked-artifacts", frames=1)
    target = comp / "other-artifacts"
    target.mkdir()
    marker = target / "original.txt"
    marker.write_text("keep")
    link = comp / ".hyperframes"
    link.symlink_to(target, target_is_directory=True)
    render_layer(comp, tmp_path / "out.mov", frames=1, variables={},
                 install=fake_renderer(), checks=False)
    assert link.is_symlink()
    assert marker.read_text() == "keep"


def test_failed_render_discards_history(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comp = _composition(tmp_path / "failed-history", frames=1)
    record = tmp_path / "failed-history-root.json"
    monkeypatch.setenv("YAP_TEST_HISTORY_RECORD", str(record))
    body = (
        "import json, os, sys\nfrom pathlib import Path\n"
        "root = os.environ.get('HYPERFRAMES_HISTORY_ROOT')\n"
        "Path(os.environ['YAP_TEST_HISTORY_RECORD']).write_text(json.dumps("
        "{'root': root, 'exists': Path(root).is_dir() if root else False}))\n"
        "sys.exit(4)\n"
    )
    with pytest.raises(ProcessingError):
        render_layer(comp, tmp_path / "out.mov", frames=1, variables={},
                     install=fake_renderer(body), checks=False)
    observed = json.loads(record.read_text())
    assert observed["exists"]
    assert not Path(observed["root"]).is_relative_to(Path.home())
    assert not Path(observed["root"]).exists()


@pytest.mark.parametrize("workers", [None, "auto"])
def test_render_uses_requested_workers_and_screenshot_capture(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall], workers: str | None,
) -> None:
    comp = _composition(tmp_path / "workers", frames=3)
    options = {} if workers is None else {"workers": workers}
    result = render_layer(
        comp, tmp_path / "out.mov", frames=3, variables={}, install=fake_renderer(), **options
    )
    args = json.loads((comp / "invoked.json").read_text())["args"]
    assert args.count("--workers") == 1
    assert args[args.index("--workers") + 1] == (workers or str(hyperframes.RENDER_WORKERS))
    assert args.count("--experimental-fast-capture=false") == 1
    assert result.checks["renderer"]["workers"] == (workers or hyperframes.RENDER_WORKERS)


@pytest.mark.parametrize("workers", ["0", 0, -1, "fast", True])
def test_render_rejects_invalid_workers_before_launch(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall], workers: object,
) -> None:
    comp = _composition(tmp_path / "workers", frames=3)
    out = tmp_path / "out.mov"
    with pytest.raises(ValueError, match="workers"):
        render_layer(comp, out, frames=3, variables={}, install=fake_renderer(),
                     workers=workers)
    assert not (comp / "invoked.json").exists()
    assert not out.exists() and not list(tmp_path.glob(".yap_tmp_*"))


@pytest.mark.parametrize("browser_gpu", [None, True, False])
def test_render_uses_requested_browser_gpu(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall], browser_gpu: bool | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comp = _composition(tmp_path / "browser-gpu", frames=3)
    log = tmp_path / "check-calls.jsonl"
    monkeypatch.setenv("YAP_TEST_CHECK_LOG", str(log))
    options = {} if browser_gpu is None else {"browser_gpu": browser_gpu}
    render_layer(comp, tmp_path / "out.mov", frames=3, variables={}, install=fake_renderer(),
                 **options)
    args = json.loads((comp / "invoked.json").read_text())["args"]
    assert args.count("--no-browser-gpu") == (1 if browser_gpu is False else 0)
    lint, check = [json.loads(line) for line in log.read_text().splitlines()]
    assert lint["args"][0] == "lint" and "--no-browser-gpu" not in lint["args"]
    assert check["args"][0] == "check"
    assert check["args"].count("--no-browser-gpu") == (1 if browser_gpu is False else 0)


@pytest.mark.parametrize("browser_gpu", ["false", 0, None])
def test_render_rejects_invalid_browser_gpu_before_launch(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall], browser_gpu: object,
) -> None:
    comp = _composition(tmp_path / "browser-gpu", frames=3)
    out = tmp_path / "out.mov"
    renderer = fake_renderer()
    before = set(tmp_path.rglob("*"))
    with pytest.raises(ValueError, match="browser_gpu"):
        render_layer(comp, out, frames=3, variables={}, install=renderer,
                     browser_gpu=browser_gpu)
    assert not (comp / "invoked.json").exists()
    assert not out.exists() and set(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("checks", [True, False])
def test_render_returns_checks_and_can_skip_them(
    tmp_path: Path,
    fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch,
    checks: bool,
) -> None:
    comp = _composition(tmp_path / "checked", frames=3)
    log = tmp_path / "check-calls.jsonl"
    monkeypatch.setenv("YAP_TEST_CHECK_LOG", str(log))
    result = render_layer(
        comp, tmp_path / "out.mov", frames=3, variables={}, install=fake_renderer(), checks=checks
    )
    assert ffmpeg.probe(result.path).video_frames == 3
    if checks:
        assert result.checks["status"] == "passed"
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        assert [call["args"][0] for call in calls] == ["lint", "check"]
        assert all(not call["variables_exist"] for call in calls)
        assert all(not Path(call["directory"]).exists() for call in calls)
    else:
        assert result.checks is None
        assert not log.exists()


def test_render_keeps_check_failures_as_evidence(
    tmp_path: Path,
    fake_renderer: Callable[[str], RendererInstall],
) -> None:
    body = _FAKE_RENDERER.replace(
        'if args[0] in {"lint", "check"}:',
        'if args[0] in {"lint", "check"}:\n'
        '    sys.stderr.write("checks unavailable")\n    sys.exit(2)\n'
        'if args[0] in {"lint", "check"}:',
        1,
    )
    comp = _composition(tmp_path / "unavailable-checks", frames=3)
    result = render_layer(
        comp, tmp_path / "out.mov", frames=3, variables={}, install=fake_renderer(body)
    )
    assert result.checks["status"] == "incomplete"
    assert all(item["status"] == "skipped" for item in result.checks["passes"].values())
    assert ffmpeg.probe(result.path).video_frames == 3


def test_nondefault_fps_is_explicit_and_has_exact_frames(
    tmp_path: Path,
    fake_renderer: Callable[[str], RendererInstall],
) -> None:
    comp = _composition(tmp_path / "fps", frames=3, fps=24)
    result = render_layer(
        comp, tmp_path / "out.mov", frames=3, variables={}, fps=24, install=fake_renderer()
    )
    assert ffmpeg.probe(result.path).video_frames == 3
    assert ffmpeg.probe(result.path).fps == 24


def test_off_by_one_duration_is_refused_before_install_or_launch(
    tmp_path: Path,
    fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comp = _composition(tmp_path / "bad-duration", frames=30, duration=1.001)
    renderer = fake_renderer()

    def unexpected() -> RendererInstall:
        pytest.fail("preflight reached installation")

    monkeypatch.setattr(hyperframes, "ensure_installed", unexpected)
    for install in (None, renderer):
        with pytest.raises(YapError) as caught:
            render_layer(comp, tmp_path / "out.mov", frames=30, variables={}, install=install)
        assert caught.value.code == "layer_frame_mismatch"
    assert not (comp / "invoked.json").exists()
    assert not (comp / "variables.json").exists()


@pytest.mark.parametrize(
    "attribute", ["data-no-timeline", "data-duration", "data-width", "data-height"]
)
def test_preflight_requires_root_attributes(tmp_path: Path, attribute: str) -> None:
    comp = _composition(tmp_path / "missing")
    html = (comp / "index.html").read_text()
    # Alter only the attribute name so the parser must inspect real root attributes.
    (comp / "index.html").write_text(html.replace(attribute, "ignored"))
    with pytest.raises(YapError) as caught:
        preflight(comp, 30)
    assert caught.value.code == "layer_composition"
    assert caught.value.suggested_action


@pytest.mark.parametrize("duration", ["nan", "inf", "-1"])
def test_preflight_rejects_invalid_duration(tmp_path: Path, duration: str) -> None:
    comp = _composition(tmp_path / "invalid", duration=float(duration))
    with pytest.raises(YapError) as caught:
        preflight(comp, 30)
    assert caught.value.code == "layer_composition"


@pytest.mark.parametrize(
    "asset",
    [
        '<script src="https://cdn.example/gsap.min.js"></script>',
        '<script src="//cdn.example/gsap.min.js"></script>',
        '<link href="http://example/style.css">',
        '<img src="file:///outside.png">',
        '<style>.box{background:url(https://example/image.png)}</style>',
        '<style>@import "https://example/style.css";</style>',
        '<style>@import url("https://example/style.css");</style>',
        '<div style="background:url(//x/y.png)"></div>',
        '<img src="../outside.png">',
        '<img src="/abs.png">',
        '<img src="folder/../outside.png">',
        '<img src="folder\\outside.png">',
        '<video poster="ftp://example/poster.png"></video>',
        '<img srcset="image.png 1x, https://example/image.png 2x">',
        '<img srcset="data:image/png;base64,AAAA 1x, //example/image.png 2x">',
        '<svg><use xlink:href="https://example/image.svg#clip"/></svg>',
        '<div data-composition-src="ws://example/composition"></div>',
        '<img src="blob:example">',
        '<img src="%2e%2e/outside.png">',
        '<img src="%2Fabs.png">',
        r'<style>.box{background:url(https\3a //example/image.png)}</style>',
        '<object data="https://x"></object>',
        '<embed src="//x">',
        '<body background="http://x">',
        '<form action="https://x"></form>',
        '<button formaction="https://x"></button>',
        '<link imagesrcset="https://x 1x">',
        '<div style="background:url(https://x)"></div>',
        '<div data-x="https://x"></div>',
        '<rect fill="url(https://x#g)"/>',
        '<meta http-equiv="refresh" content="0;url=//evil.example">',
        '<base href="sub/">',
        '<base>',
        '<iframe srcdoc="&lt;img src=x&gt;"></iframe>',
        '<iframe srcdoc></iframe>',
        '<style>.box{content:"url(https://ignored)"}</style>',
        '<style>.box{background:image-set("https://x/a.png" 1x)}</style>',
        '<style>.box{background:-webkit-image-set("//x/a.png" 1x)}</style>',
        '<style>.box{background:image-set("../a.png" 1x)}</style>',
        '<style>.box{content:"prefix //evil.example/a"}</style>',
        '<style>.box{content:https://x}</style>',
        '<style>.box{content://evil.example/a}</style>',
        '<div data-x="custom+scheme://x"></div>',
        '<svg filter="url(//x/g)" clip-path="url(../g)" mask="url(/g)"/>',
    ],
)
def test_external_assets_are_refused_before_install_or_launch(
    tmp_path: Path,
    fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch,
    asset: str,
) -> None:
    comp = _composition(tmp_path / "external")
    source = comp / "index.html"
    source.write_text(source.read_text().replace("</body>", asset + "</body>"))
    renderer = fake_renderer()

    def unexpected() -> RendererInstall:
        pytest.fail("external asset preflight reached installation")

    monkeypatch.setattr(hyperframes, "ensure_installed", unexpected)
    with pytest.raises(YapError) as caught:
        preflight(comp, 30)
    assert caught.value.code == "layer_external_asset"
    assert caught.value.error_type == "validation_error"
    assert "index.html:1:" in caught.value.message
    assert "relative path" in (caught.value.suggested_action or "")
    for install in (None, renderer):
        with pytest.raises(YapError) as caught:
            render_layer(comp, tmp_path / "out.mov", frames=30, variables={}, install=install)
        assert caught.value.code == "layer_external_asset"
    assert not (comp / "invoked.json").exists()
    assert not (comp / "variables.json").exists()


@pytest.mark.parametrize(
    "asset",
    [
        '<link href="fonts/x.woff2">',
        '<img src="data:image/png;base64,AAAA">',
        '<svg><use xlink:href="#clip"/></svg>',
        '<img src="image.png">',
        '<video poster="image.png"></video>',
        '<img srcset="data:image/png;base64,AAAA 1x, image.png 2x">',
        '<img srcset="image.png, image2.png 2x">',
        '<div data-composition-src="nested/index.html"></div>',
        '<style>@font-face{src:url("fonts/x.woff2")}</style>',
        '<style>@import "styles/local.css"; @import url(styles/other.css);</style>',
        '<div style="clip-path:url(#clip);background:url(image.png)"></div>',
        '<style>/* url(https://example/ignored) */ .box{color:red}</style>',
        '<style>/* https://x //x @import "https://x" image-set("https://x" 1x) */</style>',
        '<style>/* https://x unclosed comment</style>',
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">'
        '<rect fill="url(#grad)"/></svg>',
        '<div data-x="folder//name"></div>',
        '<style>.box{background:image-set("a.png" 1x, url(b.png) 2x)}</style>',
        '<style>.box{background:-webkit-image-set("data:image/png;base64,AAAA" 1x)}</style>',
        '<style>.box{background:image-set("a.png" 1x type("image/png"))}</style>',
        '<img src="data:image/svg+xml,&lt;svg xmlns=\'http://www.w3.org/2000/svg\'&gt;">',
        '<script>\n// plain comment\nwindow.localValue = 1;</script>',
        '<div onload="window.localValue = 1"></div>',
    ],
)
def test_preflight_allows_local_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asset: str,
) -> None:
    comp = _composition(tmp_path / "local")
    source = comp / "index.html"
    source.write_text(source.read_text().replace("</body>", asset + "</body>"))

    def unexpected() -> RendererInstall:
        pytest.fail("local preflight reached installation")

    monkeypatch.setattr(hyperframes, "ensure_installed", unexpected)
    assert preflight(comp, 30) == layer_duration_s(30)
    assert not (comp / "variables.json").exists()


def _assert_preflight_refusal(
    comp: Path, renderer: RendererInstall, monkeypatch: pytest.MonkeyPatch,
    *, code: str, location: str,
) -> YapError:
    """Check that both entry points refuse a source before installation or writes."""
    def unexpected() -> RendererInstall:
        pytest.fail("offline preflight reached installation")

    monkeypatch.setattr(hyperframes, "ensure_installed", unexpected)
    with pytest.raises(YapError) as caught:
        preflight(comp, 30)
    error = caught.value
    assert error.code == code and error.error_type == "validation_error"
    assert location in error.message and error.suggested_action
    for install in (None, renderer):
        with pytest.raises(YapError) as caught:
            render_layer(comp, comp.parent / "out.mov", frames=30, variables={}, install=install)
        assert caught.value.code == code
        assert location in caught.value.message
    assert not (comp / "invoked.json").exists()
    assert not (comp / "variables.json").exists()
    assert not (comp.parent / "out.mov").exists()
    assert not list(comp.glob(".yap_tmp_*"))
    return error


@pytest.mark.parametrize(
    "markup",
    [
        '<img src="local.png" alt="Use url(../icon.png) in CSS">',
        '<div title="url(/abs.png)"></div>',
        '<div aria-label="url(x)"></div>',
        '<div data-note="url(../x)"></div>',
        '<div data-x="&quot;url(../outside.png)&quot;"></div>',
    ],
)
def test_preflight_allows_css_mentions_in_text_attributes(tmp_path: Path, markup: str) -> None:
    comp = _composition(tmp_path / "text-attributes")
    source = comp / "index.html"
    source.write_text(source.read_text().replace("</body>", markup + "</body>"))
    assert preflight(comp, 30) == layer_duration_s(30)


@pytest.mark.parametrize(
    "markup",
    [
        '<rect fill="url(../g)"/>',
        '<rect STROKE="url(../g)"/>',
        '<g clip-path="url(/g)"></g>',
        '<g FILTER="url(../g)"></g>',
        '<g mask="url(/g)"></g>',
        '<path marker-start="url(../g)"/>',
        '<path MARKER-MID="url(/g)"/>',
        '<path marker-end="url(//x/m)"/>',
        '<g cursor="url(../g)"></g>',
        '<div style="background:url(../x.png)"></div>',
        '<div STYLE="background:url(/x.png)"></div>',
        '<div title="see https://x"></div>',
    ],
)
def test_preflight_refuses_css_attribute_assets_and_text_url_literals(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch, markup: str,
) -> None:
    comp = _composition(tmp_path / "css-attributes")
    source = comp / "index.html"
    source.write_text(source.read_text().replace("</body>", f"\n{markup}</body>"))
    _assert_preflight_refusal(
        comp, fake_renderer(), monkeypatch, code="layer_external_asset", location="index.html:2:"
    )


@pytest.mark.parametrize(
    ("prefix", "encoding", "content"),
    [
        (b'@charset "ISO-8859-1";', "cp1252", "\u00e9\u20ac"),
        (b'@charset "latin1";', "cp1252", "\u00e9\u20ac"),
        (b'@charset "l1";', "cp1252", "\u00e9\u20ac"),
        (b'@charset "ascii";', "cp1252", "\u00e9\u20ac"),
        (b'@charset "us-ascii";', "cp1252", "\u00e9\u20ac"),
        (b'@charset "windows-1252";', "cp1252", "\u00e9\u20ac"),
        (b'@charset "utf-16";', "utf-8", "\u00e9"),
        (b'@charset "utf-16be";', "utf-8", "\u00e9"),
        (b'@charset "utf-16le";', "utf-8", "\u00e9"),
        (b'@charset "unknown-label";', "utf-8", "\u00e9"),
        (b'@charset "windows-1251";', "cp1251", "\u0416"),
        (b"\xef\xbb\xbf", "utf-8", "\u00e9"),
        (b"\xff\xfe", "utf-16-le", "\u00e9"),
        (b"\xfe\xff", "utf-16-be", "\u00e9"),
    ],
)
@pytest.mark.parametrize("remote", [False, True])
def test_preflight_decodes_and_scans_stylesheet_encodings(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch, prefix: bytes, encoding: str, content: str, remote: bool,
) -> None:
    comp = _composition(tmp_path / "css-encoding")
    index = comp / "index.html"
    index.write_text(index.read_text().replace(
        "</head>", '<link rel="stylesheet" href="style.css"></head>'
    ))
    asset = "https://x" if remote else "local.png"
    # A contradictory declaration also checks that each BOM takes precedence.
    charset = (
        '@charset "ascii";\n' if prefix in {b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff"} else "\n"
    )
    css = f'{charset}.box{{content:"{content}";background:url({asset})}}'
    data = prefix + css.encode(encoding)
    (comp / "style.css").write_bytes(data)
    assert content in hyperframes._decode_css(data)
    if remote:
        _assert_preflight_refusal(
            comp, fake_renderer(), monkeypatch,
            code="layer_external_asset", location="style.css:2:",
        )
    else:
        assert preflight(comp, 30) == layer_duration_s(30)


@pytest.mark.parametrize(
    "declaration",
    [
        b' @charset "ISO-8859-1";', b"@charset 'ISO-8859-1';",
        b'@charset  "ISO-8859-1";', b'@CHARSET "ISO-8859-1";',
        b'@charset "ISO-8859-1" ;', b'@charset "\xe9";',
    ],
)
def test_css_charset_requires_exact_ascii_syntax(declaration: bytes) -> None:
    with pytest.raises(UnicodeDecodeError):
        hyperframes._decode_css(declaration + b'.box{content:"\xe9"}')


@pytest.mark.parametrize(
    ("filename", "data", "encoding"),
    [
        ("index.html", b"\xff", "utf-8"),
        ("nested/a.html", b"\xff", "utf-8"),
        ("nested/a.htm", b"\xff", "utf-8"),
        ("nested/a.svg", b"\xff", "utf-8"),
        ("nested/style.css", b'.box{content:"\xff"}', "utf-8"),
        ("nested/app.js", b"\xff", "utf-8"),
        ("nested/app.mjs", b"\xff", "utf-8"),
        ("nested/app.cjs", b"\xff", "utf-8"),
        ("style.css", b'@charset "unknown-label";\xff', "utf-8"),
        ("style.css", b'@charset "shift_jis";\x81', "shift_jis"),
        ("style.css", b'@charset "ascii";\x81', "cp1252"),
        ("style.css", b'@charset "base64_codec";eA==', "base64"),
        ("style.css", b"\xff\xfe\x00", "utf-16"),
    ],
)
def test_preflight_refuses_invalid_source_encodings_before_install_or_writes(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch, filename: str, data: bytes, encoding: str,
) -> None:
    comp = _composition(tmp_path / "invalid-encoding")
    source = comp / filename
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(data)
    error = _assert_preflight_refusal(
        comp, fake_renderer(), monkeypatch, code="layer_source_encoding", location=f"{filename}:"
    )
    assert repr(encoding) in error.message
    assert "UTF-8" in (error.suggested_action or "")
    assert "@charset" in (error.suggested_action or "")


@pytest.mark.parametrize(
    "filename", ["index.html", "a.html", "a.htm", "a.svg", "a.js", "a.mjs", "a.cjs"]
)
@pytest.mark.parametrize(
    ("bom", "encoding"),
    [(b"\xef\xbb\xbf", "utf-8"), (b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be")],
)
@pytest.mark.parametrize("remote", [False, True])
def test_preflight_honours_boms_in_markup_and_scripts(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch, filename: str, bom: bytes, encoding: str, remote: bool,
) -> None:
    comp = _composition(tmp_path / "source-bom")
    source = comp / filename
    script = source.suffix in {".js", ".mjs", ".cjs"}
    asset = "https://x" if remote else "local.png"
    text = (
        f'const label = "\u00e9"; const asset = "{asset}";' if script
        else f'<img src="{asset}" alt="\u00e9">'
    )
    if filename == "index.html":
        text = source.read_text().replace("</body>", text + "</body>")
    source.write_bytes(bom + text.encode(encoding))
    if remote:
        _assert_preflight_refusal(
            comp, fake_renderer(), monkeypatch,
            code="layer_network_script" if script else "layer_external_asset",
            location=f"{filename}:1:",
        )
    else:
        assert preflight(comp, 30) == layer_duration_s(30)


@pytest.mark.parametrize(
    "attribute",
    [
        "src", "href", "xlink:href", "data", "poster", "background", "action", "formaction",
        "cite", "longdesc", "usemap", "manifest", "codebase", "lowsrc", "dynsrc", "icon",
        "profile", "data-composition-src", "archive", "ping", "srcset", "imagesrcset",
    ],
)
def test_url_attributes_are_checked_on_any_element(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch, attribute: str,
) -> None:
    comp = _composition(tmp_path / "attributes")
    source = comp / "index.html"
    value = (
        "local.png https://x" if attribute in {"archive", "ping"}
        else "local.png 1x, https://x 2x" if attribute in {"srcset", "imagesrcset"}
        else "https://x"
    )
    source.write_text(source.read_text().replace(
        "</body>", f'\n<div {attribute}="{value}"></div></body>'
    ))
    _assert_preflight_refusal(
        comp, fake_renderer(), monkeypatch, code="layer_external_asset", location="index.html:2:"
    )


@pytest.mark.parametrize("attribute", ["archive", "ping", "srcset", "imagesrcset"])
def test_preflight_allows_local_url_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attribute: str,
) -> None:
    comp = _composition(tmp_path / "lists")
    source = comp / "index.html"
    value = (
        "a.png b.png #clip" if attribute in {"archive", "ping"}
        else "data:image/png;base64,AAAA 1x, b.png 2x"
    )
    source.write_text(source.read_text().replace(
        "</body>", f'<div {attribute}="{value}"></div></body>'
    ))

    def unexpected() -> RendererInstall:
        pytest.fail("local list preflight reached installation")

    monkeypatch.setattr(hyperframes, "ensure_installed", unexpected)
    assert preflight(comp, 30) == layer_duration_s(30)
    assert not (comp / "variables.json").exists()


@pytest.mark.parametrize("prefix", ["", "prefix ", "x=", "x(", "x,", "x;", "'", '"'])
def test_other_attributes_refuse_protocol_relative_url_boundaries(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch, prefix: str,
) -> None:
    comp = _composition(tmp_path / "boundaries")
    source = comp / "index.html"
    value = (prefix + "//evil.example/a").replace('"', "&quot;")
    source.write_text(source.read_text().replace(
        "</body>", f'\n<div data-x="{value}"></div></body>'
    ))
    _assert_preflight_refusal(
        comp, fake_renderer(), monkeypatch, code="layer_external_asset", location="index.html:2:"
    )


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("style.css", "/* local stylesheet */\n@import url(https://x/y.css);"),
        ("sub/a.css", "/* nested stylesheet */\n.box{background:url(//x/y.png)}"),
        ("sub/a.css", '/* nested stylesheet */\n.box{background:image-set("https://x/a.png" 1x)}'),
        ("sub/a.css", '/* nested stylesheet */\n.box{content:"https://x/a"}'),
        ("sub/a.css", '/* nested stylesheet */\n@import "../a.css";'),
        ("nested/index.html", '<!-- nested markup -->\n<img src="https://x/a.png">'),
        ("nested/a.htm", '<!-- nested markup -->\n<form action="https://x"></form>'),
        ("nested/a.svg", '<!-- nested markup -->\n<rect fill="url(https://x#g)"/>'),
        ("nested/a.svg", '<!-- nested markup -->\n<image href="//x/a.png"/>'),
        ("nested/a.html", '<!-- nested markup -->\n<base href="sub/">'),
    ],
)
def test_preflight_scans_all_markup_and_stylesheets(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch, filename: str, content: str,
) -> None:
    comp = _composition(tmp_path / "tree")
    index = comp / "index.html"
    index.write_text(index.read_text().replace("</head>", '<link href="style.css"></head>'))
    source = comp / filename
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(content)
    _assert_preflight_refusal(
        comp, fake_renderer(), monkeypatch,
        code="layer_external_asset", location=f"{filename}:2:",
    )


@pytest.mark.parametrize(
    ("markup", "line"),
    [
        ('<img\nsrc=\n"https://x">', 4),
        ('<img src="\nhttps://x">', 3),
        ('<img srcset="local.png 1x,\n//x/a.png 2x">', 3),
        ('<div\nstyle=\n"background:url(https://x)"></div>', 4),
        ('<style>\n/* https://ignored */\n.box{background:url(https://x)}</style>', 4),
        ('<style>\n.box{background:image-set(\n"https://x/a.png" 1x)}</style>', 4),
        ('<div\ndata-x=\n"prefix https://x"></div>', 4),
    ],
)
def test_external_asset_diagnostics_track_multiline_sources(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch, markup: str, line: int,
) -> None:
    comp = _composition(tmp_path / "lines")
    source = comp / "index.html"
    source.write_text(source.read_text().replace("</body>", f"\n{markup}</body>"))
    _assert_preflight_refusal(
        comp, fake_renderer(), monkeypatch,
        code="layer_external_asset", location=f"index.html:{line}:",
    )


@pytest.mark.parametrize(
    ("markup", "code"),
    [("<script>fetch('x')", "layer_network_script"),
     ("<style>.box{background:url(https://x)}", "layer_external_asset")],
)
def test_preflight_scans_unclosed_script_and_style_text(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch, markup: str, code: str,
) -> None:
    comp = _composition(tmp_path / "unclosed")
    source = comp / "index.html"
    source.write_text(source.read_text() + "\n" + markup)
    _assert_preflight_refusal(
        comp, fake_renderer(), monkeypatch, code=code, location="index.html:2:",
    )


def _close_without_flushing_cdata(parser: HTMLParser) -> None:
    parser.goahead(0)


@pytest.mark.parametrize(
    ("markup", "code"),
    [("<script>fetch('x')", "layer_network_script"),
     ("<style>.box{background:url(https://x)}", "layer_external_asset")],
)
def test_preflight_scans_unclosed_text_without_eof_flush(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch, markup: str, code: str,
) -> None:
    monkeypatch.setattr(HTMLParser, "close", _close_without_flushing_cdata)
    collected: list[str] = []

    class TextParser(HTMLParser):
        def handle_data(self, data: str) -> None:
            collected.append(data)

    parser = TextParser()
    parser.feed(markup)
    parser.close()
    assert collected == []
    assert parser.rawdata == markup.partition(">")[2]

    comp = _composition(tmp_path / "unclosed-no-flush")
    source = comp / "index.html"
    source.write_text(source.read_text() + "\n" + markup)
    _assert_preflight_refusal(
        comp, fake_renderer(), monkeypatch, code=code, location="index.html:2:",
    )


@pytest.mark.parametrize("legacy_close", [False, True])
def test_preflight_scans_closed_script_and_unclosed_local_style_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy_close: bool,
) -> None:
    if legacy_close:
        monkeypatch.setattr(HTMLParser, "close", _close_without_flushing_cdata)
    scanned: list[tuple[str, str]] = []
    handle_endtag = hyperframes._CompositionParser.handle_endtag

    def record_endtag(parser: hyperframes._CompositionParser, tag: str) -> None:
        if tag == parser._text_tag:
            scanned.append((tag, "".join(parser._text)))
        handle_endtag(parser, tag)

    monkeypatch.setattr(hyperframes._CompositionParser, "handle_endtag", record_endtag)
    comp = _composition(tmp_path / "unclosed-local")
    (comp / "fonts").mkdir()
    (comp / "fonts" / "a.woff2").write_bytes(b"local font")
    source = comp / "index.html"
    script = "window.a = 1;"
    style = "@font-face{src:url(fonts/a.woff2)}"
    source.write_text(source.read_text() + f"\n<script>{script}</script><style>{style}")

    assert preflight(comp, 30) == layer_duration_s(30)
    assert scanned.count(("script", script)) == 1
    assert scanned.count(("style", style)) == 1


@pytest.mark.parametrize("target_kind", ["file", "directory", "missing"])
def test_preflight_refuses_unreferenced_external_symlinks(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch, target_kind: str,
) -> None:
    comp = _composition(tmp_path / "symlink")
    target = tmp_path / "outside"
    if target_kind == "file":
        target.write_bytes(b"caller file")
    elif target_kind == "directory":
        target.mkdir()
        (target / "a.css").write_text(".box{color:red}")
    link = comp / "external"
    link.symlink_to(target, target_is_directory=target_kind == "directory")
    original = os.readlink(link)
    _assert_preflight_refusal(
        comp, fake_renderer(), monkeypatch, code="layer_external_asset", location="external:1:"
    )
    assert link.is_symlink() and os.readlink(link) == original
    if target_kind == "file":
        assert target.read_bytes() == b"caller file"
    elif target_kind == "directory":
        assert (target / "a.css").read_text() == ".box{color:red}"
    else:
        assert not target.exists()


def test_preflight_allows_local_tree_assets_and_skips_root_gsap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    comp = _composition(tmp_path / "local-tree")
    index = comp / "index.html"
    index.write_text(index.read_text().replace("</head>", '<link href="style.css"></head>'))
    (comp / "fonts").mkdir()
    (comp / "fonts" / "a.woff2").write_bytes(b"local font")
    (comp / "style.css").write_text(
        '/* url(https://x) */\n@font-face{src:url("fonts/a.woff2")}\n'
        '@import "nested/a.css"; .box{background:image-set("a.png" 1x, "b.png" 2x)}'
    )
    nested = comp / "nested"
    nested.mkdir()
    (nested / "a.css").write_text('.box{background:url("a.png");fill:url(#grad)}')
    (nested / "a.png").write_bytes(b"local image")
    (nested / "a.htm").write_text('<img src="a.png"><script src="app.js"></script>')
    (nested / "a.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><rect fill="url(#grad)"/></svg>'
    )
    for extension in ("js", "mjs", "cjs"):
        (nested / f"app.{extension}").write_text("// plain comment\nwindow.localValue = 1;")
    (comp / "alias.css").symlink_to(nested / "a.css")
    (nested / "loop").symlink_to(comp, target_is_directory=True)
    payload = b'/* https://x */ fetch("/x");'
    (comp / GSAP_SCRIPT).write_bytes(payload)
    monkeypatch.setattr(hyperframes, "GSAP_SHA256", hashlib.sha256(payload).hexdigest())

    def unexpected() -> RendererInstall:
        pytest.fail("local tree preflight reached installation")

    monkeypatch.setattr(hyperframes, "ensure_installed", unexpected)
    assert preflight(comp, 30) == layer_duration_s(30)
    assert not (comp / "variables.json").exists()


def test_render_plain_layer_preserves_unrelated_root_gsap(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comp = _composition(tmp_path / "unrelated-gsap", frames=3)
    script = comp / GSAP_SCRIPT
    payload = b"const callerValue = 1;\n"
    script.write_bytes(payload)
    original_inode = script.stat().st_ino

    def unexpected(*args: object) -> None:
        pytest.fail("plain layer reached GSAP ownership logic")

    monkeypatch.setattr(hyperframes, "_existing_gsap", unexpected)
    monkeypatch.setattr(hyperframes, "_copy_gsap", unexpected)
    monkeypatch.setattr(hyperframes, "gsap_file", unexpected)
    result = render_layer(
        comp, tmp_path / "out.mov", frames=3, variables={}, install=fake_renderer()
    )
    invoked = json.loads((comp / "invoked.json").read_text())
    assert invoked["gsap_inode"] == original_inode == script.stat().st_ino
    assert invoked["gsap_sha256"] == hashlib.sha256(payload).hexdigest()
    assert script.read_bytes() == payload
    assert ffmpeg.probe(result.path).video_frames == 3
    assert not (comp / "variables.json").exists()


@pytest.mark.parametrize("symlink", [False, True])
def test_preflight_scans_unrelated_root_gsap(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch, symlink: bool,
) -> None:
    comp = _composition(tmp_path / "unscanned-gsap")
    (comp / "app.mjs").write_text('import "./gsap.min.js";')
    payload = b'\nfetch("/x");'
    script = comp / GSAP_SCRIPT
    if symlink:
        target = comp / "caller.txt"
        target.write_bytes(payload)
        script.symlink_to(target)
        monkeypatch.setattr(hyperframes, "GSAP_SHA256", hashlib.sha256(payload).hexdigest())
    else:
        script.write_bytes(payload)
    _assert_preflight_refusal(
        comp, fake_renderer(), monkeypatch,
        code="layer_network_script", location=f"{GSAP_SCRIPT}:2:",
    )
    assert script.read_bytes() == payload


@pytest.mark.parametrize(
    "code",
    [
        'fetch("/x")', "fetch \n ('x')", "new XMLHttpRequest()", "new WebSocket('x')",
        "new EventSource('x')", "navigator.sendBeacon('x')", "importScripts('x.js')",
        "new Worker('x.js')", "new SharedWorker ('x.js')", "new RTCPeerConnection()",
        "navigator.serviceWorker.register('x.js')", 'import("./m.js")',
        'import \n ("./m.js")', '"https://x"', "'http://x'", "'ws://x'", "'wss://x'",
        "'ftp://x'", "'//cdn.example/a.js'", '"//cdn.example/a.js"', "`//cdn.example/a.js`",
        '// fetch("x")', '/* https://x */',
    ],
)
def test_preflight_refuses_common_inline_network_patterns(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch, code: str,
) -> None:
    comp = _composition(tmp_path / "inline")
    source = comp / "index.html"
    source.write_text(source.read_text().replace("</body>", f"\n<script>\n{code}</script></body>"))
    error = _assert_preflight_refusal(
        comp, fake_renderer(), monkeypatch, code="layer_network_script", location="index.html:3:"
    )
    assert "network requests" in (error.suggested_action or "")
    assert "composition folder" in (error.suggested_action or "")


@pytest.mark.parametrize("attribute", ["onload", "onclick", "onerror"])
def test_preflight_refuses_network_event_handlers(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch, attribute: str,
) -> None:
    comp = _composition(tmp_path / "handlers")
    source = comp / "index.html"
    source.write_text(source.read_text().replace(
        "</body>", f'\n<div\n{attribute}="fetch(\'x\')"></div></body>'
    ))
    error = _assert_preflight_refusal(
        comp, fake_renderer(), monkeypatch, code="layer_network_script", location="index.html:3:"
    )
    assert repr("fetch(") in error.message


@pytest.mark.parametrize(
    ("filename", "code"),
    [
        ("app.js", "'//cdn.example/a.js'"),
        ("nested/app.mjs", 'import("./m.js")'),
        ("nested/app.cjs", "new WebSocket('x')"),
        ("nested/gsap.min.js", '"https://x"'),
        ("nested/a.html", '<script>fetch("x")</script>'),
        ("nested/a.htm", '<div onload="fetch(\'x\')"></div>'),
        ("nested/a.svg", '<script>navigator.sendBeacon("x")</script>'),
    ],
)
def test_preflight_scans_network_code_in_every_script_file(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch, filename: str, code: str,
) -> None:
    comp = _composition(tmp_path / "scripts")
    source = comp / filename
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(f"\n{code}")
    _assert_preflight_refusal(
        comp, fake_renderer(), monkeypatch,
        code="layer_network_script", location=f"{filename}:2:",
    )


@pytest.mark.parametrize(
    "code",
    [
        "// plain comment\nwindow.localValue = 1;", "const prefetch = () => {}; prefetch();",
        "const myWebSocket = 1;", "const importScriptsLocal = 1;", "const serviceWorkers = 1;",
        "const path = 'folder//a.js';", 'const url = "./m.js";',
    ],
)
def test_preflight_allows_local_scripts_and_respects_word_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: str,
) -> None:
    comp = _composition(tmp_path / "safe-scripts")
    source = comp / "index.html"
    source.write_text(source.read_text().replace("</body>", f"<script>{code}</script></body>"))
    (comp / "app.js").write_text(code)

    def unexpected() -> RendererInstall:
        pytest.fail("local script preflight reached installation")

    monkeypatch.setattr(hyperframes, "ensure_installed", unexpected)
    assert preflight(comp, 30) == layer_duration_s(30)
    assert not (comp / "variables.json").exists()


def test_external_asset_error_quotes_a_truncated_value(tmp_path: Path) -> None:
    value = "https://example/" + "x" * 200
    comp = _composition(tmp_path / "long")
    source = comp / "index.html"
    source.write_text(source.read_text().replace("</body>", f'<img src="{value}"></body>'))
    with pytest.raises(YapError) as caught:
        preflight(comp, 30)
    assert repr(value[:120]) in caught.value.message
    assert value not in caught.value.message
    assert "index.html:1:" in caught.value.message


def test_network_script_error_quotes_a_truncated_pattern(tmp_path: Path) -> None:
    value = "https://x/" + "a" * 200
    comp = _composition(tmp_path / "long-script")
    source = comp / "app.js"
    source.write_text(f'// local comment\n"{value}"')
    with pytest.raises(YapError) as caught:
        preflight(comp, 30)
    assert caught.value.code == "layer_network_script"
    assert "app.js:2:" in caught.value.message
    assert repr(value[:120]) in caught.value.message
    assert value not in caught.value.message


@pytest.mark.parametrize("uses_gsap", [True, False])
def test_preflight_enforces_the_gsap_timeline_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, uses_gsap: bool,
) -> None:
    comp = (
        _gsap_composition(tmp_path / "timeline", frames=30)
        if uses_gsap else _composition(tmp_path / "timeline")
    )
    source = comp / "index.html"
    html = source.read_text()
    source.write_text(
        html.replace('data-composition-id="test"', 'data-composition-id="test" data-no-timeline')
        if uses_gsap else html.replace(" data-no-timeline", "")
    )

    def unexpected() -> RendererInstall:
        pytest.fail("timeline preflight reached installation")

    monkeypatch.setattr(hyperframes, "ensure_installed", unexpected)
    with pytest.raises(YapError) as caught:
        preflight(comp, 30)
    assert caught.value.code == "layer_composition"
    assert ("Remove" if uses_gsap else "Set") in (caught.value.suggested_action or "")
    with pytest.raises(YapError) as caught:
        render_layer(comp, tmp_path / "out.mov", frames=30, variables={})
    assert caught.value.code == "layer_composition"
    assert not (comp / "variables.json").exists()


@pytest.mark.parametrize("existing", [False, True])
def test_render_copies_verified_gsap_and_cleans_up(
    tmp_path: Path,
    fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
) -> None:
    renderer = fake_renderer()
    gsap = renderer.prefix / "node_modules" / "gsap" / "dist" / GSAP_SCRIPT
    gsap.parent.mkdir(parents=True)
    payload = b"/* fake pinned GSAP https://x */\n"
    gsap.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(hyperframes, "GSAP_SHA256", digest)
    log = tmp_path / "check-calls.jsonl"
    monkeypatch.setenv("YAP_TEST_CHECK_LOG", str(log))
    assert gsap_file(renderer) == gsap
    comp = _gsap_composition(tmp_path / "gsap", frames=3)
    script = comp / GSAP_SCRIPT
    if existing:
        script.write_bytes(payload)
    original_inode = script.stat().st_ino if existing else None
    assert preflight(comp, 3) == layer_duration_s(3)
    result = render_layer(comp, tmp_path / "out.mov", frames=3, variables={}, install=renderer)
    invoked = json.loads((comp / "invoked.json").read_text())
    assert invoked["gsap_exists"] is True and invoked["gsap_sha256"] == digest
    assert ffmpeg.probe(result.path).video_frames == 3
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [call["args"][0] for call in calls] == ["lint", "check"]
    assert all(call["gsap_exists"] and not call["variables_exist"] for call in calls)
    assert gsap.read_bytes() == payload
    if existing:
        assert script.read_bytes() == payload
        assert script.stat().st_ino == original_inode == invoked["gsap_inode"]
    else:
        assert not script.exists()
    assert not (comp / "variables.json").exists()
    assert not list(comp.glob(".layer-*.mov"))
    assert not list(comp.glob(".yap_tmp_*"))


@pytest.mark.parametrize(
    "kind", ["different", "symlink", "local_symlink", "dangling_symlink", "directory"]
)
def test_render_preserves_conflicting_gsap_before_any_writes(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    comp = _gsap_composition(tmp_path / "conflict", frames=30)
    script = comp / GSAP_SCRIPT
    target = (comp if kind == "local_symlink" else tmp_path) / "caller.js"
    payload = b"caller-owned GSAP bytes\n"
    if kind == "different":
        script.write_bytes(payload)
    elif kind in {"symlink", "local_symlink", "dangling_symlink"}:
        if kind != "dangling_symlink":
            target.write_bytes(payload)
        script.symlink_to(target)
    else:
        script.mkdir()
        (script / "caller.txt").write_bytes(payload)
    original = script.lstat()

    def unexpected() -> RendererInstall:
        pytest.fail("GSAP conflict reached installation")

    monkeypatch.setattr(hyperframes, "ensure_installed", unexpected)
    for install in (None, fake_renderer()):
        with pytest.raises(YapError) as caught:
            render_layer(comp, tmp_path / "out.mov", frames=30, variables={}, install=install)
        assert caught.value.code == "layer_gsap_conflict"
        assert caught.value.error_type == "validation_error"
        assert "Remove or rename" in (caught.value.suggested_action or "")
        assert "pinned GSAP" in (caught.value.suggested_action or "")
        assert script.lstat() == original
        if kind == "different":
            assert script.read_bytes() == payload
        elif kind in {"symlink", "local_symlink", "dangling_symlink"}:
            assert script.is_symlink() and os.readlink(script) == str(target)
            if kind != "dangling_symlink":
                assert target.read_bytes() == payload
            else:
                assert not target.exists()
        else:
            assert (script / "caller.txt").read_bytes() == payload
        assert not (comp / "variables.json").exists()
        assert not (comp / "invoked.json").exists()
        assert not list(comp.glob(".yap_tmp_*"))
    assert not (tmp_path / "out.mov").exists()


@pytest.mark.parametrize("pinned", [False, True])
def test_atomic_gsap_copy_preserves_a_file_created_during_staging(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch, pinned: bool,
) -> None:
    renderer = fake_renderer()
    gsap = renderer.prefix / "node_modules" / "gsap" / "dist" / GSAP_SCRIPT
    gsap.parent.mkdir(parents=True)
    gsap.write_bytes(b"fake pinned GSAP")
    monkeypatch.setattr(hyperframes, "GSAP_SHA256", hashlib.sha256(gsap.read_bytes()).hexdigest())
    comp = _gsap_composition(tmp_path / "atomic-gsap", frames=3)
    script = comp / GSAP_SCRIPT
    payload = gsap.read_bytes() if pinned else b"caller file appeared during staging"
    original_copy = shutil.copyfile

    def copy(source: Path, destination: Path) -> Path:
        result = original_copy(source, destination)
        if source == gsap:
            script.write_bytes(payload)
        return result

    monkeypatch.setattr(shutil, "copyfile", copy)
    if pinned:
        result = render_layer(comp, tmp_path / "out.mov", frames=3, variables={}, install=renderer)
        assert ffmpeg.probe(result.path).video_frames == 3
        invoked = json.loads((comp / "invoked.json").read_text())
        assert invoked["gsap_inode"] == script.stat().st_ino
    else:
        with pytest.raises(YapError) as caught:
            render_layer(comp, tmp_path / "out.mov", frames=3, variables={}, install=renderer)
        assert caught.value.code == "layer_gsap_conflict"
        assert not (comp / "invoked.json").exists()
    assert script.read_bytes() == payload
    assert not (comp / "variables.json").exists()
    assert not list(comp.glob(".yap_tmp_*"))


@pytest.mark.parametrize("fault", ["missing", "hash", "directory"])
def test_gsap_failure_precedes_renderer_launch(
    tmp_path: Path,
    fake_renderer: Callable[[str], RendererInstall],
    fault: str,
) -> None:
    renderer = fake_renderer()
    gsap = renderer.prefix / "node_modules" / "gsap" / "dist" / GSAP_SCRIPT
    if fault != "missing":
        gsap.parent.mkdir(parents=True)
        if fault == "hash":
            gsap.write_text("wrong GSAP bytes")
        else:
            gsap.mkdir()
    comp = _gsap_composition(tmp_path / "bad-gsap")
    with pytest.raises(DependencyError) as caught:
        render_layer(comp, tmp_path / "out.mov", frames=29, variables={}, install=renderer)
    assert caught.value.code == "renderer_gsap_missing"
    assert caught.value.error_type == "dependency_error"
    assert str(renderer.prefix) in caught.value.message
    assert str(renderer.prefix) in (caught.value.suggested_action or "")
    assert "run the command again" in (caught.value.suggested_action or "")
    assert not (comp / "invoked.json").exists()
    assert not (comp / "variables.json").exists()
    assert not (comp / GSAP_SCRIPT).exists()


@pytest.mark.parametrize("existing", [False, True])
def test_renderer_failure_cleans_up_gsap(
    tmp_path: Path,
    fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
) -> None:
    renderer = fake_renderer("import sys\nsys.exit(4)\n")
    gsap = renderer.prefix / "node_modules" / "gsap" / "dist" / GSAP_SCRIPT
    gsap.parent.mkdir(parents=True)
    gsap.write_bytes(b"fake GSAP")
    monkeypatch.setattr(hyperframes, "GSAP_SHA256", hashlib.sha256(gsap.read_bytes()).hexdigest())
    comp = _gsap_composition(tmp_path / "failed-gsap")
    script = comp / GSAP_SCRIPT
    if existing:
        script.write_bytes(gsap.read_bytes())
    original_inode = script.stat().st_ino if existing else None
    with pytest.raises(ProcessingError):
        render_layer(comp, tmp_path / "out.mov", frames=29, variables={}, install=renderer)
    if existing:
        assert script.read_bytes() == gsap.read_bytes()
        assert script.stat().st_ino == original_inode
    else:
        assert not script.exists()
    assert not (comp / "variables.json").exists()
    assert not list(comp.glob(".yap_tmp_*"))


def test_duration_helpers_match_renderer_rule() -> None:
    for fps in (24, 25, 30, 50, 60, 120, 240):
        for frames in (1, 2, 3, 30, 1001, 10000):
            duration = layer_duration_s(frames, fps)
            assert duration == math.floor((frames * 1000 - 1) / fps) / 1000
            assert renderer_frame_count(duration, fps) == frames
    with pytest.raises(ValueError):
        layer_duration_s(0)


@pytest.mark.parametrize(
    "stderr",
    [
        _SANDBOX_STDERR,
        "Error: Failed to launch the browser process!\n",
        "chrome-headless-shell: Operation not permitted in sandbox\n",
        "bootstrap_check_in: Operation not permitted (mach)\n",
    ],
)
def test_blocked_renderer_has_action_and_no_output(
    tmp_path: Path,
    fake_renderer: Callable[[str], RendererInstall],
    stderr: str,
) -> None:
    renderer = fake_renderer("import sys\nsys.stderr.write(" + repr(stderr) + ")\nsys.exit(1)\n")
    comp = _composition(tmp_path / "blocked")
    out = tmp_path / "out.mov"
    with pytest.raises(RendererBlockedError) as caught:
        render_layer(comp, out, frames=30, variables={}, install=renderer)
    assert caught.value.code == "renderer_blocked"
    assert "outside the agent sandbox" in (caught.value.suggested_action or "")
    assert not out.exists()
    assert not list(comp.glob(".yap_tmp_*"))
    assert not (comp / "variables.json").exists()


@pytest.mark.parametrize("stderr", ["encoder failed", "Operation not permitted opening an asset"])
def test_other_renderer_failures_keep_tool_error(
    tmp_path: Path,
    fake_renderer: Callable[[str], RendererInstall],
    stderr: str,
) -> None:
    renderer = fake_renderer("import sys\nsys.stderr.write(" + repr(stderr) + ")\nsys.exit(4)\n")
    comp = _composition(tmp_path / "failed")
    with pytest.raises(ProcessingError) as caught:
        render_layer(comp, tmp_path / "out.mov", frames=30, variables={}, install=renderer)
    assert caught.value.code == "hyperframes_exit_4"
    assert caught.value.full_stderr == stderr


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_page_error_with_successful_exit_refuses_output(
    tmp_path: Path, fake_renderer: Callable[[str], RendererInstall], stream: str,
) -> None:
    renderer = fake_renderer(
        "import sys\nfrom pathlib import Path\n"
        "Path(sys.argv[sys.argv.index('-o') + 1]).write_bytes(b'partial render')\n"
        f"sys.{stream}.write('[Browser:PAGEERROR] injected script failure\\n')\n"
    )
    comp = _composition(tmp_path / "page-error")
    out = tmp_path / "out.mov"
    with pytest.raises(YapError, match="injected script failure") as caught:
        render_layer(comp, out, frames=30, variables={}, install=renderer, checks=False)
    assert caught.value.code == "renderer_script_error"
    assert not out.exists()
    assert not list(comp.glob("*.mov"))
    assert not (comp / "variables.json").exists()


@pytest.mark.parametrize("checks", [True, False])
@pytest.mark.parametrize("error", _LOCAL_SERVER_ERRORS)
def test_render_retries_once_when_local_server_answers_with_another_program(
    tmp_path: Path, retry_renderer: RendererInstall, checks: bool,
    monkeypatch: pytest.MonkeyPatch, error: str,
) -> None:
    monkeypatch.setenv(
        "YAP_TEST_RENDER_STDERR", _LOCAL_SERVER_STDERR.replace("ERR_INVALID_HTTP_RESPONSE", error)
    )
    comp = _gsap_composition(tmp_path / "retry", frames=3)
    result = render_layer(
        comp, tmp_path / "out.mov", frames=3, variables={}, install=retry_renderer,
        checks=checks,
    )
    assert result.frames == ffmpeg.probe(result.path).video_frames == 3
    assert (tmp_path / "render-count").read_text() == "2"
    assert result.render_retries == 1
    log = tmp_path / "check-calls.jsonl"
    if checks:
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        assert [call["args"][0] for call in calls] == ["lint", "check"]
        assert isinstance(result.checks, dict)
        assert result.checks["render_retries"] == [{
            "reason": "renderer_local_server",
            "detail": f"net::{error} at http://localhost:57764/index.html",
        }]
    else:
        assert result.checks is None
        assert not log.exists()
    assert not list(comp.glob(".layer-*.mov"))
    assert not list(comp.glob(".yap_tmp_*"))
    assert not (comp / "variables.json").exists()
    assert not (comp / GSAP_SCRIPT).exists()
    assert not (comp / ".hyperframes").exists()


@pytest.mark.parametrize("error", _LOCAL_SERVER_ERRORS)
def test_render_raises_after_second_local_server_failure(
    tmp_path: Path, retry_renderer: RendererInstall, monkeypatch: pytest.MonkeyPatch,
    error: str,
) -> None:
    stderr = _LOCAL_SERVER_STDERR.replace("ERR_INVALID_HTTP_RESPONSE", error)
    monkeypatch.setenv("YAP_TEST_RENDER_STDERR", stderr)
    monkeypatch.setenv("YAP_TEST_RENDER_FAILURES", "2")
    comp = _gsap_composition(tmp_path / "failed-retry", frames=3)
    out = tmp_path / "out.mov"
    with pytest.raises(ProcessingError) as caught:
        render_layer(comp, out, frames=3, variables={}, install=retry_renderer)
    assert caught.value.code == "hyperframes_exit_1"
    assert caught.value.full_stderr == stderr
    assert (tmp_path / "render-count").read_text() == "2"
    assert not out.exists()
    assert not list(comp.glob(".layer-*.mov"))
    assert not list(comp.glob(".yap_tmp_*"))
    assert not (comp / "variables.json").exists()
    assert not (comp / GSAP_SCRIPT).exists()
    assert not (comp / ".hyperframes").exists()


@pytest.mark.parametrize("stderr", [
    "✗  Render failed\n\n   Protocol error (Page.navigate): Target closed",
    "net::ERR_CONNECTION_RESET at https://example.invalid/",
    "net::ERR_CONNECTION_REFUSED at http://localhost:57344/index.html",
])
def test_other_render_failures_are_not_retried(
    tmp_path: Path, retry_renderer: RendererInstall, monkeypatch: pytest.MonkeyPatch,
    stderr: str,
) -> None:
    monkeypatch.setenv("YAP_TEST_RENDER_STDERR", stderr)
    comp = _gsap_composition(tmp_path / "failed-render", frames=3)
    out = tmp_path / "out.mov"
    with pytest.raises(ProcessingError) as caught:
        render_layer(comp, out, frames=3, variables={}, install=retry_renderer)
    assert caught.value.code == "hyperframes_exit_1"
    assert caught.value.full_stderr == stderr
    assert (tmp_path / "render-count").read_text() == "1"
    assert not out.exists()
    assert not list(comp.glob(".layer-*.mov"))
    assert not list(comp.glob(".yap_tmp_*"))
    assert not (comp / "variables.json").exists()
    assert not (comp / GSAP_SCRIPT).exists()


def test_render_with_retry_records_first_matching_line() -> None:
    detail = "net::ERR_EMPTY_RESPONSE at http://localhost:57344/" + "x" * 200
    error = ProcessingError("hyperframes", 1, f"banner\n   {detail}\n{_LOCAL_SERVER_STDERR}")
    calls = []

    def attempt() -> None:
        calls.append(None)
        if len(calls) == 1:
            raise error

    assert hyperframes.render_with_retry(attempt) == {
        "reason": "renderer_local_server", "detail": detail[:200],
    }
    assert len(calls) == 2
    assert hyperframes.render_with_retry(attempt) is None
    assert len(calls) == 3


@pytest.mark.parametrize("error", [
    ProcessingError("hyperframes", 1, "encoder failed"),
    TimeoutError("renderer timed out"),
    RendererBlockedError(),
    YapError("Renderer variable validation failed.", code="renderer_variables"),
])
@pytest.mark.parametrize("after_retry", [False, True])
def test_render_with_retry_propagates_failure_unchanged(
    error: Exception, after_retry: bool,
) -> None:
    calls = []

    def attempt() -> None:
        calls.append(None)
        if after_retry and len(calls) == 1:
            raise ProcessingError("hyperframes", 1, _LOCAL_SERVER_STDERR)
        raise error

    with pytest.raises(type(error)) as caught:
        hyperframes.render_with_retry(attempt)
    assert caught.value is error
    assert len(calls) == (2 if after_retry else 1)


def test_strict_variable_failure_has_own_code(
    tmp_path: Path,
    fake_renderer: Callable[[str], RendererInstall],
) -> None:
    comp = _composition(tmp_path / "variables")
    with pytest.raises(YapError) as caught:
        render_layer(
            comp,
            tmp_path / "out.mov",
            frames=30,
            variables={"undeclared": 1},
            install=fake_renderer(),
        )
    assert caught.value.code == "renderer_variables"
    assert caught.value.suggested_action
    assert not (tmp_path / "out.mov").exists()


def test_box_layers_overlay_at_offsets(
    tmp_path: Path,
    fake_renderer: Callable[[str], RendererInstall],
) -> None:
    renderer = fake_renderer()
    band = render_layer(
        _composition(tmp_path / "band", frames=1, width=1080, height=120),
        tmp_path / "band.mov",
        frames=1,
        variables={"swatches": [[64, 64, 64, 255]]},
        install=renderer,
    )
    card = render_layer(
        _composition(tmp_path / "card", frames=1, width=160, height=80),
        tmp_path / "card.mov",
        frames=1,
        variables={"swatches": [[32, 80, 224, 255]]},
        install=renderer,
    )
    assert (ffmpeg.probe(band.path).width, ffmpeg.probe(band.path).height) == (1080, 120)
    assert (ffmpeg.probe(card.path).width, ffmpeg.probe(card.path).height) == (160, 80)
    pixels = ffmpeg.run(
        [
            "ffmpeg",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "color=black:s=1080x1920:r=30",
            "-i",
            str(band.path),
            "-i",
            str(card.path),
            "-filter_complex",
            "[0:v]format=yuva444p10le,setparams=colorspace=bt709[base];"
            "[base][1:v]overlay=0:1700:format=auto[band];"
            "[band][2:v]overlay=120:300:format=auto,"
            "scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgba[out]",
            "-map",
            "[out]",
            "-an",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        timeout_s=60,
        binary=True,
    ).stdout
    for x, y, expected in (
        (10, 1701, (64, 64, 64, 255)),
        (1079, 1819, (64, 64, 64, 255)),
        (10, 1699, (0, 0, 0, 255)),
        (10, 1820, (0, 0, 0, 255)),
        (121, 301, (32, 80, 224, 255)),
        (279, 379, (32, 80, 224, 255)),
        (119, 301, (0, 0, 0, 255)),
        (280, 301, (0, 0, 0, 255)),
    ):
        _assert_pixel(_pixel(pixels, 1080, x, y), expected)


@pytest.fixture
def install_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, list[str], Mapping[str, object]]]:
    """Stub downloads while retaining staging, metadata, and cache publication."""
    monkeypatch.setenv("YAP_STUDIO_CACHE", str(tmp_path))
    return stub_install_tools(tmp_path, monkeypatch)


def stub_install_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, list[str], Mapping[str, object]]]:
    """Stub the Node version, npm and browser calls without changing the cache root."""
    calls: list[tuple[str, list[str], Mapping[str, object]]] = []
    monkeypatch.setattr(tools, "find_tool", lambda _: "npm")
    monkeypatch.setattr(tools, "node_version", lambda: "v22.23.1")

    def run(
        name: str,
        args: Sequence[str],
        *,
        timeout_s: float,
        env: Mapping[str, str] | None = None,
        input_bytes: bytes | None = None,
        executable: str | os.PathLike[str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((name, list(args), {"timeout": timeout_s, "env": env, "cwd": cwd}))
        assert env is not None and input_bytes is None
        assert all(env[key] == value for key, value in RENDERER_ENV.items())
        if name == "hyperframes":
            assert env.get("CI") == "1"
            assert Path(env["HYPERFRAMES_HISTORY_ROOT"]).is_dir()
        assert timeout_s > 0
        stdout = ""
        if name == "npm":
            assert args == ["ci", "--no-audit", "--no-fund", "--ignore-scripts"]
            assert cwd is not None
            staging = Path(cwd)
            assert not install_prefix().exists() or installed() is None
            for filename in ("package.json", "package-lock.json"):
                assert (staging / filename).read_bytes() == (
                    hyperframes.LOCK_DIR / filename
                ).read_bytes()
            program = staging / "node_modules" / ".bin" / "hyperframes"
            program.parent.mkdir(parents=True)
            program.write_text("#!/bin/sh\nexit 0\n")
            program.chmod(0o700)
        else:
            assert name == "hyperframes" and args == ["browser", "ensure"]
            assert executable is not None and Path(executable).is_file()
            browser = tmp_path / "chrome-headless-shell"
            browser.touch()
            stdout = f"   Path:    {browser}\nReady to render.\n"
        return subprocess.CompletedProcess([name, *args], 0, stdout, "")

    monkeypatch.setattr(tools, "run_tool", run)
    return calls


def test_renderer_lock_pins_one_exact_version() -> None:
    package = json.loads((hyperframes.LOCK_DIR / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((hyperframes.LOCK_DIR / "package-lock.json").read_text(encoding="utf-8"))
    dependencies = {"hyperframes": HYPERFRAMES_VERSION, "gsap": GSAP_VERSION}
    assert package["dependencies"] == dependencies
    assert lock["packages"][""]["dependencies"] == dependencies
    for declared in (package["dependencies"], lock["packages"][""]["dependencies"]):
        assert all(re.fullmatch(r"\d+\.\d+\.\d+", version) for version in declared.values())
    renderer = lock["packages"]["node_modules/hyperframes"]
    assert renderer["version"] == HYPERFRAMES_VERSION
    assert renderer["resolved"] == (
        f"https://registry.npmjs.org/hyperframes/-/hyperframes-{HYPERFRAMES_VERSION}.tgz"
    )
    assert renderer["integrity"].startswith("sha512-")
    gsap = lock["packages"]["node_modules/gsap"]
    assert gsap["version"] == GSAP_VERSION
    assert gsap["resolved"] == f"https://registry.npmjs.org/gsap/-/gsap-{GSAP_VERSION}.tgz"
    assert gsap["integrity"].startswith("sha512-")
    assert [
        name for name in lock["packages"]
        if name == "node_modules/gsap" or name.endswith("/node_modules/gsap")
    ] == ["node_modules/gsap"]


def test_tracked_files_do_not_vendor_gsap() -> None:
    root = Path(__file__).parents[1]
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    inside = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if inside.returncode or inside.stdout.strip() != "true":
        pytest.skip("directory is not a git work tree")
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30,
        check=True,
    ).stdout.split(b"\0")
    needle = "Green" + "Sock"
    offenders: list[str] = []
    for raw_path in tracked:
        if not raw_path:
            continue
        relative = raw_path.decode(errors="surrogateescape")
        filename = Path(relative).name.lower()
        file = root / relative
        if not file.is_file():
            continue  # Tracked paths may have a pending working-tree deletion.
        if filename.startswith("gsap") and filename.endswith(".js"):
            offenders.append(relative)
        data = file.read_bytes()
        if b"\0" not in data and needle in data.decode(errors="replace"):
            offenders.append(relative)
    assert not offenders, "vendored GSAP found:\n" + "\n".join(offenders)


def test_install_is_atomic_cached_and_pinned(
    install_tools: list[tuple[str, list[str], Mapping[str, object]]],
) -> None:
    notices: list[str] = []
    result = ensure_installed(notice=notices.append, timeout_s=17)
    assert result == installed()
    assert result.version == HYPERFRAMES_VERSION
    assert result.browser_path is not None and Path(result.browser_path).is_file()
    assert result.node_version == "v22.23.1"
    assert result.prefix == install_prefix() and result.executable.is_file()
    assert [call[0] for call in install_tools] == ["npm", "hyperframes"]
    assert len(notices) == 1 and "Chrome" in notices[0]
    assert ensure_installed(notice=notices.append) == result
    assert len(install_tools) == 2 and len(notices) == 1
    assert not list(result.prefix.parent.glob(".hyperframes-*"))


@pytest.mark.parametrize("recorded_hash", [None, "0" * 64])
def test_install_lock_freshness_reinstalls_stale_cache(
    install_tools: list[tuple[str, list[str], Mapping[str, object]]],
    recorded_hash: str | None,
) -> None:
    result = ensure_installed()
    metadata_file = result.prefix / "install.json"
    metadata = json.loads(metadata_file.read_text())
    if recorded_hash is None:
        metadata.pop("lock_sha256")
    else:
        metadata["lock_sha256"] = recorded_hash
    metadata_file.write_text(json.dumps(metadata))
    marker = result.prefix / "stale"
    marker.write_text("old lock")
    install_tools.clear()
    assert installed() is None
    assert ensure_installed() == result == installed()
    assert [name for name, _, _ in install_tools] == ["npm", "hyperframes"]
    expected = hashlib.sha256(
        (hyperframes.LOCK_DIR / "package.json").read_bytes()
        + b"\0"
        + (hyperframes.LOCK_DIR / "package-lock.json").read_bytes()
    ).hexdigest()
    assert json.loads(metadata_file.read_text())["lock_sha256"] == expected
    assert not marker.exists()
    assert not list(result.prefix.parent.glob(".hyperframes-*"))


@pytest.mark.parametrize("filename", ["package.json", "package-lock.json"])
def test_changed_renderer_lock_reinstalls_cache(
    tmp_path: Path,
    install_tools: list[tuple[str, list[str], Mapping[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    result = ensure_installed()
    previous_hash = json.loads((result.prefix / "install.json").read_text())["lock_sha256"]
    lock_dir = tmp_path / "lock"
    lock_dir.mkdir()
    for name in ("package.json", "package-lock.json"):
        shutil.copyfile(hyperframes.LOCK_DIR / name, lock_dir / name)
    with (lock_dir / filename).open("ab") as stream:
        stream.write(b"\n")
    monkeypatch.setattr(hyperframes, "LOCK_DIR", lock_dir)
    install_tools.clear()
    assert installed() is None
    assert ensure_installed() == result == installed()
    assert sum(name == "npm" for name, _, _ in install_tools) == 1
    current_hash = json.loads((result.prefix / "install.json").read_text())["lock_sha256"]
    assert current_hash != previous_hash


def test_stale_lock_cache_survives_interrupted_reinstall(
    install_tools: list[tuple[str, list[str], Mapping[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = ensure_installed()
    metadata_file = result.prefix / "install.json"
    metadata = json.loads(metadata_file.read_text())
    metadata.pop("lock_sha256")
    metadata_file.write_text(json.dumps(metadata))
    marker = result.prefix / "preserved"
    marker.write_text("previous cache")
    original = tools.run_tool

    def interrupted(
        name: str, args: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if name == "hyperframes":
            raise ProcessingError("browser ensure", 1, "download failed")
        return original(name, args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(tools, "run_tool", interrupted)
    with pytest.raises(ProcessingError):
        ensure_installed()
    assert result.executable.is_file() and marker.read_text() == "previous cache"
    assert installed() is None
    assert not list(result.prefix.parent.glob(".hyperframes-*"))


def test_concurrent_installs_run_npm_once(
    install_tools: list[tuple[str, list[str], Mapping[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = threading.Barrier(2)
    local = threading.local()
    lookup = hyperframes.installed

    def synchronized_lookup() -> RendererInstall | None:
        result = lookup()
        if not getattr(local, "started", False):
            local.started = True
            barrier.wait(timeout=60)
        return result

    monkeypatch.setattr(hyperframes, "installed", synchronized_lookup)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(ensure_installed) for _ in range(2)]
        results = [future.result(timeout=120) for future in futures]
    assert results[0] == results[1] == installed()
    assert sum(name == "npm" for name, _, _ in install_tools) == 1
    assert sum(name == "hyperframes" for name, _, _ in install_tools) == 1


@pytest.mark.skipif(os.name == "nt", reason="Unix flock timeout regression")
def test_install_lock_timeout_closes_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CACHE", str(tmp_path))
    acquired = threading.Event()
    release = threading.Event()
    descriptors: list[int] = []
    main_thread = threading.get_ident()
    open_descriptor = os.open

    def tracked_open(*args: object, **kwargs: object) -> int:
        descriptor = open_descriptor(*args, **kwargs)  # type: ignore[arg-type]
        if threading.get_ident() == main_thread and Path(args[0]).name == ".install.lock":
            descriptors.append(descriptor)
        return descriptor

    def hold_lock() -> None:
        with hyperframes._install_lock(timeout_s=1):
            acquired.set()
            release.wait(timeout=2)

    monkeypatch.setattr(os, "open", tracked_open)
    with ThreadPoolExecutor(max_workers=1) as pool:
        holder = pool.submit(hold_lock)
        try:
            assert acquired.wait(timeout=5)
            started = time.monotonic()
            with pytest.raises(DependencyError) as caught:
                with hyperframes._install_lock(timeout_s=0.05):
                    pytest.fail("contended lock was acquired after its deadline")
            assert caught.value.code == "renderer_install_locked"
            assert caught.value.suggested_action
            assert time.monotonic() - started < 1
            assert len(descriptors) == 1
            with pytest.raises(OSError):
                os.fstat(descriptors[0])
        finally:
            release.set()
            holder.result(timeout=5)
    with hyperframes._install_lock(timeout_s=0.05):
        pass


def test_installed_rejects_missing_browser(
    install_tools: list[tuple[str, list[str], Mapping[str, object]]],
) -> None:
    result = ensure_installed()
    assert result.browser_path is not None
    Path(result.browser_path).unlink()
    assert installed() is None


def test_missing_browser_repairs_without_npm_install(
    install_tools: list[tuple[str, list[str], Mapping[str, object]]],
) -> None:
    result = ensure_installed()
    assert result.browser_path is not None
    Path(result.browser_path).unlink()
    marker = result.prefix / "preserved"
    marker.write_text("keep the active prefix")
    install_tools.clear()
    assert ensure_installed() == result == installed()
    assert [name for name, _, _ in install_tools] == ["hyperframes"]
    assert marker.read_text() == "keep the active prefix"


@pytest.mark.parametrize("prefix_kind", ["directory", "symlink"])
def test_missing_executable_rebuilds_valid_install(
    install_tools: list[tuple[str, list[str], Mapping[str, object]]],
    prefix_kind: str,
) -> None:
    result = ensure_installed()
    marker = result.prefix / "stale"
    marker.write_text("broken installation")
    result.executable.unlink()
    if prefix_kind == "symlink":
        previous = result.prefix.with_name("previous-install")
        result.prefix.rename(previous)
        result.prefix.symlink_to(previous, target_is_directory=True)
    install_tools.clear()
    assert installed() is None
    assert ensure_installed() == result == installed()
    assert [name for name, _, _ in install_tools] == ["npm", "hyperframes"]
    assert result.executable.is_file()
    assert not marker.exists() and not result.prefix.is_symlink()
    if prefix_kind == "symlink":
        assert (previous / "stale").read_text() == "broken installation"
        assert (previous / "install.json").is_file()


@pytest.mark.parametrize(
    "interrupt", [ProcessingError("browser ensure", 1, "download failed"), KeyboardInterrupt()]
)
def test_interrupted_install_cleans_staging(
    install_tools: list[tuple[str, list[str], Mapping[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
    interrupt: BaseException,
) -> None:
    original = tools.run_tool

    def interrupted(
        name: str, args: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if name == "hyperframes":
            raise interrupt
        return original(name, args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(tools, "run_tool", interrupted)
    with pytest.raises(type(interrupt)):
        ensure_installed()
    prefix = install_prefix()
    assert not prefix.exists() and installed() is None
    assert not list(prefix.parent.glob(".hyperframes-*"))
    assert [call[0] for call in install_tools] == ["npm"]


def test_stale_install_is_replaced(
    install_tools: list[tuple[str, list[str], Mapping[str, object]]],
) -> None:
    prefix = install_prefix()
    prefix.mkdir(parents=True)
    (prefix / "install.json").write_text("invalid")
    (prefix / "stale").write_text("incomplete")
    assert installed() is None
    assert ensure_installed() == installed()
    assert not (prefix / "stale").exists()


@pytest.mark.parametrize("stdout", ["Ready to render.\n", "   Path:   \nReady to render.\n"])
def test_missing_browser_path_does_not_publish_install(
    install_tools: list[tuple[str, list[str], Mapping[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
) -> None:
    original = tools.run_tool

    def run(name: str, args: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if name == "hyperframes":
            return subprocess.CompletedProcess([name, *args], 0, stdout, "")
        return original(name, args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(tools, "run_tool", run)
    with pytest.raises(DependencyError) as caught:
        ensure_installed()
    assert caught.value.code == "renderer_browser_missing"
    prefix = install_prefix()
    assert not prefix.exists() and not list(prefix.parent.glob(".hyperframes-*"))


@pytest.mark.parametrize(("node_version", "npm"), [
    (None, True), ("v20.19.0", True), ("invalid", True), ("v22.23.1", False),
])
def test_missing_old_or_invalid_node_prevents_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    node_version: str | None, npm: bool,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CACHE", str(tmp_path))
    calls: list[str] = []

    def run(name: str, args: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(name)
        return subprocess.CompletedProcess([name, *args], 0, "", "")

    monkeypatch.setattr(tools, "node_version", lambda: node_version)
    monkeypatch.setattr(tools, "find_tool", lambda _: "npm" if npm else None)
    monkeypatch.setattr(tools, "run_tool", run)
    with pytest.raises(DependencyError) as caught:
        ensure_installed()
    assert caught.value.code == "node_not_found"
    assert calls == [] and not install_prefix().exists()


@pytest.mark.parametrize("global_dir", ["npm", "lib"])
def test_self_update_cache_paths_are_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    global_dir: str,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CACHE", str(tmp_path))
    destination = tmp_path / global_dir / "node_modules" / "hyperframes"
    destination.mkdir(parents=True)
    prefix = tmp_path / "renderer" / f"hyperframes-{HYPERFRAMES_VERSION}"
    prefix.parent.mkdir()
    prefix.symlink_to(destination, target_is_directory=True)
    with pytest.raises(DependencyError) as caught:
        install_prefix()
    assert caught.value.code == "renderer_cache_path" and caught.value.suggested_action


def test_conversion_rejects_input_frame_mismatch_before_ffmpeg(
    tmp_path: Path,
    fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comp = _composition(tmp_path / "source", frames=1)
    src = render_layer(
        comp, tmp_path / "source.mov", frames=1, variables={}, install=fake_renderer()
    ).path

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("frame mismatch reached ffmpeg")

    # probe needs ffmpeg.run too; preserve its commands while refusing a conversion.
    original = ffmpeg.run

    def only_probe(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[object]:
        if cmd[0] == "ffmpeg":
            unexpected()
        return original(cmd, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ffmpeg, "run", only_probe)
    out = tmp_path / "mismatch.mov"
    with pytest.raises(ProcessingError) as caught:
        alpha.convert_to_bt709(src, out, frames=2)
    assert caught.value.code == "layer_frame_mismatch" and caught.value.suggested_action
    assert not out.exists()


def test_conversion_rejects_unsupported_source_matrix_before_ffmpeg(
    tmp_path: Path,
    fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comp = _composition(tmp_path / "source", frames=1)
    source = render_layer(
        comp, tmp_path / "source.mov", frames=1, variables={}, install=fake_renderer()
    ).path
    tagged = tmp_path / "bt2020.mov"
    ffmpeg.run(
        [
            "ffmpeg", "-nostdin", "-y", "-i", str(source), "-map", "0:v:0", "-an",
            "-c:v", "copy", "-colorspace", "bt2020nc", str(tagged),
        ],
        timeout_s=60,
    )
    assert ffmpeg.probe(tagged).color_space == "bt2020nc"
    original = ffmpeg.run

    def only_probe(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[object]:
        if cmd[0] == "ffmpeg":
            pytest.fail("unsupported matrix reached ffmpeg")
        return original(cmd, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ffmpeg, "run", only_probe)
    out = tmp_path / "unsupported.mov"
    with pytest.raises(ProcessingError) as caught:
        alpha.convert_to_bt709(tagged, out, frames=1)
    assert caught.value.code == "layer_colour_tags" and caught.value.suggested_action
    assert not out.exists() and not list(tmp_path.glob(".yap_tmp_*"))


@pytest.mark.parametrize("fault", ["frames", "tags", "alpha"])
def test_conversion_rejects_bad_output_before_publication(
    tmp_path: Path,
    fake_renderer: Callable[[str], RendererInstall],
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    comp = _composition(tmp_path / "source", frames=1)
    src = render_layer(
        comp, tmp_path / "source.mov", frames=1, variables={}, install=fake_renderer()
    ).path
    original = ffmpeg.probe

    def probe(path: Path) -> ffmpeg.MediaInfo:
        info = original(path)
        if path != src:
            if fault == "frames":
                return replace(info, video_frames=2)
            if fault == "tags":
                return replace(info, color_space=None)
            return replace(info, pix_fmt="yuv444p12le")
        return info

    monkeypatch.setattr(ffmpeg, "probe", probe)
    out = tmp_path / "bad.mov"
    with pytest.raises(ProcessingError) as caught:
        alpha.convert_to_bt709(src, out, frames=1)
    assert caught.value.code == (
        "layer_frame_mismatch" if fault == "frames" else "layer_colour_tags"
    )
    assert not out.exists() and not list(tmp_path.glob(".yap_tmp_*"))


def test_existing_clip_cannot_be_overwritten(
    tmp_path: Path,
    fake_renderer: Callable[[str], RendererInstall],
) -> None:
    comp = _composition(tmp_path / "source", frames=1)
    renderer = fake_renderer()
    src = render_layer(comp, tmp_path / "source.mov", frames=1, variables={}, install=renderer).path
    original = src.read_bytes()
    with pytest.raises(OutputPathError):
        alpha.convert_to_bt709(src, src, frames=1)
    assert src.read_bytes() == original


def test_offline_install_wraps_only_the_executable(tmp_path: Path) -> None:
    install = RendererInstall(
        HYPERFRAMES_VERSION, tmp_path, tmp_path / "real renderer", "chrome", "v22.0.0"
    )
    directory = tmp_path / "offline 'quoted'"
    offline = offline_renderer.offline_install(install, directory)
    assert replace(offline, executable=install.executable) == install
    assert offline.executable.is_file() and os.access(offline.executable, os.X_OK)
    wrapper = offline.executable.read_text().splitlines()
    assert wrapper[0] == "#!/bin/sh"
    profile = directory / "offline.sb"
    assert shlex.split(wrapper[1]) == [
        "exec", "/usr/bin/sandbox-exec", "-f", str(profile), str(install.executable), "$@"
    ]
    assert profile.read_text().splitlines() == [
        "(version 1)",
        "(allow default)",
        "(deny network-outbound)",
        '(allow network-outbound (remote ip "localhost:*"))',
        "(allow network-outbound (remote unix-socket))",
    ]


@pytest.mark.render
def test_real_render_keeps_history_out_of_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    install = ensure_installed()
    assert install.browser_path is not None
    monkeypatch.setenv("HYPERFRAMES_BROWSER_PATH", install.browser_path)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    inherited = process.spawn(
        [sys.executable, "-c", "import os; print(os.environ['HOME'])"],
        timeout_s=10, tool="hyperframes", env=RENDERER_ENV,
    )
    assert inherited.stdout.strip() == str(home)
    comp = _composition(tmp_path / "real-history", frames=1, width=32, height=32)
    render_layer(comp, tmp_path / "real-history.mov", frames=1, variables={},
                 install=install, checks=False)
    assert not (home / ".cache" / "hyperframes" / "history").exists()
    assert not (comp / ".hyperframes").exists()


@pytest.mark.render
def test_real_pinned_renderer_checks_text_layout(tmp_path: Path) -> None:
    """Opt-in checks measure actual slot text for overflowing and clean boxes."""
    install = ensure_installed()
    for name, text, width in (
        ("overflow", "Wide text " * 20, 40),
        ("clean", "Fits", 240),
    ):
        comp = _composition(tmp_path / name)
        (comp / "index.html").write_text(
            '<!doctype html><html data-composition-variables=\'[{"id":"text",'
            '"type":"string","label":"Text","default":""}]\'><head><style>'
            "html,body{margin:0;background:transparent}"
            f"#text{{width:{width}px;height:48px;font-size:32px;white-space:nowrap;color:white}}"
            '</style></head><body><div data-composition-id="text-check" data-no-timeline '
            f'data-duration="{layer_duration_s(30)}" data-width="320" data-height="180" '
            'style="width:320px;height:180px"><div id="text" data-var-text="text"></div>'
            '</div><script>const declarations = JSON.parse(document.documentElement'
            '.getAttribute("data-composition-variables"));'
            'document.getElementById("text").textContent = declarations[0].default;'
            '</script></body></html>',
            encoding="utf-8",
        )
        result = layer_checks.check_layer(
            comp, {"text": text}, install=install, fps=30, duration_s=layer_duration_s(30)
        )
        assert result["passes"]["layout"]["status"] != "skipped", result
        if name == "overflow":
            assert result["passes"]["layout"]["findings"], result
        else:
            assert result["passes"]["layout"]["status"] == "passed", result
            assert all(item["errors"] == 0 for item in result["passes"].values()), result


@pytest.mark.render
def test_real_pinned_renderer_script_error_refuses_output(tmp_path: Path) -> None:
    comp = _gsap_composition(tmp_path / "throwing", frames=3)
    html = comp / "index.html"
    source = html.read_text()
    injected = source.replace(
        "<script>window.__timelines",
        "<script>throw new Error('injected script failure'); window.__timelines", 1,
    )
    assert injected != source
    html.write_text(injected)
    out = tmp_path / "throwing.mov"
    with pytest.raises(YapError, match="injected script failure"):
        render_layer(comp, out, frames=3, variables={}, checks=False)
    assert not out.exists()
    assert not list(comp.glob("*.mov"))
    assert not (comp / "variables.json").exists()
    assert not (comp / GSAP_SCRIPT).exists()


@pytest.mark.render
def test_real_pinned_renderer_one_second(tmp_path: Path) -> None:
    """Opt-in real installation and Chrome render, keeping the configured cache."""
    install = ensure_installed()
    comp = _composition(tmp_path / "real", frames=30)
    result = render_layer(comp, tmp_path / "real.mov", frames=30, variables={}, install=install)
    info = ffmpeg.probe(result.path)
    assert info.video_frames == 30 and info.fps == 30
    assert (info.width, info.height) == (320, 180)
    assert info.color_space == info.color_transfer == info.color_primaries == "bt709"
    assert info.pix_fmt == "yuva444p12le"
    _assert_pixel(_pixel(_decode(result.path), 320, 160, 90), (200, 60, 40, 128))


def _assert_gsap_fade(path: Path) -> None:
    info = ffmpeg.probe(path)
    assert info.video_frames == 29 and info.fps == 30
    assert (info.width, info.height) == (320, 180)
    data = ffmpeg.run(
        [
            "ffmpeg",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            "alphaextract",
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        timeout_s=120,
        binary=True,
    ).stdout
    frame_size = 320 * 180
    assert len(data) == 29 * frame_size
    assert max(data[:frame_size]) == 0
    assert min(data[-frame_size:]) >= 250


@pytest.mark.render
def test_real_pinned_renderer_gsap_fade(tmp_path: Path) -> None:
    install = ensure_installed()
    comp = _gsap_composition(tmp_path / "real-gsap")
    result = render_layer(
        comp, tmp_path / "real-gsap.mov", frames=29, variables={}, install=install
    )
    _assert_gsap_fade(result.path)
    assert not (comp / GSAP_SCRIPT).exists()
    assert not (comp / "variables.json").exists()


@pytest.mark.render
def test_real_pinned_renderer_gsap_fade_offline(tmp_path: Path) -> None:
    if not offline_renderer.offline_supported():
        pytest.skip("offline renderer needs macOS sandbox-exec")
    sandbox = tmp_path / "offline"
    offline_renderer.assert_network_blocked(sandbox)
    install = ensure_installed()
    comp = _gsap_composition(tmp_path / "offline-gsap")
    online = render_layer(comp, tmp_path / "online.mov", frames=29, variables={}, install=install)
    _assert_gsap_fade(online.path)
    offline = offline_renderer.offline_install(install, sandbox)
    result = render_layer(comp, tmp_path / "offline.mov", frames=29, variables={}, install=offline)
    _assert_gsap_fade(result.path)
    assert not (comp / GSAP_SCRIPT).exists()
    assert not (comp / "variables.json").exists()
