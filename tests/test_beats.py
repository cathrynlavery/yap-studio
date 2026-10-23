"""Offline beat detection contracts using a fake renderer and synthetic audio."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from yap_studio import ffmpeg, tools
from yap_studio.errors import DependencyError, ProcessingError
from yap_studio.render import beats, hyperframes
from yap_studio.render.beats import BeatsError, detect_beats
from yap_studio.render.hyperframes import HYPERFRAMES_VERSION, RendererInstall

pytestmark = pytest.mark.usefixtures("_no_real_node")

DOCUMENT = {"version": 1, "audio": "music.wav", "beats": [
    {"time": 0.001, "strength": 1}, {"time": 0.501, "strength": 0.714},
]}
SUMMARY = {"ok": True, "file": "beats/music.wav.json", "count": 2, "bpm": 120}
FAKE = '''import json, os, sys
from pathlib import Path

args = sys.argv[1:]
assert args[:2] == ["beats", "--json"]
assert sys.stdin.buffer.read() == b""
assert all(os.environ[key] == "1" for key in (
    "HYPERFRAMES_NO_TELEMETRY", "HYPERFRAMES_NO_UPDATE_CHECK",
    "HYPERFRAMES_NO_AUTO_INSTALL", "HYPERFRAMES_SKIP_SKILLS"))
assert os.environ.get("CI") == "1"
root = Path(args[2])
Path(audit).write_text(json.dumps({"root": str(root),
    "html": (root / "index.html").read_text(),
    "audio": (root / "music.wav").read_bytes().hex()}))
if config["write_file"]:
    (root / "beats").mkdir()
    (root / "beats/music.wav.json").write_text(json.dumps(config["document"]))
print(config["stdout"])
sys.exit(config["exit_code"])
'''


@pytest.fixture
def fake_renderer(tmp_path: Path) -> Callable[..., RendererInstall]:
    """Write an executable that emits the requested offline beat contract."""
    def build(*, document: Any = DOCUMENT, summary: Any = SUMMARY,
              stdout: str | None = None, exit_code: int = 0,
              write_file: bool = True) -> RendererInstall:
        config = {"document": document, "stdout": json.dumps(summary) if stdout is None else stdout,
                  "exit_code": exit_code, "write_file": write_file}
        executable = tmp_path / "hyperframes"
        executable.write_text(
            "#!/usr/bin/env python3\nimport json\n"
            f"config = json.loads({json.dumps(config)!r})\n"
            f"audit = {str(tmp_path / 'invoked.json')!r}\n" + FAKE,
            encoding="utf-8",
        )
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        return RendererInstall(HYPERFRAMES_VERSION, tmp_path, executable, None, "v22.0.0")

    return build


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    """Keep the creator's file and a neighbour separate from renderer outputs."""
    folder = tmp_path / "creator"
    folder.mkdir()
    (folder / "notes.txt").write_text("Keep this file.", encoding="utf-8")
    audio = folder / "original.wav"
    audio.write_bytes(b"synthetic audio read only by the fake renderer")
    return audio


def test_detects_canned_beats_without_touching_source_folder(
    tmp_path: Path, audio_file: Path, fake_renderer: Callable[..., RendererInstall],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = {file.name: file.read_bytes() for file in audio_file.parent.iterdir()}
    install = fake_renderer()
    monkeypatch.setattr(hyperframes, "ensure_installed", lambda: install)
    assert detect_beats(audio_file, duration_s=1) == {
        "bpm": 120.0, "beats": [{"time_s": 0.001, "strength": 1.0},
                                {"time_s": 0.501, "strength": 0.714}],
    }
    invoked = json.loads((tmp_path / "invoked.json").read_text())
    assert not Path(invoked["root"]).exists()
    assert invoked["audio"] == audio_file.read_bytes().hex()
    assert 'data-composition-id="root"' in invoked["html"]
    assert 'data-duration="1.0"' in invoked["html"]
    assert 'data-timeline-role="music" src="music.wav" data-start="0"' in invoked["html"]
    assert {file.name: file.read_bytes() for file in audio_file.parent.iterdir()} == before


@pytest.mark.parametrize("field,value", [
    ("version", 2), ("version", True), ("audio", "other.wav"), ("beats", {}),
])
def test_rejects_invalid_file_contract(
    audio_file: Path, fake_renderer: Callable[..., RendererInstall], field: str, value: Any,
) -> None:
    document = {**DOCUMENT, field: value}
    with pytest.raises(BeatsError):
        detect_beats(audio_file, duration_s=1, install=fake_renderer(document=document))


@pytest.mark.parametrize("items", [
    [{"time": -0.001, "strength": 1}], [{"time": 1.001, "strength": 1}],
    [{"time": 0.5, "strength": 1}, {"time": 0.5, "strength": 0.5}],
    [{"time": 0.5, "strength": 1}, {"time": 0.1, "strength": 0.5}],
    [{"time": float("nan"), "strength": 1}],
    [{"time": float("inf"), "strength": 1}],
    [{"time": True, "strength": 1}], [{"time": "0.5", "strength": 1}],
    [{"time": 0.5, "strength": -0.1}], [{"time": 0.5, "strength": 1.1}],
    [{"time": 0.5, "strength": float("nan")}],
    [{"time": 0.5, "strength": float("inf")}],
    [{"time": 0.5, "strength": True}], [{"time": 0.5}], [None],
])
def test_rejects_invalid_beats(
    audio_file: Path, fake_renderer: Callable[..., RendererInstall], items: list[Any],
) -> None:
    # JSON round-tripping in the executable also covers non-finite JSON numbers.
    install = fake_renderer(document={**DOCUMENT, "beats": items})
    with pytest.raises(BeatsError) as caught:
        detect_beats(audio_file, duration_s=1, install=install)
    assert caught.value.code == "beats_failed" and caught.value.suggested_action


@pytest.mark.parametrize("summary", [
    [], {**SUMMARY, "ok": False}, {**SUMMARY, "file": "../elsewhere.json"},
    {**SUMMARY, "count": 3}, {**SUMMARY, "count": True},
    {**SUMMARY, "bpm": float("inf")}, {**SUMMARY, "bpm": "120"},
    {**SUMMARY, "bpm": True}, {**SUMMARY, "bpm": -120},
])
def test_rejects_invalid_summary(
    audio_file: Path, fake_renderer: Callable[..., RendererInstall], summary: Any,
) -> None:
    with pytest.raises(BeatsError):
        detect_beats(audio_file, duration_s=1, install=fake_renderer(summary=summary))


@pytest.mark.parametrize("options", [
    {"exit_code": 1}, {"stdout": "not JSON"}, {"write_file": False},
    {"document": None}, {"document": {**DOCUMENT, "beats": [{"time": -1, "strength": 1}]}},
])
def test_failure_removes_temp_project_and_preserves_source_folder(
    tmp_path: Path, audio_file: Path, fake_renderer: Callable[..., RendererInstall],
    options: dict[str, Any],
) -> None:
    before = {file.name: file.read_bytes() for file in audio_file.parent.iterdir()}
    with pytest.raises(BeatsError):
        detect_beats(audio_file, duration_s=1, install=fake_renderer(**options))
    invoked = json.loads((tmp_path / "invoked.json").read_text())
    assert not Path(invoked["root"]).exists()
    assert {file.name: file.read_bytes() for file in audio_file.parent.iterdir()} == before


def test_empty_detection_has_optional_bpm(
    audio_file: Path, fake_renderer: Callable[..., RendererInstall],
) -> None:
    install = fake_renderer(document={**DOCUMENT, "beats": []},
                            summary={**SUMMARY, "count": 0, "bpm": None})
    assert detect_beats(audio_file, duration_s=1, install=install) == {"bpm": None, "beats": []}


@pytest.mark.parametrize("duration", [0, -1, float("inf"), float("nan"), True])
def test_rejects_invalid_audio_duration(audio_file: Path, duration: float) -> None:
    with pytest.raises(BeatsError):
        detect_beats(audio_file, duration_s=duration)


def test_process_timeout_removes_temp_project(
    audio_file: Path, fake_renderer: Callable[..., RendererInstall],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = []

    def timeout(command: list[str], **kwargs: Any) -> None:
        roots.append(Path(command[-1]))
        assert roots[-1].is_dir() and kwargs["timeout_s"] == 0.1
        raise ProcessingError("hyperframes beats", -1, "synthetic timeout")

    monkeypatch.setattr(beats.process, "spawn", timeout)
    with pytest.raises(BeatsError, match="synthetic timeout"):
        detect_beats(audio_file, duration_s=1, install=fake_renderer(), timeout_s=0.1)
    assert roots and all(not root.exists() for root in roots)


def test_install_failure_is_a_beats_error(
    audio_file: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> None:
        raise DependencyError.missing_renderer()

    monkeypatch.setattr(hyperframes, "ensure_installed", unavailable)
    with pytest.raises(BeatsError) as caught:
        detect_beats(audio_file, duration_s=1)
    assert caught.value.code == "beats_failed"


def _detect_synthetic_120_bpm_clicks(tmp_path: Path, install: RendererInstall) -> None:
    audio = tmp_path / "clicks.wav"
    ffmpeg.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         r"aevalsrc=if(lt(mod(t\,0.5)\,0.01)\,0.8*sin(2*PI*1000*t)\,0):s=48000:d=12",
         "-c:a", "pcm_s16le", str(audio)], timeout_s=60,
    )
    result = detect_beats(audio, duration_s=12, install=install)
    assert len(result["beats"]) == pytest.approx(24, abs=3)
    assert result["bpm"] == pytest.approx(120, abs=5)


@pytest.mark.render
def test_pinned_renderer_detects_synthetic_120_bpm_clicks(tmp_path: Path) -> None:
    install = hyperframes.installed()
    if install is None:
        pytest.skip("the pinned renderer must already be installed")
    _detect_synthetic_120_bpm_clicks(tmp_path, install)


@pytest.mark.render
def test_pinned_renderer_detects_synthetic_120_bpm_clicks_with_broken_first_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    install = hyperframes.installed()
    if install is None:
        pytest.skip("the pinned renderer must already be installed")
    broken = tmp_path / "broken-bin"
    broken.mkdir()
    node = broken / "node"
    node.write_text('#!/bin/sh\necho "dyld: Library not loaded: missing.dylib" >&2\nexit 134\n')
    node.chmod(0o755)
    npm = broken / "npm"
    npm.write_text("#!/bin/sh\nexit 0\n")
    npm.chmod(0o755)
    monkeypatch.setenv("PATH", str(broken) + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr(tools, "NODE_INSTALL_PLACES", ())
    tools._find_node.cache_clear()
    try:
        _detect_synthetic_120_bpm_clicks(tmp_path, install)
    finally:
        tools._find_node.cache_clear()
