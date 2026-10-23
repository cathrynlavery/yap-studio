"""Speech child protocol, shutdown crash isolation, and model-free verification."""

from __future__ import annotations

import importlib
import json
import logging
import sys
import tempfile
import types
import wave
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from yap_studio.errors import ProcessingError
from yap_studio.report import exit_code
from yap_studio.transcript import (
    ParakeetOnnxBackend,
    ScriptedBackend,
    Transcript,
    Word,
    engine_child,
)

_FAKE_CHILD = '''
import json
import os
import pathlib
import sys
import time
import wave

try:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except ImportError:
    pass

mode = sys.argv[1]
request = json.loads(pathlib.Path(sys.argv[-1]).read_text())
print("engine diagnostic", file=sys.stderr, flush=True)
if mode == "abort_before":
    os.abort()
if mode == "exit_before":
    sys.exit(1)
if mode == "sleep":
    time.sleep(10)
result = pathlib.Path(request["result"])
if mode == "truncated":
    result.write_text('{"schema":')
    sys.exit(0)
transcripts = {}
for index, value in enumerate(request["paths"]):
    path = pathlib.Path(value)
    with wave.open(str(path), "rb") as reader:
        duration = reader.getnframes() / reader.getframerate()
    if path.stem.endswith("left"):
        words = [("bright", 1.2, 1.4), ("planet", 1.5, 1.9)]
    elif path.stem.endswith("right"):
        words = [("quiet", 2.1, 2.4), ("garden", 2.5, 2.8)]
    else:
        words = [("bright", 1.2, 1.4), ("planet", 1.5, 1.9),
                 ("quiet", 2.1, 2.4), ("garden", 2.5, 2.8)]
    transcripts[value] = {
        "schema": "yap-studio/transcript@1",
        "words": [{"text": text, "start": start, "end": end, "conf": None}
                  for text, start, end in words],
        "language": request["language"],
        "backend": request["engine"],
        "model": request["model"],
        "duration_s": duration,
    }
payload = {
    "schema": "yap-studio/engine-child@1",
    "engine": request["engine"],
    "model": request["model"],
    "version": "0",
    "complete": True,
    "transcripts": transcripts,
}
if mode == "missing_path":
    del transcripts[request["paths"][-1]]
if mode == "wrong_schema":
    payload["schema"] = "unknown"
if mode == "wrong_engine":
    payload["engine"] = "unknown"
if mode == "incomplete":
    payload["complete"] = False
if mode == "invalid_transcript":
    transcripts[request["paths"][0]]["schema"] = "unknown"
if mode == "not_object":
    payload = []
temporary = result.with_suffix(".tmp")
temporary.write_text(json.dumps(payload))
os.replace(temporary, result)
if mode == "abort_after":
    os.abort()
if mode == "exit_after":
    sys.exit(1)
if mode == "sleep_after":
    time.sleep(10)
'''


def _wav(path: Path, seconds: float = 3.0) -> Path:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(b"\0\0" * round(seconds * 16_000))
    return path


@pytest.fixture
def fake_child(tmp_path: Path) -> Callable[[str], list[str]]:
    script = tmp_path / "child.py"
    script.write_text(_FAKE_CHILD, encoding="utf-8")
    return lambda mode: [sys.executable, str(script), mode]


@pytest.fixture
def child_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[list[Path]]:
    """Track all private directories, including failed starts and timeouts."""
    directories: list[Path] = []
    original = tempfile.TemporaryDirectory

    def temporary_directory(*args: Any, **kwargs: Any) -> Any:
        directory = original(*args, **kwargs, dir=tmp_path)
        directories.append(Path(directory.name))
        return directory

    monkeypatch.setattr(engine_child.tempfile, "TemporaryDirectory", temporary_directory)
    yield directories
    assert directories
    assert all(not directory.exists() for directory in directories)


def _request(tmp_path: Path, engine: str, paths: list[Path], **options: Any) -> Path:
    request = tmp_path / "request.json"
    request.write_text(json.dumps({
        "engine": engine,
        "model": "fixture",
        "options": options,
        "language": "en",
        "paths": [str(path) for path in paths],
        "result": str(tmp_path / "result.json"),
    }), encoding="utf-8")
    return request


def test_child_main_batches_scripted_engine_and_redirects_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = [_wav(tmp_path / "first.wav"), _wav(tmp_path / "second.wav")]
    transcript = Transcript([Word("hello", 0.2, 0.4)], backend="scripted", model="fixture")
    calls: list[tuple[list[Path], str | None]] = []

    class RecordingBackend(ScriptedBackend):
        def transcribe_many(
            self, paths: Sequence[Path], *, language: str | None = None
        ) -> dict[Path, Transcript]:
            calls.append((list(paths), language))
            print("engine stdout")
            return super().transcribe_many(paths, language=language)

    backend = RecordingBackend(lambda _: transcript)

    def resolve(engine: str, model: str) -> ScriptedBackend:
        assert (engine, model) == ("scripted", "fixture")
        return backend

    monkeypatch.setattr(engine_child, "resolve_backend", resolve)
    request = _request(tmp_path, "scripted", paths)

    assert engine_child.main([str(request)]) == 0
    assert calls == [(paths, "en")]
    result = json.loads((tmp_path / "result.json").read_text())
    assert result == {
        "schema": engine_child.SCHEMA,
        "engine": "scripted",
        "model": "fixture",
        "version": "0",
        "complete": True,
        "transcripts": {str(path): transcript.to_dict() for path in paths},
    }
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "engine stdout\n"
    assert sorted(path.name for path in tmp_path.glob("*.json")) == ["request.json", "result.json"]


@pytest.mark.parametrize("quantization", ["int8", None])
def test_child_main_onnx_honours_quantization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quantization: str | None
) -> None:
    paths = [_wav(tmp_path / "input.wav")]
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeEngine:
        def with_timestamps(self) -> FakeEngine:
            return self

        def recognize(self, wavs: list[str]) -> list[SimpleNamespace]:
            return [SimpleNamespace(tokens=[" hello"], timestamps=[0.2]) for _ in wavs]

    def load_model(model: str, **options: Any) -> FakeEngine:
        calls.append((model, options))
        return FakeEngine()

    module = types.ModuleType("onnx_asr")
    module.load_model = load_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnx_asr", module)
    request = _request(tmp_path, "parakeet-onnx", paths, quantization=quantization)

    assert engine_child.main([str(request)]) == 0
    assert calls == [("fixture", {
        "quantization": quantization, "providers": ["CPUExecutionProvider"],
    })]
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["complete"] is True
    assert Transcript.from_dict(result["transcripts"][str(paths[0])]).text == "hello"


def test_child_main_failure_writes_no_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(_path: Path) -> Transcript:
        raise RuntimeError("recognition failed\nextra detail")

    monkeypatch.setattr(engine_child, "resolve_backend", lambda *_: ScriptedBackend(fail))
    request = _request(tmp_path, "scripted", [tmp_path / "input.wav"])
    assert engine_child.main([str(request)]) == 1
    assert not (tmp_path / "result.json").exists()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "speech engine failed: recognition failed\n"


@pytest.mark.parametrize(
    ("mode", "returncode", "signal", "used", "reason"),
    [
        ("complete", 0, None, True, None),
        ("abort_after", -6, "SIGABRT", True, None),
        ("exit_after", 1, None, True, None),
        ("abort_before", -6, "SIGABRT", False,
         "stopped before writing its result (signal 6, SIGABRT)"),
        ("exit_before", 1, None, False, "stopped before writing its result (exit status 1)"),
        ("truncated", 0, None, False, "wrote an incomplete result"),
        ("missing_path", 0, None, False, "wrote an incomplete result"),
        ("wrong_schema", 0, None, False, "wrote an incomplete result"),
        ("wrong_engine", 0, None, False, "wrote an incomplete result"),
        ("incomplete", 0, None, False, "wrote an incomplete result"),
        ("invalid_transcript", 0, None, False, "wrote an incomplete result"),
        ("not_object", 0, None, False, "wrote an incomplete result"),
    ],
)
def test_parent_uses_only_complete_results(
    tmp_path: Path,
    fake_child: Callable[[str], list[str]],
    child_directories: list[Path],
    mode: str,
    returncode: int,
    signal: str | None,
    used: bool,
    reason: str | None,
) -> None:
    paths = [_wav(tmp_path / "first.wav"), _wav(tmp_path / "second.wav")]
    run = engine_child.run_in_child(
        "parakeet-onnx", "fixture", paths, language="en", command=fake_child(mode)
    )
    assert run.returncode == returncode
    assert run.signal == signal
    assert run.used_result is used
    assert run.reason == reason
    assert run.seconds > 0
    assert run.stderr_tail == "engine diagnostic\n"
    assert "transcripts" not in run.to_dict()
    assert json.loads(json.dumps(run.to_dict())) == run.to_dict()
    if used:
        assert run.transcripts is not None
        assert list(run.transcripts) == paths
        assert all(transcript.language == "en" for transcript in run.transcripts.values())
    else:
        assert run.transcripts is None
    assert len(child_directories) == 1


@pytest.mark.parametrize("mode", ["sleep", "sleep_after"])
def test_parent_timeout(
    tmp_path: Path, fake_child: Callable[[str], list[str]], child_directories: list[Path], mode: str
) -> None:
    path = _wav(tmp_path / "input.wav")
    run = engine_child.run_in_child(
        "parakeet-onnx", "fixture", [path], command=fake_child(mode), timeout_s=0.5
    )
    assert run.returncode is None
    assert run.signal is None
    assert run.used_result is (mode == "sleep_after")
    assert run.reason == (None if run.used_result else "timed out after 0.5 s")


def test_parent_failed_start(tmp_path: Path, child_directories: list[Path]) -> None:
    path = _wav(tmp_path / "input.wav")
    run = engine_child.run_in_child(
        "parakeet-onnx", "fixture", [path], command=[str(tmp_path / "missing-command")]
    )
    assert run.transcripts is None
    assert run.returncode is None
    assert run.signal is None
    assert run.used_result is False
    assert run.reason == "failed to start"
    assert run.stderr_tail


def test_default_child_module_runs_without_a_model(child_directories: list[Path]) -> None:
    run = engine_child.run_in_child(
        "parakeet-onnx", "fixture", [], options={"quantization": "int8"}
    )
    assert run.transcripts == {}
    assert run.returncode == 0
    assert run.used_result is True
    assert run.reason is None
    assert run.stderr_tail == ""


def test_parent_missing_input_is_a_failed_start(
    tmp_path: Path, child_directories: list[Path]
) -> None:
    run = engine_child.run_in_child("parakeet-onnx", "fixture", [tmp_path / "missing.wav"])
    assert run.transcripts is None
    assert run.used_result is False
    assert run.returncode is None
    assert run.reason == "failed to start"
    assert "no file at this path" in run.stderr_tail


@pytest.mark.parametrize("seconds", [3.0, 400.0])
def test_parent_default_command_timeout_and_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, child_directories: list[Path], seconds: float
) -> None:
    path = _wav(tmp_path / "input.wav", seconds=seconds)

    def spawn(command: list[str], *, timeout_s: float, tool: str) -> SimpleNamespace:
        assert command[:3] == [sys.executable, "-m", "yap_studio.transcript.engine_child"]
        assert tool == "speech_engine"
        assert timeout_s == max(330, 150 + 1.5 * seconds)
        request = json.loads(Path(command[-1]).read_text())
        assert request == {
            "engine": "parakeet-onnx", "model": "fixture", "options": {"quantization": "int8"},
            "language": "en", "paths": [str(path)],
            "result": str(child_directories[0] / "result.json"),
        }
        return SimpleNamespace(returncode=1, stderr="failed")

    monkeypatch.setattr(engine_child.process, "spawn", spawn)
    run = engine_child.run_in_child(
        "parakeet-onnx", "fixture", [path], options={"quantization": "int8"}, language="en"
    )
    assert run.returncode == 1


def test_parent_timeout_probes_non_wav(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "input.mp4"
    path.write_bytes(b"media")
    calls: list[Path] = []

    def probe(value: Path, *, count_frames: bool) -> SimpleNamespace:
        calls.append(value)
        assert count_frames is False
        return SimpleNamespace(duration_s=None, audio_duration_s=400.0)

    monkeypatch.setattr(engine_child.ffmpeg, "probe", probe)
    assert engine_child._timeout([path]) == 750.0
    assert calls == [path]


def test_onnx_backend_empty_request_skips_child(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = ParakeetOnnxBackend()
    backend.child_command = [sys.executable, "-c", "import sys; sys.exit(3)"]
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    original = engine_child.run_in_child

    def run_in_child(*args: Any, **kwargs: Any) -> engine_child.ChildRun:
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(engine_child, "run_in_child", run_in_child)

    assert backend.transcribe_many([]) == {}
    assert backend.last_run is None
    assert calls == []


@pytest.mark.parametrize("mode", ["complete", "abort_after", "abort_before"])
def test_onnx_backend_isolates_native_modules_and_maps_failures(
    tmp_path: Path,
    fake_child: Callable[[str], list[str]],
    child_directories: list[Path],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    mode: str,
) -> None:
    native_names = ("onnx_asr", "onnxruntime")
    native_modules = {name: sys.modules[name] for name in native_names if name in sys.modules}
    path = _wav(tmp_path / "input.wav")
    backend = ParakeetOnnxBackend(model="fixture")
    backend.child_command = fake_child(mode)
    with caplog.at_level(logging.DEBUG, logger="yap_studio"):
        if mode == "abort_before":
            with pytest.raises(ProcessingError, match="signal 6, SIGABRT") as caught:
                backend.transcribe_many([path])
            assert caught.value.code == "speech_engine_failed"
            assert "engine diagnostic" in caught.value.full_stderr
        else:
            transcripts = backend.transcribe_many([path])
            assert list(transcripts)[0] is path
            assert transcripts[path].text == "bright planet quiet garden"
    assert backend.last_run is not None
    assert backend.last_run.used_result is (mode != "abort_before")
    after = {name: sys.modules[name] for name in native_names if name in sys.modules}
    assert after == native_modules
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    records = [record for record in caplog.records if record.name == "yap_studio"]
    assert len(records) == (1 if mode == "abort_after" else 0)
    assert all(record.levelno == logging.DEBUG for record in records)


def test_verify_report_and_status_survive_child_abort(
    synth_media: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_child: Callable[[str], list[str]],
    child_directories: list[Path],
) -> None:
    module = importlib.import_module("yap_studio.verify")
    edl = tmp_path / "edit.json"
    edl.write_text(json.dumps({
        "source": str(synth_media.path("calib")),
        "segments": [{"start": start, "end": end} for start, end in synth_media.spans],
    }))
    backend = ParakeetOnnxBackend(model="fixture")
    monkeypatch.setattr(module, "resolve_backend", lambda *_: backend)
    # This tests the primary child's result, independently of installed speech extras.
    monkeypatch.setattr(module, "resolve_second_backend", lambda _: None)
    reports = []
    for mode in ("complete", "abort_after"):
        backend.child_command = fake_child(mode)
        result = module.verify(
            str(synth_media.path("clean")), edl=str(edl), asr="parakeet-onnx", detail="full"
        )
        assert result["checks"]["join.words"]["ran"] is True
        assert result["cuts"]
        assert backend.last_run is not None and backend.last_run.used_result
        assert backend.last_run.returncode == (0 if mode == "complete" else -6)
        reports.append(result)
    assert exit_code(reports[0], strict=False) == exit_code(reports[1], strict=False)
    # Receipt times and transient transcript artifacts vary with each verification.
    for result in reports:
        for key in ("started_at", "ended_at", "duration_s"):
            result["receipt"].pop(key)
        for output in result["receipt"]["outputs"]:
            output["path"] = Path(output["path"]).name
            output.pop("sha256")
        for cut in result["cuts"]:
            words = cut["detail"]["words"]
            words["artifacts"] = [Path(value).name for value in words["artifacts"]]
    assert reports[0] == reports[1]
