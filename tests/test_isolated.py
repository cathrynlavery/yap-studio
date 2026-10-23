"""Independent speech runs survive child failures without changing primary checks."""

from __future__ import annotations

import importlib
import json
import signal
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_relisten import ToneBackend, _joins
from test_relisten import tone_media as tone_media

from yap_studio import process
from yap_studio.errors import ProcessingError
from yap_studio.report import exit_code
from yap_studio.transcript import (
    IsolatedBackend,
    ScriptedBackend,
    SecondPassUnmeasured,
    Transcript,
    Word,
    child,
    isolated,
)


@pytest.fixture
def speech_paths(tmp_path):
    paths = [tmp_path / "first.wav", tmp_path / "second.wav"]
    for path in paths:
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16_000)
            stream.writeframes(b"\x00\x00" * 16_000)
    return paths


@pytest.fixture
def child_requests(monkeypatch):
    """Observe request directories without changing process execution."""
    requests = []
    spawn = process.spawn

    def record(command, **kwargs):
        if kwargs.get("tool") == "second_engine":
            request = Path(command[-1])
            assert request.parent.stat().st_mode & 0o077 == 0
            requests.append(request)
        return spawn(command, **kwargs)

    monkeypatch.setattr(process, "spawn", record)
    yield requests
    assert requests
    assert all(not request.parent.exists() for request in requests)


def _fake_child(tmp_path: Path, mode: str, *, transcripts=None) -> list[str]:
    """Build a plain Python child that needs no speech packages."""
    script = tmp_path / f"child-{mode}.py"
    default = Transcript([Word("hello", 1.0, 1.2)], backend="faster-whisper", model="fixture")
    payload = {str(path): transcript.to_dict() for path, transcript in (transcripts or {}).items()}
    script.write_text(
        "import json, os, resource, sys, time\n"
        "from pathlib import Path\n"
        "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
        "request = json.loads(Path(sys.argv[1]).read_text())\n"
        f"mode = {mode!r}\n"
        "if mode == 'abort_before': os.abort()\n"
        "if mode == 'no_result': sys.exit(1)\n"
        "if mode == 'timeout': time.sleep(10)\n"
        "result = Path(request['result'])\n"
        "if mode == 'truncated':\n"
        "    result.write_text('{')\n"
        "    sys.stderr.write('x' * 800 + 'stderr tail')\n"
        "    sys.exit(1)\n"
        f"prepared = json.loads({json.dumps(payload)!r})\n"
        f"default = {default.to_dict()!r}\n"
        "transcripts = prepared or {path: default for path in request['paths']}\n"
        "if mode == 'missing': transcripts.pop(request['paths'][-1])\n"
        "data = {'schema': 'yap-studio/second-pass@1', 'engine': request['engine'],\n"
        "        'model': request['model'], 'version': '0', 'complete': True,\n"
        "        'transcripts': transcripts}\n"
        "if mode == 'wrong_schema': data['schema'] = 'other'\n"
        "if mode == 'wrong_engine': data['engine'] = 'other'\n"
        "if mode == 'partial': data['complete'] = False\n"
        "if mode == 'invalid_transcript': transcripts[request['paths'][0]] = {}\n"
        "if mode == 'non_object': data = []\n"
        "result.write_text(json.dumps(data))\n"
        "if mode == 'abort_after': os.abort()\n"
        "if mode == 'exit_after': sys.exit(1)\n",
        encoding="utf-8",
    )
    return [sys.executable, str(script)]


def _inner():
    def forbidden(_):
        pytest.fail("the parent loaded the second speech engine")

    return ScriptedBackend(forbidden, name="faster-whisper", model="fixture")


def test_child_writes_complete_round_trip(tmp_path, speech_paths, monkeypatch, capsys):
    calls = []
    expected = {
        path: Transcript(
            [Word(path.stem, 0.1, 0.5, 0.9)], language="en", backend="scripted",
            model="fixture", duration_s=1,
        )
        for path in speech_paths
    }

    class BatchBackend(ScriptedBackend):
        def transcribe_many(self, paths, *, language=None):
            print("speech progress")
            calls.append((list(paths), language))
            return super().transcribe_many(paths, language=language)

    backend = BatchBackend(expected.__getitem__, model="fixture")
    resolutions = []

    def resolve(engine, model):
        resolutions.append((engine, model))
        return backend

    monkeypatch.setattr(child, "resolve_backend", resolve)
    request, result = tmp_path / "request.json", tmp_path / "result.json"
    request.write_text(json.dumps({
        "engine": backend.name, "model": backend.model, "paths": list(map(str, speech_paths)),
        "result": str(result), "language": "en",
    }))
    assert child.main([str(request)]) == 0
    data = json.loads(result.read_text())
    assert data == {
        "schema": isolated.SECOND_PASS_SCHEMA, "engine": backend.name,
        "model": backend.model, "version": "0", "complete": True,
        "transcripts": {str(path): transcript.to_dict() for path, transcript in expected.items()},
    }
    assert {
        Path(path): Transcript.from_dict(transcript)
        for path, transcript in data["transcripts"].items()
    } == expected
    assert calls == [(speech_paths, "en")]
    assert resolutions == [(backend.name, backend.model)]
    assert sorted(path.name for path in tmp_path.glob("*.json")) == ["request.json", "result.json"]
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == "speech progress\n"


@pytest.mark.parametrize("failure", ["usage", "invalid_request", "engine"])
def test_child_reports_errors_without_a_result(tmp_path, monkeypatch, capsys, failure):
    result, request = tmp_path / "result.json", tmp_path / "request.json"
    if failure == "usage":
        assert child.main([]) == 2
    else:
        request.write_text("{" if failure == "invalid_request" else json.dumps({
            "engine": "scripted", "model": "fixture", "paths": [], "result": str(result),
        }))

        def fail(*_):
            raise ProcessingError("speech", 1, "could not transcribe")

        monkeypatch.setattr(child, "resolve_backend", fail)
        assert child.main([str(request)]) == 1
    assert not result.exists()
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err


@pytest.mark.parametrize("mode,returncode", [
    ("complete", 0), ("abort_after", -signal.SIGABRT), ("exit_after", 1),
])
def test_parent_uses_complete_result_after_any_exit(
    tmp_path, speech_paths, child_requests, mode, returncode,
):
    backend = IsolatedBackend(_inner(), command=_fake_child(tmp_path, mode))
    result = backend.transcribe_many(speech_paths)
    assert list(result) == speech_paths
    assert all(transcript.text == "hello" for transcript in result.values())
    assert backend.name == backend.inner.name and backend.model == backend.inner.model
    assert backend.version() == "0" and backend.download_notice() is None
    assert backend.last_run == {
        "returncode": returncode, "signal": "SIGABRT" if returncode < 0 else None,
        "used_result": True, "reason": None, "seconds": backend.last_run["seconds"],
    }
    assert backend.last_run["seconds"] > 0


@pytest.mark.parametrize("mode,detail,returncode", [
    ("abort_before", "stopped before writing its result (signal 6, SIGABRT)", -signal.SIGABRT),
    ("no_result", "stopped before writing its result (exit status 1)", 1),
    ("truncated", "wrote an incomplete result (exit status 1)", 1),
    ("missing", "wrote an incomplete result (exit status 0)", 0),
    ("wrong_schema", "wrote an incomplete result (exit status 0)", 0),
    ("wrong_engine", "wrote an incomplete result (exit status 0)", 0),
    ("partial", "wrote an incomplete result (exit status 0)", 0),
    ("invalid_transcript", "wrote an incomplete result (exit status 0)", 0),
    ("non_object", "wrote an incomplete result (exit status 0)", 0),
    ("timeout", "timed out after 0.1 s", None),
])
def test_parent_maps_unusable_results_to_unmeasured(
    tmp_path, speech_paths, child_requests, monkeypatch, mode, detail, returncode,
):
    backend = IsolatedBackend(_inner(), command=_fake_child(tmp_path, mode))
    if mode == "timeout":
        monkeypatch.setattr(isolated, "_timeout_seconds", lambda _: 0.1)
    with pytest.raises(SecondPassUnmeasured) as caught:
        backend.transcribe_many(speech_paths)
    error = caught.value
    assert error.reason == f"unmeasured: the second speech engine faster-whisper {detail}"
    assert error.message == error.reason and error.code == "second_engine_unmeasured"
    if mode == "truncated":
        assert error.full_stderr == ("x" * 800 + "stderr tail")[-500:]
    assert backend.last_run["returncode"] == returncode
    assert backend.last_run["signal"] == ("SIGABRT" if mode == "abort_before" else None)
    assert backend.last_run["used_result"] is False
    assert backend.last_run["reason"] == error.reason
    assert backend.last_run["seconds"] > 0


def test_parent_failed_start_is_unmeasured(tmp_path, speech_paths, child_requests):
    backend = IsolatedBackend(_inner(), command=[str(tmp_path / "missing-executable")])
    with pytest.raises(SecondPassUnmeasured, match="faster-whisper failed to start"):
        backend.transcribe_many(speech_paths)
    assert backend.last_run["returncode"] is None
    assert backend.last_run["used_result"] is False


def test_default_child_command_runs_the_module(speech_paths, child_requests):
    backend = IsolatedBackend(ScriptedBackend(lambda _: Transcript([])))
    assert backend.command == [sys.executable, "-m", "yap_studio.transcript.child"]
    with pytest.raises(SecondPassUnmeasured, match="stopped before writing its result") as caught:
        backend.transcribe_many(speech_paths)
    assert caught.value.returncode == 1
    assert "Bad value for asr" in caught.value.full_stderr


def test_timeout_scales_with_wav_headers_and_probe(speech_paths, monkeypatch):
    assert isolated._timeout_seconds(speech_paths) == 363
    monkeypatch.setattr(
        isolated.ffmpeg, "probe",
        lambda *_args, **_kwargs: SimpleNamespace(duration_s=None, audio_duration_s=400),
    )
    assert isolated._timeout_seconds([Path("media.mp4")]) == 930


class FailingSecond(ScriptedBackend):
    def __init__(self, error):
        super().__init__(lambda _: Transcript([]), name="faster-whisper")
        self.error = error

    def transcribe_many(self, paths, *, language=None):
        raise self.error


def _without_second(value):
    """Strip only second-model evidence, including copies attached to findings."""
    if isinstance(value, dict):
        return {
            key: _without_second(item) for key, item in value.items()
            if key not in {"second", "second_status", "second_reason", "second_comparison"}
        }
    if isinstance(value, list):
        return [_without_second(item) for item in value]
    return value


@pytest.mark.parametrize("case", ["clean", "short", "doubled"])
@pytest.mark.parametrize("isolated_error", [True, False])
def test_second_failure_preserves_cuts_and_findings(tone_media, tmp_path, case, isolated_error):
    reason = "unmeasured: the second speech engine faster-whisper stopped before writing its result"
    error = (
        SecondPassUnmeasured(reason) if isolated_error else ProcessingError("speech", 1, "failed")
    )
    second = FailingSecond(error)
    failed = _joins(tone_media, case, tmp_path, ToneBackend(), second)
    baseline = _joins(tone_media, case, tmp_path, ToneBackend())
    expected = reason if isolated_error else (
        f"unmeasured: the second speech engine faster-whisper failed: {error.message}"
    )
    assert failed.cuts
    for cut in failed.cuts:
        assert cut.relisten.second_status == "skipped"
        assert cut.relisten.second_reason == expected
    assert _without_second(failed.to_dict()) == _without_second(baseline.to_dict())
    assert _without_second([finding.to_dict() for finding in failed.findings]) == _without_second([
        finding.to_dict() for finding in baseline.findings
    ])


def test_primary_transcription_error_still_propagates(tone_media, tmp_path):
    error = ProcessingError("speech", 1, "primary failed")
    with pytest.raises(ProcessingError) as caught:
        _joins(tone_media, "clean", tmp_path, FailingSecond(error))
    assert caught.value is error


def test_verify_second_failure_preserves_status_exit_and_other_checks(tone_media, monkeypatch):
    module = importlib.import_module("yap_studio.verify")
    monkeypatch.setattr(module, "resolve_backend", lambda *_: ToneBackend(name="parakeet-mlx"))
    monkeypatch.setattr(module, "resolve_second_backend", lambda _: None)
    row = tone_media.cases["short"]
    baseline = module.verify(str(row.render), edl=str(row.edl), detail="full")
    reason = (
        "unmeasured: the second speech engine faster-whisper stopped before writing its result "
        "(signal 6, SIGABRT)"
    )
    monkeypatch.setattr(
        module, "resolve_second_backend", lambda _: FailingSecond(SecondPassUnmeasured(reason)),
    )
    failed = module.verify(str(row.render), edl=str(row.edl), detail="full")
    assert failed["checks"]["join.second_model"] == {
        "ran": False, "status": "skipped", "reason": reason,
    }
    assert failed["status"] == baseline["status"]
    for strict in (False, True):
        assert exit_code(failed, strict) == exit_code(baseline, strict)
    assert {
        key: value for key, value in failed["checks"].items() if key != "join.second_model"
    } == {
        key: value for key, value in baseline["checks"].items() if key != "join.second_model"
    }
    assert _without_second(failed["findings"]) == _without_second(baseline["findings"])


def test_verify_uses_complete_result_after_child_abort(
    tone_media, tmp_path, monkeypatch, child_requests,
):
    module = importlib.import_module("yap_studio.verify")
    first = ToneBackend(name="parakeet-mlx")
    row = tone_media.cases["clean"]
    baseline = _joins(tone_media, "clean", tmp_path / "baseline", first)
    assert baseline.cuts[0].relisten.second_status == "skipped"
    prepared = {path: first.hear(path) for path in first.batches[0]}
    # Verification creates new trial paths; map matching basenames in the fake child.
    command = _fake_child(tmp_path, "abort_after", transcripts=prepared)
    script = Path(command[-1])
    script.write_text(script.read_text().replace(
        "transcripts = prepared or {path: default for path in request['paths']}",
        "by_name = {Path(path).name: value for path, value in prepared.items()}\n"
        "transcripts = {path: by_name[Path(path).name] for path in request['paths']}",
    ))
    second = IsolatedBackend(_inner(), command=command)
    monkeypatch.setattr(module, "resolve_backend", lambda *_: ToneBackend(name="parakeet-mlx"))
    monkeypatch.setattr(module, "resolve_second_backend", lambda _: second)
    result = module.verify(str(row.render), edl=str(row.edl), detail="full")
    check = result["checks"]["join.second_model"]
    assert check["ran"] is True and check["status"] == "pass" and check["reason"] is None
    assert check["note"] == (
        "faster-whisper exited with signal 6 (SIGABRT) after writing a complete result; "
        "the result was used"
    )
    assert result["receipt"]["speech_second"]["run"] == second.last_run
    assert second.last_run["returncode"] == -signal.SIGABRT
    assert second.last_run["signal"] == "SIGABRT" and second.last_run["used_result"] is True
    assert result["status"] != "error" and exit_code(result, False) != 134
