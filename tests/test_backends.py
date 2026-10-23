"""Speech backend resolver, adapter, and real-engine tests."""

from __future__ import annotations

import math
import platform
import stat
import sys
import types
import wave
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import synth

from yap_studio.errors import DependencyError, ProcessingError, ValidationError
from yap_studio.transcript import Transcript, Word, from_parakeet_tokens, norm_words
from yap_studio.transcript.backends import (
    ONNX_BATCH_SECONDS,
    ONNX_MAX_CHUNK_S,
    OVERLAP_S,
    FasterWhisperBackend,
    ParakeetMlxBackend,
    ParakeetOnnxBackend,
    ScriptedBackend,
    _onnx_splice,
    _OnnxWord,
    available_engines,
    resolve_backend,
)

SENTENCES = [
    "Bright planets circle slowly over quiet hills on calm winter evenings",
    "Seven green boxes wait beside the open door near our garden",
    "Morning light fills the empty kitchen before breakfast begins each day",
]


def _wav(path: Path, *, seconds: float = 0.2) -> Path:
    count = round(16_000 * seconds)
    samples = bytearray()
    for index in range(count):
        sample = round(2_000 * math.sin(2 * math.pi * 220 * index / 16_000))
        samples.extend(int(sample).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(samples)
    return path


def _availability(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mlx: bool = False,
    onnx: bool = False,
    whisper: bool = False,
) -> None:
    monkeypatch.setattr(ParakeetMlxBackend, "available", staticmethod(lambda: mlx))
    monkeypatch.setattr(ParakeetOnnxBackend, "available", staticmethod(lambda: onnx))
    monkeypatch.setattr(FasterWhisperBackend, "available", staticmethod(lambda: whisper))


@pytest.mark.parametrize(
    ("mlx", "onnx", "whisper", "expected"),
    [
        (True, True, True, "parakeet-mlx"),
        (False, True, True, "parakeet-onnx"),
        (False, False, True, "faster-whisper"),
        (False, False, False, None),
    ],
)
def test_resolver_auto_order(
    monkeypatch: pytest.MonkeyPatch,
    mlx: bool,
    onnx: bool,
    whisper: bool,
    expected: str | None,
) -> None:
    _availability(monkeypatch, mlx=mlx, onnx=onnx, whisper=whisper)
    backend = resolve_backend()
    assert (backend.name if backend is not None else None) == expected


def test_resolver_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _availability(monkeypatch, mlx=True, onnx=True, whisper=True)
    assert resolve_backend("none") is None


def test_resolver_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    _availability(monkeypatch)
    with pytest.raises(ValidationError, match="asr"):
        resolve_backend("unknown")


@pytest.mark.parametrize(
    ("name", "extra"),
    [("parakeet-onnx", "asr"), ("faster-whisper", "whisper")],
)
def test_resolver_explicit_missing_engine_has_install_hint(
    monkeypatch: pytest.MonkeyPatch, name: str, extra: str
) -> None:
    _availability(monkeypatch)
    with pytest.raises(DependencyError) as caught:
        resolve_backend(name)
    assert caught.value.code == "speech_engine_not_found"
    assert f"yap-studio[{extra}]" in (caught.value.suggested_action or "")


@pytest.mark.parametrize(("suffix", "quantization"), [(":full", None), (":int8", "int8")])
def test_resolver_parses_onnx_model_precision(
    monkeypatch: pytest.MonkeyPatch, suffix: str, quantization: str | None
) -> None:
    _availability(monkeypatch, onnx=True)
    backend = resolve_backend("parakeet-onnx", f"custom/model{suffix}")
    assert isinstance(backend, ParakeetOnnxBackend)
    assert backend.model == "custom/model"
    assert backend.quantization == quantization


def test_available_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    _availability(monkeypatch, mlx=True, whisper=True)
    assert available_engines() == {
        "parakeet-mlx": True,
        "parakeet-onnx": False,
        "faster-whisper": True,
    }


@pytest.mark.parametrize("quantization", ["int8", None])
@pytest.mark.parametrize("cached", [True, False])
def test_onnx_download_notice_checks_only_required_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quantization: str | None, cached: bool
) -> None:
    snapshot = _cached_onnx_snapshot(tmp_path, quantization)
    calls: list[dict[str, object]] = []
    module = types.ModuleType("huggingface_hub")

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        if not cached:
            raise OSError("snapshot is incomplete")
        return str(snapshot)

    module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    notice = ParakeetOnnxBackend(quantization=quantization).download_notice()

    suffix = ".int8" if quantization == "int8" else ""
    assert calls == [{
        "repo_id": "istupakov/parakeet-tdt-0.6b-v3-onnx",
        "local_files_only": True,
        "allow_patterns": [
            "config.json",
            "config.yaml",
            f"encoder-model{suffix}.onnx",
            f"decoder_joint-model{suffix}.onnx",
            "vocab.txt",
            f"encoder-model{suffix}.onnx?data",
            f"decoder_joint-model{suffix}.onnx?data",
        ],
    }]
    assert notice == (None if cached else "First run downloads the speech model (about 670 MB).")


def _cached_onnx_snapshot(tmp_path: Path, quantization: str | None) -> Path:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    suffix = f".{quantization}" if quantization else ""
    for name in (
        "config.json", "vocab.txt",
        f"encoder-model{suffix}.onnx", f"decoder_joint-model{suffix}.onnx",
    ):
        (snapshot / name).write_bytes(b"cached model file")
    return snapshot


@pytest.mark.parametrize(
    ("cache_state", "needs_notice"),
    [
        ("cached", False),
        ("no-ref", False),
        ("empty", True),
        ("config-only", True),
        ("model-only", True),
        ("zero-byte", True),
        ("missing", True),
    ],
)
def test_mlx_download_notice_without_huggingface_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_state: str, needs_notice: bool
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    if cache_state != "missing":
        repo = tmp_path / "hub" / "models--mlx-community--parakeet-tdt-0.6b-v3"
        snapshot = repo / "snapshots" / "revision"
        snapshot.mkdir(parents=True)
        if cache_state != "no-ref":
            (repo / "refs").mkdir()
            (repo / "refs" / "main").write_text("revision\n")
        if cache_state not in ("empty", "model-only"):
            (snapshot / "config.json").write_text("{}")
        if cache_state not in ("empty", "config-only"):
            (snapshot / "model.safetensors").write_bytes(
                b"" if cache_state == "zero-byte" else b"model"
            )

    notice = ParakeetMlxBackend().download_notice()

    assert (notice is not None) == needs_notice


def test_mlx_download_notice_with_huggingface_hub_needs_both_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = lambda **kwargs: str(snapshot)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    backend = ParakeetMlxBackend()
    assert backend.download_notice() is not None
    (snapshot / "model.safetensors").write_bytes(b"model")
    assert backend.download_notice() is None


def test_mlx_prefetch_checks_downloaded_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = lambda **kwargs: str(snapshot)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    backend = ParakeetMlxBackend()
    with pytest.raises(OSError, match="incomplete"):
        backend.prefetch()
    (snapshot / "model.safetensors").write_bytes(b"model")
    backend.prefetch()


def test_onnx_download_notice_without_huggingface_hub_needs_every_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    repo = tmp_path / "hub" / "models--istupakov--parakeet-tdt-0.6b-v3-onnx"
    snapshot = repo / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("revision\n")
    for name in ParakeetOnnxBackend()._model_files()[2]:
        (snapshot / name).write_bytes(b"model")

    backend = ParakeetOnnxBackend()
    assert backend.download_notice() is None
    (snapshot / "vocab.txt").unlink()
    assert backend.download_notice() is not None


def test_mlx_prefetch_without_huggingface_hub_uses_cli_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    snapshot = (tmp_path / "hub" / "models--mlx-community--parakeet-tdt-0.6b-v3"
                / "snapshots" / "revision")
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"model")
    python = tmp_path / "python3"
    python.write_text("")
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    executable = tmp_path / "parakeet-mlx"
    executable.write_text(f"#!{python}\n")
    monkeypatch.setattr(ParakeetMlxBackend, "executable", staticmethod(lambda: executable))
    calls: list[tuple[list[str], float, str]] = []

    def spawn(command: list[str], *, timeout_s: float, tool: str) -> SimpleNamespace:
        calls.append((command, timeout_s, tool))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("yap_studio.transcript.backends.process.spawn", spawn)
    ParakeetMlxBackend().prefetch()

    assert calls == [(
        [
            str(python), "-c",
            "import sys; from huggingface_hub import snapshot_download; "
            "snapshot_download(repo_id=sys.argv[1])",
            "mlx-community/parakeet-tdt-0.6b-v3",
        ],
        3600,
        "speech_engine",
    )]
    (snapshot / "model.safetensors").unlink()
    with pytest.raises(OSError, match="incomplete"):
        ParakeetMlxBackend().prefetch()


@pytest.mark.parametrize("shebang", ["#!/bin/sh\n", "#!/missing/python3\n", "plain text\n"])
def test_mlx_prefetch_without_usable_cli_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shebang: str
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    executable = tmp_path / "parakeet-mlx"
    executable.write_text(shebang)
    monkeypatch.setattr(ParakeetMlxBackend, "executable", staticmethod(lambda: executable))
    with pytest.raises(OSError, match="python|Python"):
        ParakeetMlxBackend().prefetch()


def test_mlx_prefetch_reports_cli_download_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    python = tmp_path / "python3"
    python.write_text("")
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    executable = tmp_path / "parakeet-mlx"
    executable.write_text(f"#!{python}\n")
    monkeypatch.setattr(ParakeetMlxBackend, "executable", staticmethod(lambda: executable))
    monkeypatch.setattr(
        "yap_studio.transcript.backends.process.spawn",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stderr="download failed"),
    )
    with pytest.raises(OSError, match="download failed"):
        ParakeetMlxBackend().prefetch()


def test_onnx_prefetch_downloads_required_model_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _cached_onnx_snapshot(tmp_path, "int8")
    calls: list[dict[str, object]] = []
    module = types.ModuleType("huggingface_hub")

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot)

    module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    backend = ParakeetOnnxBackend()
    backend.prefetch()
    assert calls == [{
        "repo_id": "istupakov/parakeet-tdt-0.6b-v3-onnx",
        "allow_patterns": backend._model_files()[1],
    }]
    (snapshot / "vocab.txt").unlink()
    with pytest.raises(OSError, match="incomplete"):
        backend.prefetch()


def test_faster_whisper_notice_uses_local_cache_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "whisper"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.bin").write_bytes(b"model")
    calls: list[tuple[str, bool]] = []
    module = types.ModuleType("faster_whisper")

    def download_model(model: str, *, local_files_only: bool) -> str:
        calls.append((model, local_files_only))
        return str(snapshot)

    module.download_model = download_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    backend = FasterWhisperBackend()
    assert backend.download_notice() is None
    assert calls == [("small.en", True)]
    (snapshot / "model.bin").unlink()
    assert backend.download_notice() is not None


def test_faster_whisper_prefetch_downloads_without_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    module = types.ModuleType("faster_whisper")
    module.download_model = lambda model: calls.append(model)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    FasterWhisperBackend(model="small.en").prefetch()
    assert calls == ["small.en"]
    local = tmp_path / "whisper"
    local.mkdir()
    FasterWhisperBackend(model=str(local)).prefetch()
    assert calls == ["small.en"]


@pytest.mark.parametrize(
    ("quantization", "filename", "state", "needs_notice"),
    [
        ("int8", None, "complete", False),
        ("int8", "encoder-model.int8.onnx", "missing", True),
        ("int8", "encoder-model.int8.onnx", "dangling", True),
        ("int8", "encoder-model.int8.onnx", "empty", True),
        ("int8", "encoder-model.int8.onnx", "directory", True),
        ("int8", "decoder_joint-model.int8.onnx", "missing", True),
        ("int8", "config.json", "missing", True),
        ("int8", "vocab.txt", "missing", True),
        (None, "encoder-model.onnx.data", "dangling", True),
        (None, "encoder-model.onnx.data", "empty", True),
        (None, "encoder-model.onnx.data", "directory", True),
        (None, "encoder-model.onnx_data", "dangling", True),
        (None, "decoder_joint-model.onnx.data", "dangling", True),
        (None, "encoder-model.onnx.data", "present", False),
        (None, None, "complete", False),
    ],
)
def test_onnx_download_notice_validates_cached_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quantization: str | None,
    filename: str | None,
    state: str,
    needs_notice: bool,
) -> None:
    snapshot = _cached_onnx_snapshot(tmp_path, quantization)
    if filename is not None:
        path = snapshot / filename
        if path.exists():
            path.unlink()
        if state == "dangling":
            path.symlink_to(tmp_path / "missing-blob")
        elif state == "empty":
            path.write_bytes(b"")
        elif state == "directory":
            path.mkdir()
        elif state == "present":
            path.write_bytes(b"cached model data")
    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = lambda **kwargs: str(snapshot)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    notice = ParakeetOnnxBackend(quantization=quantization).download_notice()

    assert notice == (
        "First run downloads the speech model (about 670 MB)." if needs_notice else None
    )


def test_mlx_executable_prefers_active_python_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = tmp_path / "python"
    executable = tmp_path / "parakeet-mlx"
    python.write_text("")
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setattr("shutil.which", lambda _name: "/fallback/parakeet-mlx")
    assert ParakeetMlxBackend.executable() == executable


def test_mlx_fake_cli_batches_unique_stems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "parakeet-mlx"
    script.write_text(
        """#!/usr/bin/env python3
import json
import pathlib
import sys

args = sys.argv[1:]
first_flag = args.index("--model")
wavs = [pathlib.Path(value) for value in args[:first_flag]]
out = pathlib.Path(args[args.index("--output-dir") + 1])
for index, wav in enumerate(wavs):
    data = {
        "text": f"Bright {index}",
        "sentences": [{
            "tokens": [
                {"text": " B", "start": 0.0, "end": 0.08, "confidence": 0.9},
                {"text": "right", "start": 0.08, "end": 0.16, "confidence": 0.8},
                {"text": f" {index}", "start": 0.16, "end": 0.20, "confidence": 0.7},
            ]
        }],
    }
    (out / f"{wav.stem}.json").write_text(json.dumps(data))
"""
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(ParakeetMlxBackend, "executable", staticmethod(lambda: script))
    first = _wav(tmp_path / "same.wav")
    second_dir = tmp_path / "other"
    second_dir.mkdir()
    second = _wav(second_dir / "same.wav")

    transcripts = ParakeetMlxBackend(model="fixture").transcribe_many([first, second])

    assert list(transcripts) == [first, second]
    assert transcripts[first].text == "Bright 0"
    assert transcripts[second].text == "Bright 1"
    assert transcripts[first].backend == "parakeet-mlx"
    assert transcripts[first].model == "fixture"
    assert transcripts[first].duration_s == pytest.approx(0.2)


def _onnx_wav(
    path: Path,
    *,
    seconds: float,
    silent_gap: tuple[float, float] | None = None,
    continuous_tone: bool = False,
) -> Path:
    """Write synthetic mono audio efficiently, optionally planting a quiet cut."""
    sample_rate = 16_000
    samples = np.zeros(round(sample_rate * seconds), dtype="<i2")
    tone = (2_000 * np.sin(2 * np.pi * 220 * np.arange(1_600) / sample_rate)).astype("<i2")
    if continuous_tone:
        tone = (
            2_000 * np.sin(2 * np.pi * 220 * np.arange(sample_rate) / sample_rate)
        ).astype("<i2")
        for start in range(0, len(samples), sample_rate):
            end = min(start + sample_rate, len(samples))
            samples[start:end] = tone[: end - start]
    else:
        for start in range(0, len(samples) - len(tone), 100 * sample_rate):
            samples[start : start + len(tone)] = tone
    if silent_gap is not None:
        search_start = round(240 * sample_rate)
        search_end = round(300 * sample_rate)
        positions = np.arange(search_end - search_start)
        samples[search_start:search_end] = (
            2_000 * np.sin(2 * np.pi * 220 * positions / sample_rate)
        ).astype("<i2")
        gap_start, gap_end = silent_gap
        samples[round(gap_start * sample_rate) : round(gap_end * sample_rate)] = 0
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())
    return path


def _onnx_result(duration_s: float, *, include_logprobs: bool = True) -> SimpleNamespace:
    """Return one timestamped fake token per second of local chunk audio."""
    timestamps = np.arange(0.5, duration_s, 1.0).tolist()
    result = SimpleNamespace(
        tokens=[f" word{index}" for index in range(len(timestamps))],
        timestamps=timestamps,
    )
    if include_logprobs:
        result.logprobs = [-0.1 - index / 1_000 for index in range(len(timestamps))]
    return result


@dataclass
class _OnnxCall:
    """Paths, durations, and local token results observed in one fake call."""

    paths: list[Path]
    durations: list[float]
    results: list[SimpleNamespace]


@dataclass
class _FakeOnnxEngine:
    """Record recognition batches without loading a speech model."""

    calls: list[_OnnxCall] = field(default_factory=list)
    include_logprobs: bool = True
    single_result: bool = False
    chunk_results: list[SimpleNamespace] | None = None

    def with_timestamps(self) -> _FakeOnnxEngine:
        """Match the timestamp-enabled engine interface."""
        return self

    def recognize(self, paths: list[str]) -> list[SimpleNamespace] | SimpleNamespace:
        """Derive local timestamps from actual WAV headers and record the call."""
        durations = []
        for path in paths:
            with wave.open(path, "rb") as reader:
                assert reader.getframerate() == 16_000
                assert reader.getnchannels() == 1
                assert reader.getsampwidth() == 2
                durations.append(reader.getnframes() / reader.getframerate())
        if self.chunk_results is None:
            results = [
                _onnx_result(duration, include_logprobs=self.include_logprobs)
                for duration in durations
            ]
        else:
            result_index = sum(len(call.paths) for call in self.calls)
            results = self.chunk_results[result_index : result_index + len(paths)]
        self.calls.append(_OnnxCall([Path(path) for path in paths], durations, results))
        return results[0] if self.single_result and len(results) == 1 else results


@pytest.fixture
def fake_onnx(monkeypatch: pytest.MonkeyPatch) -> _FakeOnnxEngine:
    """Install a fake onnx-asr module for bounded recognition tests."""
    engine = _FakeOnnxEngine()
    module = types.ModuleType("onnx_asr")

    def load_model(_model: str, **_options: object) -> _FakeOnnxEngine:
        return engine

    module.load_model = load_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnx_asr", module)
    return engine


def test_parakeet_onnx_chunks_long_audio_and_offsets_timestamps(
    tmp_path: Path, fake_onnx: _FakeOnnxEngine
) -> None:
    path = _onnx_wav(tmp_path / "long.wav", seconds=900.0)

    transcript = ParakeetOnnxBackend(model="fixture", in_process=True).transcribe_many([path])[path]

    chunks = [
        (wav, duration, result)
        for call in fake_onnx.calls
        for wav, duration, result in zip(call.paths, call.durations, call.results, strict=True)
    ]
    assert len(chunks) >= 3
    assert all(0 < duration <= ONNX_MAX_CHUNK_S for _, duration, _ in chunks)
    assert all(sum(call.durations) <= ONNX_BATCH_SECONDS for call in fake_onnx.calls)
    assert sum(duration for _, duration, _ in chunks) == pytest.approx(900.0)
    assert [wav.name for wav, _, _ in chunks] == [
        f"0000-chunk-{index:04d}.wav" for index in range(len(chunks))
    ]
    assert len({wav.parent for wav, _, _ in chunks}) == 1
    expected_starts = []
    expected_text = []
    expected_confidence = []
    start_s = 0.0
    for _, duration, result in chunks:
        expected_starts.extend(start_s + timestamp for timestamp in result.timestamps)
        expected_text.extend(token.strip() for token in result.tokens)
        expected_confidence.extend(math.exp(logprob) for logprob in result.logprobs)
        start_s += duration
    starts = [word.start for word in transcript.words]
    assert starts == pytest.approx(expected_starts)
    assert all(left < right for left, right in zip(starts, starts[1:], strict=False))
    assert starts[0] == 0.5
    assert starts[-1] > 899.0
    second_chunk_token = len(chunks[0][2].tokens)
    assert starts[second_chunk_token] == pytest.approx(chunks[0][1] + 0.5)
    assert [word.text for word in transcript.words] == expected_text
    assert [word.conf for word in transcript.words] == pytest.approx(expected_confidence)
    assert transcript.duration_s == 900.0
    assert transcript.backend == "parakeet-onnx"
    assert transcript.model == "fixture"


def test_parakeet_onnx_preserves_word_boundary_between_chunks(
    tmp_path: Path, fake_onnx: _FakeOnnxEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    gap = (251.037, 251.437)
    path = _onnx_wav(tmp_path / "boundary.wav", seconds=900.0, silent_gap=gap)
    recognize = fake_onnx.recognize

    def boundary_tokens(paths: list[str]) -> list[SimpleNamespace]:
        results = recognize(paths)
        assert isinstance(results, list)
        assert len(results) == 1
        result = results[0]
        if len(fake_onnx.calls) == 1:
            result.tokens = [" good", "bye"]
            result.timestamps = [0.5, 0.58]
            result.logprobs = [-0.1, -0.2]
        elif len(fake_onnx.calls) == 2:
            result.tokens = ["hello", " there"]
            result.timestamps = [0.0, 0.5]
            result.logprobs = [-0.3, -0.4]
        else:
            result.tokens = []
            result.timestamps = []
            result.logprobs = []
        return results

    monkeypatch.setattr(fake_onnx, "recognize", boundary_tokens)

    transcript = ParakeetOnnxBackend(in_process=True).transcribe_many([path])[path]

    assert len(fake_onnx.calls) >= 3
    second_chunk_offset = fake_onnx.calls[0].durations[0]
    assert second_chunk_offset == pytest.approx(gap[0] + 0.1, abs=1 / 16_000)
    assert sum(sum(call.durations) for call in fake_onnx.calls) == pytest.approx(900.0)
    assert all(sum(call.durations) <= ONNX_BATCH_SECONDS for call in fake_onnx.calls)
    assert [word.text for word in transcript.words] == ["goodbye", "hello", "there"]
    assert transcript.words[0].end == pytest.approx(0.66)
    assert transcript.words[1].start == pytest.approx(second_chunk_offset)
    assert transcript.words[2].start == pytest.approx(second_chunk_offset + 0.5)
    assert [word.conf for word in transcript.words] == pytest.approx(
        [math.exp(-0.2), math.exp(-0.3), math.exp(-0.4)]
    )
    assert fake_onnx.calls[1].results[0].tokens == ["hello", " there"]
    assert transcript.duration_s == 900.0


@pytest.mark.parametrize("tail_s", [0.1, 0.16, 0.17])
def test_parakeet_onnx_overlaps_continuous_audio_and_stitches_words(
    tmp_path: Path,
    fake_onnx: _FakeOnnxEngine,
    monkeypatch: pytest.MonkeyPatch,
    tail_s: float,
) -> None:
    path = _onnx_wav(tmp_path / "continuous.wav", seconds=900.0, continuous_tone=True)
    recognize = fake_onnx.recognize
    truncated = tail_s < 0.16

    def overlapping_words(paths: list[str]) -> list[SimpleNamespace]:
        results = recognize(paths)
        assert isinstance(results, list)
        assert len(results) == 1
        result = results[0]
        if len(fake_onnx.calls) == 1:
            result.tokens = [" good"] if truncated else [" good", "bye"]
            result.timestamps = (
                [ONNX_MAX_CHUNK_S - tail_s]
                if truncated
                else [ONNX_MAX_CHUNK_S - OVERLAP_S, ONNX_MAX_CHUNK_S - tail_s]
            )
            result.logprobs = [-0.1] if truncated else [-0.1, -0.2]
        elif len(fake_onnx.calls) == 2:
            result.tokens = [" good", "bye", " there"]
            result.timestamps = (
                [OVERLAP_S - tail_s, OVERLAP_S + 0.05, OVERLAP_S + 1.0]
                if truncated
                else [0.0, OVERLAP_S - tail_s, OVERLAP_S + 1.0]
            )
            result.logprobs = [-0.3, -0.4, -0.5]
        else:
            result.tokens = []
            result.timestamps = []
            result.logprobs = []
        return results

    monkeypatch.setattr(fake_onnx, "recognize", overlapping_words)

    transcript = ParakeetOnnxBackend(in_process=True).transcribe_many([path])[path]

    assert [call.durations for call in fake_onnx.calls] == [[300.0], [300.0], [300.0], [6.0]]
    assert all(sum(call.durations) <= ONNX_BATCH_SECONDS for call in fake_onnx.calls)
    assert sum(sum(call.durations) for call in fake_onnx.calls) == 900.0 + 3 * OVERLAP_S
    assert [word.text for word in transcript.words] == ["goodbye", "there"]
    starts = [word.start for word in transcript.words]
    expected_start = ONNX_MAX_CHUNK_S - (tail_s if truncated else OVERLAP_S)
    assert starts == pytest.approx([expected_start, ONNX_MAX_CHUNK_S + 1.0])
    assert all(left < right for left, right in zip(starts, starts[1:], strict=False))
    expected_end = ONNX_MAX_CHUNK_S + 0.13 if truncated else ONNX_MAX_CHUNK_S - tail_s + 0.08
    assert transcript.words[0].end == pytest.approx(expected_end)
    assert [word.conf for word in transcript.words] == pytest.approx(
        [math.exp(-0.4 if truncated else -0.2), math.exp(-0.5)]
    )
    assert fake_onnx.calls[1].results[0].tokens == [" good", "bye", " there"]
    assert transcript.duration_s == 900.0


def test_onnx_splice_keeps_later_word_before_anchor() -> None:
    earlier = [
        _OnnxWord(tokens=[" before"], timestamps=[298.2], logprobs=[-0.1]),
        _OnnxWord(tokens=[" after"], timestamps=[299.5], logprobs=[-0.2]),
    ]
    later = [
        _OnnxWord(tokens=[" missed"], timestamps=[298.8], logprobs=[-0.3]),
        _OnnxWord(tokens=[" after"], timestamps=[299.54], logprobs=[-0.8]),
        _OnnxWord(tokens=[" there"], timestamps=[301.0], logprobs=[-0.5]),
    ]

    words = _onnx_splice(earlier, later, start_s=298.0, end_s=300.0)

    assert [word.normalized_text for word in words] == ["before", "missed", "after", "there"]
    assert [word.start_s for word in words] == pytest.approx([298.2, 298.8, 299.5, 301.0])


def test_onnx_splice_keeps_earlier_word_after_anchor() -> None:
    earlier = [
        _OnnxWord(tokens=[" before"], timestamps=[298.2], logprobs=[-0.1]),
        _OnnxWord(tokens=[" anchor"], timestamps=[299.0], logprobs=[-0.2]),
        _OnnxWord(tokens=[" missed"], timestamps=[299.5], logprobs=[-0.3]),
    ]
    later = [
        _OnnxWord(tokens=[" anchor"], timestamps=[299.04], logprobs=[-0.8]),
        _OnnxWord(tokens=[" there"], timestamps=[301.0], logprobs=[-0.5]),
    ]

    words = _onnx_splice(earlier, later, start_s=298.0, end_s=300.0)

    assert [word.normalized_text for word in words] == ["before", "anchor", "missed", "there"]
    assert [word.start_s for word in words] == pytest.approx([298.2, 299.0, 299.5, 301.0])


@pytest.mark.parametrize("before_anchor", [True, False])
@pytest.mark.parametrize(
    ("gap_s", "conflicts"), [(-0.04, True), (0.04, True), (0.08, False), (0.081, False)]
)
def test_onnx_splice_drops_conflicting_alternate_reading(
    before_anchor: bool, gap_s: float, conflicts: bool
) -> None:
    earlier = [
        _OnnxWord(tokens=[" th", "ere"], timestamps=[298.4, 298.8], logprobs=[-0.1, -0.2]),
        _OnnxWord(tokens=[" anchor"], timestamps=[299.0], logprobs=[-0.2]),
        _OnnxWord(tokens=[" th", "ere"], timestamps=[299.5, 299.8], logprobs=[-0.3, -0.4]),
    ]
    later = [
        _OnnxWord(tokens=[" their"], timestamps=[298.8 + gap_s], logprobs=[-0.4]),
        _OnnxWord(tokens=[" anchor"], timestamps=[299.04], logprobs=[-0.8]),
        _OnnxWord(tokens=[" their"], timestamps=[299.8 + gap_s], logprobs=[-0.5]),
    ]
    if before_anchor:
        earlier = earlier[:2]
        later = later[:2]
        expected = [earlier[0], earlier[1]] if conflicts else [earlier[0], later[0], earlier[1]]
    else:
        earlier = earlier[1:]
        later = later[1:]
        expected = [earlier[0], later[1]] if conflicts else [earlier[0], earlier[1], later[1]]

    words = _onnx_splice(earlier, later, start_s=298.0, end_s=300.0)

    assert words == expected


def test_onnx_splice_preserves_order_and_token_alignment() -> None:
    earlier = [
        _OnnxWord(tokens=[" be", "fore"], timestamps=[298.2, 298.3], logprobs=[-0.1, -0.2]),
        _OnnxWord(tokens=[" an", "chor"], timestamps=[299.0, 299.1], logprobs=[-0.3, -0.4]),
        _OnnxWord(tokens=[" ear", "lier"], timestamps=[299.5, 299.6], logprobs=[-0.5, -0.6]),
    ]
    later = [
        _OnnxWord(tokens=[" la", "ter"], timestamps=[298.8, 298.9], logprobs=[-0.7, -0.8]),
        _OnnxWord(tokens=[" anchor"], timestamps=[299.04], logprobs=[-0.9]),
        _OnnxWord(tokens=[" there"], timestamps=[301.0], logprobs=[-1.0]),
    ]

    words = _onnx_splice(earlier, later, start_s=298.0, end_s=300.0)

    assert [word.start_s for word in words] == sorted(word.start_s for word in words)
    assert len(words) == 5
    for word, original in zip(
        words, [earlier[0], later[0], earlier[1], earlier[2], later[2]], strict=True
    ):
        assert word is original
    assert [word.tokens for word in words] == [
        [" be", "fore"], [" la", "ter"], [" an", "chor"], [" ear", "lier"], [" there"]
    ]
    assert [word.timestamps for word in words] == [
        [298.2, 298.3], [298.8, 298.9], [299.0, 299.1], [299.5, 299.6], [301.0]
    ]
    assert [word.logprobs for word in words] == [
        [-0.1, -0.2], [-0.7, -0.8], [-0.3, -0.4], [-0.5, -0.6], [-1.0]
    ]


def test_parakeet_onnx_keeps_words_only_recognized_in_overlap(
    tmp_path: Path, fake_onnx: _FakeOnnxEngine
) -> None:
    path = _onnx_wav(tmp_path / "missing.wav", seconds=310.0, continuous_tone=True)
    fake_onnx.chunk_results = [
        SimpleNamespace(
            tokens=[" before", " after"],
            timestamps=[298.2, 299.5],
            logprobs=[-0.1, -0.2],
        ),
        SimpleNamespace(
            tokens=[" after", " mis", "sed", " there"],
            timestamps=[1.54, 1.8, 2.2, 3.0],
            logprobs=[-0.8, -0.3, -0.4, -0.5],
        ),
    ]

    transcript = ParakeetOnnxBackend(in_process=True).transcribe_many([path])[path]

    assert len(fake_onnx.calls) == 2
    expected = sorted(
        [
            ("before", 298.2, -0.1),
            ("after", 299.5, -0.2),
            ("missed", 299.8, -0.4),
            ("there", 301.0, -0.5),
        ],
        key=lambda item: item[1],
    )
    assert [word.text for word in transcript.words] == [text for text, _, _ in expected]
    starts = [word.start for word in transcript.words]
    assert starts == pytest.approx([start for _, start, _ in expected])
    assert all(left < right for left, right in zip(starts, starts[1:], strict=False))
    assert [word.conf for word in transcript.words] == pytest.approx(
        [math.exp(logprob) for _, _, logprob in expected]
    )
    missed = next(word for word in transcript.words if word.text == "missed")
    assert missed.end == pytest.approx(300.28)


@pytest.mark.parametrize("later_repeats_both", [True, False])
def test_parakeet_onnx_preserves_repeated_words_at_overlap_cut(
    tmp_path: Path, fake_onnx: _FakeOnnxEngine, later_repeats_both: bool
) -> None:
    path = _onnx_wav(tmp_path / "repeated.wav", seconds=310.0, continuous_tone=True)
    fake_onnx.chunk_results = [
        SimpleNamespace(tokens=[" no", " no"], timestamps=[299.78, 299.90], logprobs=[-0.1, -0.2]),
        SimpleNamespace(
            tokens=[" no", " no", " there"] if later_repeats_both else [" no", " there"],
            timestamps=[1.78, 1.90, 3.0] if later_repeats_both else [1.90, 3.0],
            logprobs=[-0.3, -0.4, -0.5] if later_repeats_both else [-0.4, -0.5],
        ),
    ]

    transcript = ParakeetOnnxBackend(in_process=True).transcribe_many([path])[path]

    assert [word.text for word in transcript.words] == ["no", "no", "there"]
    starts = [word.start for word in transcript.words]
    assert starts == pytest.approx([299.78, 299.90, 301.0])
    assert all(left < right for left, right in zip(starts, starts[1:], strict=False))
    assert [word.conf for word in transcript.words] == pytest.approx(
        [math.exp(-0.1), math.exp(-0.4 if later_repeats_both else -0.2), math.exp(-0.5)]
    )


def test_parakeet_onnx_keeps_one_alternate_recognition_in_overlap(
    tmp_path: Path, fake_onnx: _FakeOnnxEngine
) -> None:
    path = _onnx_wav(tmp_path / "alternate.wav", seconds=310.0, continuous_tone=True)
    fake_onnx.chunk_results = [
        SimpleNamespace(tokens=[" there"], timestamps=[299.0], logprobs=[-0.1]),
        SimpleNamespace(tokens=[" their", " next"], timestamps=[1.04, 3.0], logprobs=[-0.2, -0.3]),
    ]

    transcript = ParakeetOnnxBackend(in_process=True).transcribe_many([path])[path]

    assert [word.text for word in transcript.words] == ["there", "next"]
    starts = [word.start for word in transcript.words]
    assert starts == pytest.approx([299.0, 301.0])
    assert all(left < right for left, right in zip(starts, starts[1:], strict=False))
    assert [word.conf for word in transcript.words] == pytest.approx(
        [math.exp(-0.1), math.exp(-0.3)]
    )


def test_parakeet_onnx_breaks_anchor_time_difference_ties_at_midpoint(
    tmp_path: Path, fake_onnx: _FakeOnnxEngine
) -> None:
    path = _onnx_wav(tmp_path / "anchor.wav", seconds=310.0, continuous_tone=True)
    fake_onnx.chunk_results = [
        SimpleNamespace(
            tokens=[" first", " anchor", " final"],
            timestamps=[298.2, 299.04, 299.8],
            logprobs=[-0.1, -0.2, -0.3],
        ),
        SimpleNamespace(
            tokens=[" first", " anchor", " final", " extra"],
            timestamps=[0.24, 1.0, 1.84, 3.0],
            logprobs=[-0.4, -0.5, -0.6, -0.7],
        ),
    ]

    transcript = ParakeetOnnxBackend(in_process=True).transcribe_many([path])[path]

    assert [word.text for word in transcript.words] == ["first", "anchor", "final", "extra"]
    starts = [word.start for word in transcript.words]
    assert starts == pytest.approx([298.2, 299.04, 299.84, 301.0])
    assert all(left < right for left, right in zip(starts, starts[1:], strict=False))
    assert [word.conf for word in transcript.words] == pytest.approx(
        [math.exp(-0.1), math.exp(-0.2), math.exp(-0.6), math.exp(-0.7)]
    )


def test_parakeet_onnx_keeps_word_at_fallback_frame_boundary(
    tmp_path: Path, fake_onnx: _FakeOnnxEngine
) -> None:
    path = _onnx_wav(tmp_path / "frame-boundary.wav", seconds=310.0, continuous_tone=True)
    fake_onnx.chunk_results = [
        SimpleNamespace(tokens=[" a"], timestamps=[299.0], logprobs=[-0.1]),
        SimpleNamespace(tokens=[" \u0063at"], timestamps=[1.08], logprobs=[-0.2]),
    ]

    transcript = ParakeetOnnxBackend(in_process=True).transcribe_many([path])[path]

    assert len(fake_onnx.calls) == 2
    assert [word.text for word in transcript.words] == ["a", "\u0063at"]
    starts = [word.start for word in transcript.words]
    assert starts == pytest.approx([299.0, 299.08])
    assert all(left < right for left, right in zip(starts, starts[1:], strict=False))
    assert transcript.words[0].end == pytest.approx(299.08)
    assert [word.conf for word in transcript.words] == pytest.approx(
        [math.exp(-0.1), math.exp(-0.2)]
    )


def test_parakeet_onnx_splices_after_complete_word_without_anchor(
    tmp_path: Path, fake_onnx: _FakeOnnxEngine
) -> None:
    path = _onnx_wav(tmp_path / "fallback.wav", seconds=310.0, continuous_tone=True)
    fake_onnx.chunk_results = [
        SimpleNamespace(
            tokens=[" complete", " tail"], timestamps=[299.0, 299.9], logprobs=[-0.1, -0.2]
        ),
        SimpleNamespace(
            tokens=[" different", "tail", " next"],
            timestamps=[1.5, 2.1, 3.0],
            logprobs=[-0.3, -0.4, -0.5],
        ),
    ]

    transcript = ParakeetOnnxBackend(in_process=True).transcribe_many([path])[path]

    assert [word.text for word in transcript.words] == ["complete", "differenttail", "next"]
    starts = [word.start for word in transcript.words]
    assert starts == pytest.approx([299.0, 299.5, 301.0])
    assert all(left < right for left, right in zip(starts, starts[1:], strict=False))
    assert transcript.words[1].end == pytest.approx(300.18)
    assert [word.conf for word in transcript.words] == pytest.approx(
        [math.exp(-0.1), math.exp(-0.4), math.exp(-0.5)]
    )


@pytest.mark.parametrize(
    ("earlier_token", "later_token"),
    [
        (" the", " the"),
        (" The!", " the"),
        (" the", "THE?"),
        (" th", " the"),
        (" the", " th"),
        (" the", " \u201cTHE\u201d"),
    ],
)
def test_parakeet_onnx_keeps_distinct_neighbour_in_overlap(
    tmp_path: Path, fake_onnx: _FakeOnnxEngine, earlier_token: str, later_token: str
) -> None:
    path = _onnx_wav(tmp_path / "neighbour.wav", seconds=310.0, continuous_tone=True)
    fake_onnx.chunk_results = [
        SimpleNamespace(tokens=[earlier_token], timestamps=[299.70], logprobs=[-0.1]),
        SimpleNamespace(
            tokens=[later_token, " \u0063at"],
            timestamps=[1.69, 1.78],
            logprobs=[-0.9, -0.2],
        ),
    ]

    transcript = ParakeetOnnxBackend(in_process=True).transcribe_many([path])[path]

    assert len(fake_onnx.calls) == 2
    assert [word.text for word in transcript.words] == [earlier_token.strip(), "\u0063at"]
    starts = [word.start for word in transcript.words]
    assert starts == pytest.approx([299.70, 299.78])
    assert all(left < right for left, right in zip(starts, starts[1:], strict=False))
    assert [word.conf for word in transcript.words] == pytest.approx(
        [math.exp(-0.1), math.exp(-0.2)]
    )


def test_parakeet_onnx_recovers_word_with_backward_timestamp_shift(
    tmp_path: Path, fake_onnx: _FakeOnnxEngine
) -> None:
    path = _onnx_wav(tmp_path / "replacement.wav", seconds=310.0, continuous_tone=True)
    fake_onnx.chunk_results = [
        SimpleNamespace(
            tokens=[" ready", " good"], timestamps=[298.0, 299.9], logprobs=[-0.1, -0.2]
        ),
        SimpleNamespace(
            tokens=[" good", "bye", " there"],
            timestamps=[1.85, 2.2, 3.0],
            logprobs=[-0.3, -0.4, -0.5],
        ),
    ]

    transcript = ParakeetOnnxBackend(in_process=True).transcribe_many([path])[path]

    assert len(fake_onnx.calls) == 2
    assert [word.text for word in transcript.words] == ["ready", "goodbye", "there"]
    starts = [word.start for word in transcript.words]
    assert starts == pytest.approx([298.0, 299.85, 301.0])
    assert all(left < right for left, right in zip(starts, starts[1:], strict=False))
    assert transcript.words[1].end == pytest.approx(300.28)
    assert [word.conf for word in transcript.words] == pytest.approx(
        [math.exp(-0.1), math.exp(-0.4), math.exp(-0.5)]
    )


@pytest.mark.parametrize("shift_s", [0.04, 0.12])
def test_parakeet_onnx_deduplicates_words_with_shifted_timestamps(
    tmp_path: Path, fake_onnx: _FakeOnnxEngine, shift_s: float
) -> None:
    path = _onnx_wav(tmp_path / "duplicate.wav", seconds=310.0, continuous_tone=True)
    fake_onnx.chunk_results = [
        SimpleNamespace(
            tokens=[" shared", "word"], timestamps=[299.0, 299.02], logprobs=[-0.1, -0.2]
        ),
        SimpleNamespace(
            tokens=[" shared", "word", " there"],
            timestamps=[1.0 + shift_s, 1.02 + shift_s, 3.0],
            logprobs=[-0.6, -0.7, -0.5],
        ),
    ]

    transcript = ParakeetOnnxBackend(in_process=True).transcribe_many([path])[path]

    assert len(fake_onnx.calls) == 2
    assert [word.text for word in transcript.words] == ["sharedword", "there"]
    starts = [word.start for word in transcript.words]
    assert starts == pytest.approx([299.0, 301.0])
    assert all(left < right for left, right in zip(starts, starts[1:], strict=False))
    assert transcript.words[0].end == pytest.approx(299.1)
    assert [word.conf for word in transcript.words] == pytest.approx(
        [math.exp(-0.2), math.exp(-0.5)]
    )


def test_parakeet_onnx_cuts_inside_quiet_gap(
    tmp_path: Path, fake_onnx: _FakeOnnxEngine
) -> None:
    gap = (291.037, 291.437)
    path = _onnx_wav(tmp_path / "quiet.wav", seconds=310.0, silent_gap=gap)

    transcript = ParakeetOnnxBackend(in_process=True).transcribe_many([path])[path]

    first_duration = fake_onnx.calls[0].durations[0]
    assert gap[0] + 0.1 <= first_duration <= gap[1] - 0.1
    assert first_duration == pytest.approx(gap[0] + 0.1, abs=1 / 16_000)
    assert sum(sum(call.durations) for call in fake_onnx.calls) == pytest.approx(310.0)
    next_token = len(fake_onnx.calls[0].results[0].tokens)
    assert transcript.words[next_token].start == pytest.approx(first_duration + 0.5)


@pytest.mark.parametrize("seconds", [120.0, ONNX_MAX_CHUNK_S])
@pytest.mark.parametrize("include_logprobs", [True, False])
@pytest.mark.parametrize("single_result", [True, False])
def test_parakeet_onnx_preserves_unsplit_transcript(
    tmp_path: Path,
    fake_onnx: _FakeOnnxEngine,
    seconds: float,
    include_logprobs: bool,
    single_result: bool,
) -> None:
    path = _onnx_wav(tmp_path / "short.wav", seconds=seconds)
    fake_onnx.include_logprobs = include_logprobs
    fake_onnx.single_result = single_result
    result = _onnx_result(seconds, include_logprobs=include_logprobs)
    expected = from_parakeet_tokens(
        result.tokens,
        result.timestamps,
        logprobs=getattr(result, "logprobs", None),
        duration_s=seconds,
        backend="parakeet-onnx",
        model="fixture",
    )

    transcript = ParakeetOnnxBackend(model="fixture", in_process=True).transcribe_many([path])[path]

    assert transcript == expected
    assert len(fake_onnx.calls) == 1
    assert [wav.name for wav in fake_onnx.calls[0].paths] == ["0000.wav"]
    assert fake_onnx.calls[0].durations == [seconds]


def test_parakeet_onnx_batches_three_short_clips(
    tmp_path: Path, fake_onnx: _FakeOnnxEngine
) -> None:
    paths = [_onnx_wav(tmp_path / f"clip-{index}.wav", seconds=5.0) for index in range(3)]

    transcripts = ParakeetOnnxBackend(in_process=True).transcribe_many(paths)

    assert list(transcripts) == paths
    assert len(fake_onnx.calls) == 1
    assert fake_onnx.calls[0].durations == [5.0, 5.0, 5.0]
    assert [wav.name for wav in fake_onnx.calls[0].paths] == [
        "0000.wav", "0001.wav", "0002.wav"
    ]
    for path in paths:
        assert [word.start for word in transcripts[path].words] == [0.5, 1.5, 2.5, 3.5, 4.5]
        assert transcripts[path].duration_s == 5.0


def test_parakeet_onnx_bounds_short_file_batches(
    tmp_path: Path, fake_onnx: _FakeOnnxEngine
) -> None:
    seconds = [120.0, 180.0, 5.0]
    paths = [
        _onnx_wav(tmp_path / f"clip-{index}.wav", seconds=duration)
        for index, duration in enumerate(seconds)
    ]

    transcripts = ParakeetOnnxBackend(in_process=True).transcribe_many(paths)

    assert list(transcripts) == paths
    assert [call.durations for call in fake_onnx.calls] == [[120.0, 180.0], [5.0]]
    assert [wav.name for call in fake_onnx.calls for wav in call.paths] == [
        "0000.wav", "0001.wav", "0002.wav"
    ]
    for path, duration in zip(paths, seconds, strict=True):
        assert transcripts[path].duration_s == duration
        assert transcripts[path].words[0].start == 0.5


def test_parakeet_onnx_checks_result_count_per_batch(
    tmp_path: Path, fake_onnx: _FakeOnnxEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [_onnx_wav(tmp_path / f"clip-{index}.wav", seconds=180.0) for index in range(2)]
    recognize = fake_onnx.recognize

    def wrong_count(paths: list[str]) -> list[SimpleNamespace] | SimpleNamespace:
        results = recognize(paths)
        return [] if len(fake_onnx.calls) == 2 else results

    monkeypatch.setattr(fake_onnx, "recognize", wrong_count)

    with pytest.raises(ProcessingError, match="wrong result count") as caught:
        ParakeetOnnxBackend(model="fixture", in_process=True).transcribe_many(paths)

    assert caught.value.code == "speech_engine_failed"
    assert caught.value.command == "onnx-asr fixture"
    assert len(fake_onnx.calls) == 2


def test_parakeet_onnx_wraps_recognition_errors(
    tmp_path: Path, fake_onnx: _FakeOnnxEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _onnx_wav(tmp_path / "short.wav", seconds=5.0)

    def fail(_paths: list[str]) -> list[SimpleNamespace]:
        raise RuntimeError("encoder failed")

    monkeypatch.setattr(fake_onnx, "recognize", fail)

    with pytest.raises(ProcessingError, match="encoder failed") as caught:
        ParakeetOnnxBackend(model="fixture", in_process=True).transcribe_many([path])

    assert caught.value.code == "speech_engine_failed"
    assert caught.value.command == "onnx-asr fixture"


def test_scripted_backend_uses_original_paths(tmp_path: Path) -> None:
    path = tmp_path / "input.wav"
    seen: list[Path] = []

    def scripted(source: Path) -> Transcript:
        seen.append(source)
        return Transcript([Word(source.stem, 0.0, 1.0)])

    backend = ScriptedBackend(scripted, name="fixture", model="v1")
    assert backend.version() == "0"
    assert backend.download_notice() is None
    assert backend.transcribe_many([path])[path].text == "input"
    assert seen == [path]


def test_faster_whisper_backend_loads_once_and_uses_safe_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _wav(tmp_path / "first.wav")
    second = _wav(tmp_path / "second.wav")
    loads: list[tuple[str, str]] = []
    calls: list[dict[str, object]] = []

    class FakeModel:
        def __init__(self, model: str, *, compute_type: str) -> None:
            loads.append((model, compute_type))

        def transcribe(self, _path: str, **kwargs: object) -> tuple[list[object], object]:
            calls.append(kwargs)
            word = SimpleNamespace(word=" hello", start=0.0, end=0.1, probability=0.88)
            return [SimpleNamespace(words=[word])], SimpleNamespace(language="en")

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)

    backend = FasterWhisperBackend(model="fixture", compute_type="int8")
    transcripts = backend.transcribe_many([first, second])

    assert loads == [("fixture", "int8")]
    assert calls == [
        {
            "language": None,
            "word_timestamps": True,
            "vad_filter": False,
            "condition_on_previous_text": False,
        },
        {
            "language": None,
            "word_timestamps": True,
            "vad_filter": False,
            "condition_on_previous_text": False,
        },
    ]
    assert transcripts[first].words == [Word("hello", 0.0, 0.1, 0.88)]
    assert transcripts[second].language == "en"


def _speech_paths(synth_media: object) -> list[Path]:
    paths = []
    for index, sentence in enumerate(SENTENCES):
        path = synth_media.cached(
            f"asr-sentence-{index}",
            lambda out, text=sentence: synth.speech_source(out, [text]),
            suffix=".mp4",
        )
        paths.append(path)
    return paths


def _bag_recall(expected: list[str], got: list[str]) -> float:
    expected_counts = Counter(expected)
    got_counts = Counter(got)
    matches = sum((expected_counts & got_counts).values())
    return matches / sum(expected_counts.values())


@pytest.mark.asr
def test_parakeet_onnx_recovers_espeak_words(synth_media: object) -> None:
    if not ParakeetOnnxBackend.available():
        pytest.skip("onnx-asr is not importable")
    paths = _speech_paths(synth_media)
    transcripts = ParakeetOnnxBackend().transcribe_many(paths)
    recalls = []
    for path, sentence in zip(paths, SENTENCES, strict=True):
        recall = _bag_recall(norm_words(sentence), norm_words(transcripts[path].text))
        recalls.append(recall)
        assert recall >= 0.9, (sentence, transcripts[path].text, recall)
    print("parakeet-onnx bag-of-words recall:", ", ".join(f"{value:.3f}" for value in recalls))


@pytest.mark.asr
def test_parakeet_mlx_and_onnx_agree_on_espeak_words(synth_media: object) -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        pytest.skip("engine comparison requires Apple Silicon")
    if not ParakeetMlxBackend.available() or not ParakeetOnnxBackend.available():
        pytest.skip("both Parakeet engines are required")
    paths = _speech_paths(synth_media)
    mlx = ParakeetMlxBackend().transcribe_many(paths)
    onnx = ParakeetOnnxBackend().transcribe_many(paths)
    for path in paths:
        assert norm_words(mlx[path].text) == norm_words(onnx[path].text)
