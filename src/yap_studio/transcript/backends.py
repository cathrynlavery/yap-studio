"""Speech-engine backends behind one transcript interface."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import logging
import os
import platform
import shlex
import shutil
import sys
import tempfile
import wave
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from yap_studio import ffmpeg, process
from yap_studio.errors import DependencyError, ProcessingError, ValidationError

from .mappers import from_parakeet, from_parakeet_tokens
from .model import Transcript, Word

if TYPE_CHECKING:
    from .engine_child import ChildRun

ENGINES = ("parakeet-mlx", "parakeet-onnx", "faster-whisper")
ENGINE_FAMILIES = {
    "parakeet-mlx": "parakeet",
    "parakeet-onnx": "parakeet",
    "faster-whisper": "whisper",
}
ONNX_MAX_CHUNK_S = 300.0
ONNX_BATCH_SECONDS = 300.0
OVERLAP_S = 2.0
_ONNX_CUT_LOOKBACK_S = 60.0
_ONNX_QUIET_WINDOW_S = 0.2
_ONNX_SILENCE_DBFS = -40.0
_ONNX_TRUNCATED_WORD_S = 0.16
_ONNX_ANCHOR_MAX_SHIFT_S = 0.12
_ONNX_TOKEN_FRAME_S = 0.08


class Backend(Protocol):
    """Common interface implemented by speech engines."""

    name: str
    model: str

    def version(self) -> str | None:
        """Return the installed engine version, when known."""
        ...

    def transcribe_many(
        self,
        paths: Sequence[Path],
        *,
        language: str | None = None,
    ) -> dict[Path, Transcript]:
        """Transcribe paths with one model load."""
        ...

    def download_notice(self) -> str | None:
        """Describe a model download needed on first use."""
        ...

    def prefetch(self) -> None:
        """Download the model before the first transcription."""
        ...


@contextmanager
def _converted_wavs(
    paths: Sequence[Path],
) -> Iterator[tuple[list[Path], list[Path], list[float]]]:
    originals = [Path(path) for path in paths]
    with tempfile.TemporaryDirectory(prefix="yap-asr-") as temporary:
        directory = Path(temporary)
        converted: list[Path] = []
        durations: list[float] = []
        for index, source in enumerate(originals):
            output = directory / f"{index:04d}.wav"
            ffmpeg.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-map",
                    "0:a:0",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(output),
                ],
                timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
            )
            info = ffmpeg.probe(output, count_frames=False)
            duration = info.duration_s if info.duration_s is not None else info.audio_duration_s
            if duration is None:
                raise ProcessingError(
                    "ffprobe converted speech input",
                    1,
                    f"could not determine duration of {output.name}",
                    code="speech_engine_failed",
                )
            converted.append(output)
            durations.append(duration)
        yield originals, converted, durations


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _model_download_notice(
    repo_id: str,
    size: str,
    allow_patterns: Sequence[str] | None = None,
    required_files: Sequence[str] | None = None,
) -> str | None:
    notice = f"First run downloads the speech model ({size})."
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        try:
            return None if _cached_model_ready(repo_id, required_files or ()) else notice
        except OSError:
            return notice
    try:
        snapshot = snapshot_download(
            repo_id=repo_id, local_files_only=True, allow_patterns=allow_patterns
        )
        if not _snapshot_ready(Path(snapshot), required_files or ()):
            return notice
    except Exception:
        return notice
    return None


def _cached_model_ready(repo_id: str, required_files: Sequence[str]) -> bool:
    cache = os.environ.get("HF_HUB_CACHE")
    home = os.environ.get("HF_HOME")
    if cache:
        root = Path(cache)
    elif home:
        root = Path(home) / "hub"
    else:
        root = Path.home() / ".cache" / "huggingface" / "hub"
    repo = root.expanduser() / f"models--{repo_id.replace('/', '--')}"
    snapshots = repo / "snapshots"
    ref = repo / "refs" / "main"
    if ref.is_file():
        candidates = [snapshots / ref.read_text(encoding="utf-8").strip()]
    elif snapshots.is_dir():
        candidates = [path for path in snapshots.iterdir() if path.is_dir()]
    else:
        return False
    return any(_snapshot_ready(path, required_files) for path in candidates)


def _snapshot_ready(snapshot: Path, required_files: Sequence[str]) -> bool:
    if not snapshot.is_dir():
        return False
    if required_files:
        return _model_files_ready(snapshot, required_files)
    return any(path.is_file() and path.stat().st_size > 0 for path in snapshot.rglob("*"))


def _model_files_ready(snapshot: Path, required_files: Sequence[str]) -> bool:
    for name in required_files:
        model_file = snapshot / name
        files = [model_file]
        if model_file.suffix == ".onnx":
            files.extend(model_file.parent.glob(f"{model_file.name}?data"))
        if any(not path.is_file() or path.stat().st_size == 0 for path in files):
            return False
    return True


def _prefetch_model(
    repo_id: str,
    allow_patterns: Sequence[str] | None = None,
    required_files: Sequence[str] = (),
) -> None:
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns))
    if not _model_files_ready(snapshot, required_files):
        raise OSError(f"the downloaded speech model is incomplete: {repo_id}")


def _processing_error(
    command: Sequence[str] | str,
    returncode: int,
    stderr: Any,
) -> ProcessingError:
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    rendered = command if isinstance(command, str) else shlex.join(str(part) for part in command)
    return ProcessingError(
        rendered,
        returncode,
        str(stderr or "speech engine failed"),
        code="speech_engine_failed",
    )


class ParakeetMlxBackend:
    """Apple Silicon Parakeet backend driven through its batch CLI."""

    name = "parakeet-mlx"
    required_files = ("config.json", "model.safetensors")

    def __init__(self, model: str = "mlx-community/parakeet-tdt-0.6b-v3") -> None:
        self.model = model

    @staticmethod
    def executable() -> Path | None:
        """Find the engine CLI, preferring the active Python environment."""
        adjacent = Path(sys.executable).parent / "parakeet-mlx"
        if adjacent.is_file() and os.access(adjacent, os.X_OK):
            return adjacent
        found = shutil.which("parakeet-mlx")
        return Path(found) if found is not None else None

    @staticmethod
    def available() -> bool:
        """Return whether the backend can run on this machine."""
        return (
            sys.platform == "darwin"
            and platform.machine() == "arm64"
            and ParakeetMlxBackend.executable() is not None
        )

    def version(self) -> str | None:
        return _distribution_version("parakeet-mlx")

    def download_notice(self) -> str | None:
        return _model_download_notice(self.model, "about 2.5 GB",
                                      required_files=self.required_files)

    def prefetch(self) -> None:
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            executable = self.executable()
            try:
                if executable is None:
                    raise OSError("CLI is missing")
                with executable.open(encoding="utf-8") as script:
                    shebang = script.readline().strip()
                python = Path(shebang[2:].strip()) if shebang.startswith("#!") else None
                if (
                    python is None
                    or not python.name.startswith("python")
                    or not python.is_file()
                    or not os.access(python, os.X_OK)
                ):
                    raise OSError("CLI has no usable Python interpreter")
            except (OSError, UnicodeError) as exc:
                raise OSError(
                    "Cannot download the speech model: install huggingface_hub in "
                    "Yap Studio or reinstall parakeet-mlx with a Python CLI."
                ) from exc
            command = [
                str(python), "-c",
                "import sys; from huggingface_hub import snapshot_download; "
                "snapshot_download(repo_id=sys.argv[1])",
                self.model,
            ]
            try:
                result = process.spawn(command, timeout_s=3600, tool="speech_engine")
            except ProcessingError as exc:
                raise OSError(exc.full_stderr) from exc
            if result.returncode:
                raise OSError(result.stderr or "speech model download failed") from None
            if not _cached_model_ready(self.model, self.required_files):
                raise OSError(f"the downloaded speech model is incomplete: {self.model}") from None
        else:
            _prefetch_model(self.model, required_files=self.required_files)

    def transcribe_many(
        self,
        paths: Sequence[Path],
        *,
        language: str | None = None,
    ) -> dict[Path, Transcript]:
        del language
        if not paths:
            return {}
        executable = self.executable()
        if executable is None:
            raise DependencyError.missing_engine(self.name)
        with _converted_wavs(paths) as (originals, converted, durations):
            output_dir = converted[0].parent / "json"
            output_dir.mkdir()
            command = [
                str(executable),
                *(str(path) for path in converted),
                "--model",
                self.model,
                "--output-format",
                "json",
                "--output-dir",
                str(output_dir),
            ]
            try:
                result = process.spawn(
                    command,
                    timeout_s=max(300 + 30 * len(converted),
                                  120 + 30 * len(converted) + 1.5 * sum(durations)),
                    tool="speech_engine",
                )
            except ProcessingError as exc:
                raise _processing_error(command, exc.returncode, exc.full_stderr) from None
            if result.returncode:
                raise _processing_error(command, result.returncode, result.stderr)

            transcripts: dict[Path, Transcript] = {}
            for original, wav, duration in zip(originals, converted, durations, strict=True):
                result_path = output_dir / f"{wav.stem}.json"
                if not result_path.is_file():
                    raise _processing_error(
                        command, result.returncode, f"missing {result_path.name}"
                    )
                try:
                    data = json.loads(result_path.read_text(encoding="utf-8"))
                    transcript = from_parakeet(data, backend=self.name, model=self.model)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    raise _processing_error(command, result.returncode, str(exc)) from exc
                transcript.duration_s = duration
                transcripts[original] = transcript
            return transcripts


@dataclass(frozen=True)
class _OnnxChunk:
    """A bounded WAV segment with its position in an original input."""

    path: Path
    original_index: int
    start_s: float
    duration_s: float
    overlap_cut_s: float | None = None


@dataclass
class _OnnxWord:
    """A chunk word retaining its tokens, file timestamps, and logprobs."""

    tokens: list[str] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    logprobs: list[float] | None = None

    @property
    def start_s(self) -> float:
        """Return the first token's time in the original input."""
        return self.timestamps[0]

    @property
    def end_s(self) -> float:
        """Return the last token's time, without adding a frame duration."""
        return self.timestamps[-1]

    @property
    def normalized_text(self) -> str:
        """Return lowercase word text with punctuation removed for alignment."""
        return "".join(
            character for character in "".join(self.tokens).lower() if character.isalnum()
        )


def _onnx_splice(
    earlier: Sequence[_OnnxWord],
    later: Sequence[_OnnxWord],
    *,
    start_s: float,
    end_s: float,
) -> list[_OnnxWord]:
    """Splice overlapping sequences at a matched anchor or a complete word's end.

    At an anchor, prefer earlier words before it and later words after it, keeping
    the earlier anchor once. Retain unused words unless their token intervals
    conflict with a kept word. Intervals conflict when they overlap or their gap
    is less than one token frame (_ONNX_TOKEN_FRAME_S); end_s is the last token's
    time, so compare half-open intervals [start_s, end_s + _ONNX_TOKEN_FRAME_S).
    Return retained words unchanged, sorted by start time.
    """
    earlier_overlap = [
        (index, word) for index, word in enumerate(earlier) if start_s <= word.start_s <= end_s
    ]
    later_overlap = [
        (index, word) for index, word in enumerate(later) if start_s <= word.start_s <= end_s
    ]
    midpoint_s = (start_s + end_s) / 2
    anchors: list[tuple[float, float, int, int]] = []
    for earlier_index, earlier_word in earlier_overlap:
        for later_index, later_word in later_overlap:
            if earlier_word.normalized_text != later_word.normalized_text:
                continue
            if (
                later_index + 1 < len(later)
                and later[later_index + 1].start_s <= earlier_word.start_s
            ):
                continue
            if (
                earlier_word.start_s - _ONNX_ANCHOR_MAX_SHIFT_S
                <= later_word.start_s
                <= earlier_word.start_s + _ONNX_ANCHOR_MAX_SHIFT_S
            ):
                # Ignore sub-nanosecond float noise when breaking timestamp ties.
                anchors.append(
                    (
                        round(abs(earlier_word.start_s - later_word.start_s), 9),
                        round(abs(earlier_word.start_s - midpoint_s), 9),
                        earlier_index,
                        later_index,
                    )
                )
    if anchors:
        _, _, earlier_index, later_index = min(anchors)
        kept = [*earlier[: earlier_index + 1], *later[later_index + 1 :]]
        for word in [*later[:later_index], *earlier[earlier_index + 1 :]]:
            if not any(
                word.start_s < kept_word.end_s + _ONNX_TOKEN_FRAME_S
                and kept_word.start_s < word.end_s + _ONNX_TOKEN_FRAME_S
                for kept_word in kept
            ):
                kept.append(word)
        return sorted(kept, key=lambda word: word.start_s)

    complete = [word for word in earlier if word.end_s <= end_s - _ONNX_TRUNCATED_WORD_S]
    threshold_s = complete[-1].end_s + _ONNX_TOKEN_FRAME_S if complete else start_s
    return [*complete, *(word for word in later if word.start_s >= threshold_s)]


@dataclass
class _OnnxTokens:
    """Words reconciled across one input's recognition results."""

    words: list[_OnnxWord] = field(default_factory=list)
    overlap_cut_s: float | None = None
    has_logprobs: bool = True

    def extend(
        self,
        tokens: Sequence[str],
        timestamps: Sequence[float],
        logprobs: Sequence[float] | None,
        *,
        start_s: float,
        overlap_cut_s: float | None = None,
    ) -> None:
        """Splice independent chunk words while retaining their aligned token data."""
        if len(tokens) != len(timestamps):
            raise ValidationError("tokens", "tokens and timestamps must have the same length")
        if logprobs is not None and len(tokens) != len(logprobs):
            raise ValidationError("logprobs", "tokens and logprobs must have the same length")

        words: list[_OnnxWord] = []
        for index, raw_text in enumerate(tokens):
            text = str(raw_text)
            if text.startswith(" ") or not words:
                words.append(_OnnxWord(logprobs=[] if logprobs is not None else None))
            word = words[-1]
            word.tokens.append(text)
            word.timestamps.append(start_s + float(timestamps[index]))
            if word.logprobs is not None and logprobs is not None:
                word.logprobs.append(logprobs[index])

        if self.overlap_cut_s is not None:
            self.words = _onnx_splice(
                self.words, words, start_s=start_s, end_s=self.overlap_cut_s
            )
        else:
            self.words.extend(words)

        self.overlap_cut_s = overlap_cut_s
        if logprobs is None:
            self.has_logprobs = False

    def flatten(self) -> tuple[list[str], list[float], list[float] | None]:
        """Flatten reconciled words with explicit starts and aligned token data."""
        tokens: list[str] = []
        timestamps: list[float] = []
        logprobs: list[float] | None = [] if self.has_logprobs else None
        for word in self.words:
            first = word.tokens[0]
            tokens.append(first if first.startswith(" ") else " " + first)
            tokens.extend(word.tokens[1:])
            timestamps.extend(word.timestamps)
            if logprobs is not None and word.logprobs is not None:
                logprobs.extend(word.logprobs)
        return tokens, timestamps, logprobs


def _onnx_chunk_end(reader: wave.Wave_read, limit_frame: int) -> tuple[int, bool]:
    """Return a silent cut, or the chunk limit with overlap required."""
    sample_rate = reader.getframerate()
    window_frames = round(_ONNX_QUIET_WINDOW_S * sample_rate)
    search_start = limit_frame - round(_ONNX_CUT_LOOKBACK_S * sample_rate)
    reader.setpos(search_start)
    samples = np.frombuffer(reader.readframes(limit_frame - search_start), dtype="<i2")
    squared = samples.astype(np.float64)
    np.square(squared, out=squared)
    energy = np.empty(squared.size + 1, dtype=np.float64)
    energy[0] = 0.0
    np.cumsum(squared, out=energy[1:])
    rms = np.sqrt((energy[window_frames:] - energy[:-window_frames]) / window_frames)
    quietest = int(np.argmin(rms))
    silence_rms = 32_768 * 10 ** (_ONNX_SILENCE_DBFS / 20)
    if rms[quietest] < silence_rms:
        return search_start + quietest + window_frames // 2, False
    return limit_frame, True


def _onnx_chunks(wav: Path, original_index: int) -> Iterator[_OnnxChunk]:
    """Split a converted mono WAV without loading the entire input into memory."""
    with wave.open(str(wav), "rb") as reader:
        sample_rate = reader.getframerate()
        total_frames = reader.getnframes()
        max_frames = round(ONNX_MAX_CHUNK_S * sample_rate)
        overlap_frames = round(OVERLAP_S * sample_rate)
        if total_frames <= max_frames:
            yield _OnnxChunk(wav, original_index, 0.0, total_frames / sample_rate)
            return

        start_frame = 0
        chunk_index = 0
        while start_frame < total_frames:
            end_frame = min(start_frame + max_frames, total_frames)
            overlap_after = False
            if end_frame < total_frames:
                end_frame, overlap_after = _onnx_chunk_end(reader, end_frame)
            chunk_path = wav.with_name(f"{wav.stem}-chunk-{chunk_index:04d}.wav")
            reader.setpos(start_frame)
            with wave.open(str(chunk_path), "wb") as output:
                output.setparams(reader.getparams())
                output.writeframes(reader.readframes(end_frame - start_frame))
            yield _OnnxChunk(
                chunk_path,
                original_index,
                start_frame / sample_rate,
                (end_frame - start_frame) / sample_rate,
                end_frame / sample_rate if overlap_after else None,
            )
            start_frame = end_frame - overlap_frames if overlap_after else end_frame
            chunk_index += 1


def _onnx_batches(chunks: Iterator[_OnnxChunk]) -> Iterator[list[_OnnxChunk]]:
    """Group consecutive chunks within the recognition audio budget."""
    batch: list[_OnnxChunk] = []
    duration_s = 0.0
    for chunk in chunks:
        if batch and duration_s + chunk.duration_s > ONNX_BATCH_SECONDS:
            yield batch
            batch = []
            duration_s = 0.0
        batch.append(chunk)
        duration_s += chunk.duration_s
    if batch:
        yield batch


class ParakeetOnnxBackend:
    """Portable CPU Parakeet backend using onnx-asr."""

    name = "parakeet-onnx"
    child_command: Sequence[str] | None = None

    def __init__(
        self,
        model: str = "nemo-parakeet-tdt-0.6b-v3",
        quantization: str | None = "int8",
        *,
        in_process: bool = False,
    ) -> None:
        self.model = model
        self.quantization = quantization
        self.in_process = in_process
        self.last_run: ChildRun | None = None

    @staticmethod
    def available() -> bool:
        """Return whether onnx-asr is importable."""
        return importlib.util.find_spec("onnx_asr") is not None

    def version(self) -> str | None:
        return _distribution_version("onnx-asr")

    def download_notice(self) -> str | None:
        repo_id, patterns, required = self._model_files()
        return _model_download_notice(
            repo_id, "about 670 MB", allow_patterns=patterns, required_files=required,
        )

    def _model_files(self) -> tuple[str, list[str], list[str]]:
        suffix = f".{self.quantization}" if self.quantization else ""
        onnx_files = [f"encoder-model{suffix}.onnx", f"decoder_joint-model{suffix}.onnx"]
        return (
            "istupakov/parakeet-tdt-0.6b-v3-onnx",
            [
                "config.json", "config.yaml", *onnx_files, "vocab.txt",
                *(f"{name}?data" for name in onnx_files),
            ],
            ["config.json", "vocab.txt", *onnx_files],
        )

    def prefetch(self) -> None:
        repo_id, patterns, required = self._model_files()
        _prefetch_model(repo_id, patterns, required)

    def transcribe_many(
        self,
        paths: Sequence[Path],
        *,
        language: str | None = None,
    ) -> dict[Path, Transcript]:
        """Isolate native recognition and accept completed results after shutdown crashes."""
        if self.in_process:
            return self.transcribe_in_process(paths, language=language)
        if not paths:
            return {}
        from .engine_child import run_in_child

        run = run_in_child(
            self.name,
            self.model,
            paths,
            options={"quantization": self.quantization},
            language=language,
            command=self.child_command,
        )
        self.last_run = run
        if run.transcripts is not None:
            if run.returncode:
                logging.getLogger("yap_studio").debug(
                    "%s child exited with status %s after writing a complete result",
                    self.name,
                    run.returncode,
                )
            return {path: run.transcripts[path] for path in paths}
        raise _processing_error(
            f"onnx-asr {self.model}",
            run.returncode if run.returncode is not None else -1,
            f"{run.stderr_tail}\n{run.reason}".strip(),
        )

    def transcribe_in_process(
        self,
        paths: Sequence[Path],
        *,
        language: str | None = None,
    ) -> dict[Path, Transcript]:
        """Load and run the native engine within the current interpreter."""
        del language
        if not paths:
            return {}
        with _converted_wavs(paths) as (originals, converted, durations):
            command = f"onnx-asr {self.model}"
            try:
                import onnx_asr

                engine = onnx_asr.load_model(
                    self.model,
                    quantization=self.quantization,
                    providers=["CPUExecutionProvider"],
                ).with_timestamps()
                stitched = [_OnnxTokens() for _ in originals]
                chunks = (
                    chunk
                    for index, wav in enumerate(converted)
                    for chunk in _onnx_chunks(wav, index)
                )
                for batch in _onnx_batches(chunks):
                    results = engine.recognize([str(chunk.path) for chunk in batch])
                    if len(batch) == 1 and hasattr(results, "tokens"):
                        results = [results]
                    if len(results) != len(batch):
                        raise _processing_error(
                            command, 1, "speech engine returned the wrong result count"
                        )
                    for chunk, result in zip(batch, results, strict=True):
                        stitched[chunk.original_index].extend(
                            result.tokens,
                            result.timestamps,
                            getattr(result, "logprobs", None),
                            start_s=chunk.start_s,
                            overlap_cut_s=chunk.overlap_cut_s,
                        )
            except ProcessingError:
                raise
            except Exception as exc:
                raise _processing_error(command, 1, str(exc)) from exc

            transcripts: dict[Path, Transcript] = {}
            for original, result, duration in zip(originals, stitched, durations, strict=True):
                try:
                    tokens, timestamps, logprobs = result.flatten()
                    transcript = from_parakeet_tokens(
                        tokens,
                        timestamps,
                        logprobs=logprobs,
                        duration_s=duration,
                        backend=self.name,
                        model=self.model,
                    )
                except (ValueError, TypeError, AttributeError) as exc:
                    raise _processing_error(command, 1, str(exc)) from exc
                transcripts[original] = transcript
            return transcripts


class FasterWhisperBackend:
    """Optional fallback speech backend."""

    name = "faster-whisper"

    def __init__(self, model: str = "small.en", compute_type: str = "int8") -> None:
        self.model = model
        self.compute_type = compute_type

    @staticmethod
    def available() -> bool:
        """Return whether faster-whisper is importable."""
        return importlib.util.find_spec("faster_whisper") is not None

    def version(self) -> str | None:
        return _distribution_version("faster-whisper")

    def download_notice(self) -> str | None:
        notice = "First run downloads the speech model."
        try:
            model_path = Path(self.model)
            if not model_path.is_dir():
                from faster_whisper import download_model

                model_path = Path(download_model(self.model, local_files_only=True))
            return None if _model_files_ready(model_path, ("config.json", "model.bin")) else notice
        except Exception:
            return notice

    def prefetch(self) -> None:
        if Path(self.model).is_dir():
            return
        from faster_whisper import download_model

        download_model(self.model)

    def transcribe_many(
        self,
        paths: Sequence[Path],
        *,
        language: str | None = None,
    ) -> dict[Path, Transcript]:
        if not paths:
            return {}
        with _converted_wavs(paths) as (originals, converted, durations):
            command = f"faster-whisper {self.model}"
            try:
                from faster_whisper import WhisperModel

                engine = WhisperModel(self.model, compute_type=self.compute_type)
            except Exception as exc:
                raise _processing_error(command, 1, str(exc)) from exc

            transcripts: dict[Path, Transcript] = {}
            for original, wav, duration in zip(originals, converted, durations, strict=True):
                try:
                    segments, info = engine.transcribe(
                        str(wav),
                        language=language,
                        word_timestamps=True,
                        vad_filter=False,
                        condition_on_previous_text=False,
                    )
                    words: list[Word] = []
                    for segment in segments:
                        for item in segment.words or []:
                            if item.start is None or item.end is None:
                                continue
                            words.append(
                                Word(
                                    text=str(item.word).strip(),
                                    start=float(item.start),
                                    end=float(item.end),
                                    conf=(
                                        float(item.probability)
                                        if item.probability is not None
                                        else None
                                    ),
                                )
                            )
                    detected_language = language or getattr(info, "language", None)
                    transcripts[original] = Transcript(
                        words=words,
                        language=detected_language,
                        backend=self.name,
                        model=self.model,
                        duration_s=duration,
                    )
                except Exception as exc:
                    raise _processing_error(command, 1, str(exc)) from exc
            return transcripts


class ScriptedBackend:
    """Deterministic backend for tests and callers with prepared transcripts."""

    def __init__(
        self,
        fn: Callable[[Path], Transcript],
        *,
        name: str = "scripted",
        model: str = "scripted",
    ) -> None:
        self.fn = fn
        self.name = name
        self.model = model

    def version(self) -> str | None:
        return "0"

    def download_notice(self) -> str | None:
        return None

    def prefetch(self) -> None:
        pass

    def transcribe_many(
        self,
        paths: Sequence[Path],
        *,
        language: str | None = None,
    ) -> dict[Path, Transcript]:
        del language
        originals = [Path(path) for path in paths]
        return {path: self.fn(path) for path in originals}


def available_engines() -> dict[str, bool]:
    """Return availability for every supported speech engine."""
    return {
        "parakeet-mlx": ParakeetMlxBackend.available(),
        "parakeet-onnx": ParakeetOnnxBackend.available(),
        "faster-whisper": FasterWhisperBackend.available(),
    }


def _onnx_backend(model: str | None) -> ParakeetOnnxBackend:
    selected = model or "nemo-parakeet-tdt-0.6b-v3"
    quantization: str | None = "int8"
    if selected.endswith(":full"):
        selected = selected.removesuffix(":full")
        quantization = None
    elif selected.endswith(":int8"):
        selected = selected.removesuffix(":int8")
    return ParakeetOnnxBackend(model=selected, quantization=quantization)


def resolve_backend(name: str = "auto", model: str | None = None) -> Backend | None:
    """Resolve a requested speech engine or the best available default."""
    if name == "none":
        return None
    constructors: dict[str, Callable[[], Backend]] = {
        "parakeet-mlx": lambda: ParakeetMlxBackend(
            model=model or "mlx-community/parakeet-tdt-0.6b-v3"
        ),
        "parakeet-onnx": lambda: _onnx_backend(model),
        "faster-whisper": lambda: FasterWhisperBackend(model=model or "small.en"),
    }
    availability = available_engines()
    if name == "auto":
        for engine_name in ENGINES:
            if availability[engine_name]:
                return constructors[engine_name]()
        return None
    if name not in constructors:
        raise ValidationError("asr", f"must be one of auto, none, {', '.join(ENGINES)}")
    if not availability[name]:
        hint = "pip install 'yap-studio[whisper]'" if name == "faster-whisper" else None
        if hint is None:
            raise DependencyError.missing_engine(name)
        raise DependencyError.missing_engine(name, hint=hint)
    return constructors[name]()


def resolve_second_backend(first: Backend | None) -> Backend | None:
    """Isolate an available engine from another family, leaving scripted engines alone."""
    if first is None or first.name not in ENGINES:
        return None
    from .isolated import IsolatedBackend

    availability = available_engines()
    family = ENGINE_FAMILIES[first.name]
    preferred = ("faster-whisper",) if family == "parakeet" else ()
    for name in dict.fromkeys((*preferred, *ENGINES)):
        if ENGINE_FAMILIES[name] != family and availability[name]:
            inner = resolve_backend(name)
            if inner is not None:
                return IsolatedBackend(inner)
    return None


def second_engine_unmeasured_reason(first_name: str) -> str:
    """Explain the missing independent engine using installed engines and an install hint."""
    family = ENGINE_FAMILIES.get(first_name)
    availability = available_engines()
    same_family = [
        name for name in ENGINES
        if name != first_name and availability[name] and ENGINE_FAMILIES[name] == family
    ]
    extra = "asr" if family == "whisper" else "whisper"
    hint = f"for example pip install 'yap-studio[{extra}]'"
    if same_family:
        names = ", ".join(same_family)
        verb = "is" if len(same_family) == 1 else "are"
        return (
            "unmeasured: no speech engine from another model family is installed "
            f"({names} {verb} the same model family as {first_name}); install one, {hint}"
        )
    return (
        f"unmeasured: only {first_name} is installed; install a speech engine "
        f"from another model family, {hint}"
    )
