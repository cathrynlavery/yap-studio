"""Raw speech results are shared across takes and their consumers."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from yap_studio import cache, ffmpeg, transcript
from yap_studio.errors import InputError
from yap_studio.transcript import ScriptedBackend, Transcript, Word, shared


class CountingBackend:
    name = "parakeet-mlx"
    model = "test-model"
    engine_version: str | None = "test-version"
    notice: str | None = None

    def __init__(self) -> None:
        self.calls: list[list[Path]] = []
        self.durations: list[float] = []
        self.words = [Word(token, 0.2 + i * 0.3, 0.4 + i * 0.3)
                      for i, token in enumerate("These words explain the useful idea.".split())]

    def version(self):
        return self.engine_version

    def download_notice(self):
        return self.notice

    def transcribe_many(self, paths, *, language=None):
        self.calls.append(list(paths))
        results = {}
        for path in paths:
            info = ffmpeg.probe(path, count_frames=False)
            duration = info.audio_duration_s or info.duration_s
            self.durations.append(duration)
            results[path] = Transcript(list(self.words), "en", self.name, self.model, duration)
        return results


@pytest.fixture
def take_request(tmp_path):
    media = tmp_path / "take.wav"
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "anullsrc=r=16000:cl=mono", "-t", "1", str(media)], timeout_s=30)
    return shared.Request(shared.take_digest(media), 0, media)


def test_hit_refresh_and_raw_result(take_request):
    backend = CountingBackend()
    first, hit = shared.transcribe(backend, [take_request])[take_request.media]
    assert not hit
    first.words.clear()
    second, hit = shared.transcribe(backend, [take_request], language="en")[take_request.media]
    assert hit and second.words and len(backend.calls) == 1
    backend.words = [Word("refreshed", 0, 0.5)]
    fresh, hit = shared.transcribe(backend, [take_request], refresh=True)[take_request.media]
    assert not hit and len(backend.calls) == 2
    assert shared.lookup(backend, take_request) == fresh


@pytest.mark.parametrize("change", ["unchanged", "modify", "replace", "delete"])
@pytest.mark.parametrize("when", ["before", "during"])
def test_transcribe_only_publishes_for_the_hashed_file(take_request, monkeypatch, change, when):
    backend = CountingBackend()
    media = take_request.media

    def change_file():
        if change == "modify":
            media.write_bytes(media.read_bytes() + b"changed")
        elif change == "replace":
            replacement = media.with_name("replacement.wav")
            replacement.write_bytes(media.read_bytes())
            replacement.replace(media)
        elif change == "delete":
            media.unlink()

    def transcribe(paths, *, language=None):
        backend.calls.append(list(paths))
        if when == "during":
            change_file()
        return {path: Transcript(list(backend.words)) for path in paths}

    monkeypatch.setattr(backend, "transcribe_many", transcribe)
    if when == "before":
        change_file()
    if change == "unchanged":
        result, hit = shared.transcribe(backend, [take_request])[media]
        assert not hit and result.words == backend.words
        assert shared.lookup(backend, take_request) == result
    else:
        with pytest.raises(InputError, match="changed|disappeared"):
            shared.transcribe(backend, [take_request])
        assert shared.lookup(backend, take_request) is None
        assert not list(cache.cache_dir("transcripts", take_request.take_sha256).glob("*.json"))
        assert len(backend.calls) == (when == "during")


@pytest.mark.parametrize("decoded", [False, True])
def test_changed_engine_input_aborts_publication(take_request, monkeypatch, decoded):
    backend = CountingBackend()
    other = take_request.media.with_name("copy.wav")
    other.write_bytes(take_request.media.read_bytes())
    if decoded:
        requests = [shared.Request(
            take_request.take_sha256, 0, other, source=take_request.media,
            source_entry={**shared._identity(take_request.media),
                          "sha256": take_request.take_sha256},
        )]
    else:
        requests = [take_request, shared.Request(take_request.take_sha256, 0, other)]

    def transcribe(paths, *, language=None):
        other.write_bytes(other.read_bytes() + b"changed")
        return {path: Transcript(list(backend.words)) for path in paths}

    monkeypatch.setattr(backend, "transcribe_many", transcribe)
    with pytest.raises(InputError, match="changed"):
        shared.transcribe(backend, requests)
    assert shared.lookup(backend, take_request) is None


@pytest.mark.parametrize("field,value", [
    ("name", "parakeet-onnx"), ("model", "other"), ("engine_version", "other"),
    ("engine_version", None),
])
def test_engine_key_fields(take_request, field, value):
    backend = CountingBackend()
    shared.transcribe(backend, [take_request])
    setattr(backend, field, value)
    assert shared.lookup(backend, take_request) is None
    shared.transcribe(backend, [take_request])
    assert len(backend.calls) == 2


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_model_key_distinguishes_relative_paths(take_request, tmp_path, monkeypatch, kind):
    backend = CountingBackend()
    backend.name = "faster-whisper"
    backend.model = "./model"
    keys = []
    specs = []
    models = []
    for name in ("first", "second"):
        folder = tmp_path / name
        folder.mkdir()
        model = folder / "model"
        if kind == "directory":
            model.mkdir()
            (model / "weights.bin").write_bytes(name.encode())
        else:
            model.write_bytes(name.encode())
        models.append(str(model.resolve()))
        monkeypatch.chdir(folder)
        keys.append(shared._key(backend, take_request, None))
        specs.append(shared.model_spec(backend))
    assert keys[0] != keys[1]
    assert [key["model"] for key in keys] == models
    assert specs == models


@pytest.mark.parametrize("model", ["org/remote-model", "tiny.en"])
def test_model_key_preserves_remote_ids(take_request, tmp_path, monkeypatch, model):
    backend = CountingBackend()
    backend.name = "faster-whisper"
    backend.model = model
    monkeypatch.chdir(tmp_path)
    assert shared._key(backend, take_request, None)["model"] == model
    assert shared.model_spec(backend) == model


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_model_key_matches_relative_and_absolute_paths(take_request, tmp_path, monkeypatch, kind):
    backend = CountingBackend()
    backend.name = "faster-whisper"
    model = tmp_path / "model"
    if kind == "directory":
        model.mkdir()
    else:
        model.write_bytes(b"weights")
    monkeypatch.chdir(tmp_path)
    backend.model = "./model"
    relative = shared._key(backend, take_request, None)
    relative_spec = shared.model_spec(backend)
    backend.model = str(model.resolve())
    assert shared._key(backend, take_request, None) == relative
    assert relative["model"] == backend.model
    assert shared.model_spec(backend) == relative_spec == backend.model


@pytest.mark.parametrize("engine,option,first,second", [
    ("parakeet-onnx", "quantization", None, "int8"),
    ("parakeet-onnx", "quantization", "int8", None),
    ("faster-whisper", "compute_type", "int8", "float32"),
])
def test_engine_options_have_separate_entries(take_request, engine, option, first, second):
    backend = CountingBackend()
    backend.name = engine
    setattr(backend, option, first)
    shared.transcribe(backend, [take_request])
    other = CountingBackend()
    other.name = engine
    setattr(other, option, second)
    assert shared.lookup(other, take_request) is None
    result, hit = shared.transcribe(other, [take_request])[take_request.media]
    assert not hit and result.words == other.words
    assert len(backend.calls) == len(other.calls) == 1
    assert shared.lookup(backend, take_request) is not None
    entries = list(cache.cache_dir("transcripts", take_request.take_sha256).glob("*.json"))
    assert len(entries) == 2


@pytest.mark.parametrize("engine,option", [
    ("parakeet-onnx", "quantization"), ("faster-whisper", "compute_type"),
])
def test_absent_engine_option_is_none(take_request, engine, option):
    backend = CountingBackend()
    backend.name = engine
    shared.transcribe(backend, [take_request])
    [entry] = cache.cache_dir("transcripts", take_request.take_sha256).glob("*.json")
    assert json.loads(entry.read_text())[option] is None
    setattr(backend, option, None)
    assert shared.lookup(backend, take_request) is not None


def test_stream_language_and_batch(take_request):
    backend = CountingBackend()
    shared.transcribe(backend, [take_request])
    assert shared.lookup(backend, take_request, language="fr") is not None
    other_stream = shared.Request(take_request.take_sha256, 1, take_request.media)
    assert shared.lookup(backend, other_stream) is None
    backend.name = "faster-whisper"
    shared.transcribe(backend, [take_request], language="en")
    assert shared.lookup(backend, take_request, language="fr") is None
    other = take_request.media.with_name("other.wav")
    other.write_bytes(take_request.media.read_bytes() + b"other")
    shared.transcribe(backend, [take_request, shared.Request(shared.take_digest(other), 0, other)],
                      language="fr")
    assert backend.calls[-1] == [take_request.media, other]


@pytest.mark.parametrize("detected", ["en", "fr", None])
def test_whisper_reuses_auto_detected_language_only_when_it_matches(take_request, detected):
    backend = CountingBackend()
    backend.name = "faster-whisper"
    shared.store(backend, take_request, Transcript(backend.words, language=detected))
    hit = shared.lookup(backend, take_request, language="en")
    assert (hit is not None) is (detected == "en")
    result, cached = shared.transcribe(backend, [take_request], language="en")[take_request.media]
    assert cached is (detected == "en")
    assert result.language == "en"
    assert len(backend.calls) == (0 if detected == "en" else 1)
    assert shared.lookup(backend, take_request).language == detected
    if detected != "en":
        entries = cache.cache_dir("transcripts", take_request.take_sha256).glob("*.json")
        assert {json.loads(entry.read_text())["language"] for entry in entries} == {None, "en"}


@pytest.mark.parametrize("change", ["corrupt", "schema", "take_sha256", "engine", "model",
                                    "version", "audio_stream", "stream_type", "language",
                                    "quantization", "compute_type", "transcript"])
def test_bad_entries_miss(take_request, change):
    backend = CountingBackend()
    shared.transcribe(backend, [take_request])
    [entry] = cache.cache_dir("transcripts", take_request.take_sha256).glob("*.json")
    data = json.loads(entry.read_text())
    if change == "corrupt":
        entry.write_text("{")
    else:
        if change == "stream_type":
            data["audio_stream"] = False
        else:
            data[change] = "wrong"
        entry.write_text(json.dumps(data))
    assert shared.lookup(backend, take_request) is None


@pytest.mark.parametrize("name", ["scripted", "other"])
def test_uncached_backends(take_request, name):
    calls = []
    backend = ScriptedBackend(lambda path: calls.append(path) or Transcript([]), name=name)
    for _ in range(2):
        assert shared.transcribe(backend, [take_request])[take_request.media][1] is False
    assert len(calls) == 2
    assert not list(cache.cache_dir("transcripts", take_request.take_sha256).glob("*.json"))


@pytest.mark.parametrize("digest", ["../bad", "A" * 64, "a" * 63, "g" * 64])
def test_invalid_digest(take_request, digest):
    with pytest.raises(ValueError, match="sha256"):
        shared.transcribe(CountingBackend(), [shared.Request(digest, 0, take_request.media)])


def test_store_failure_does_not_lose_engine_output(take_request, monkeypatch):
    backend = CountingBackend()

    def fail(*_args):
        raise OSError("synthetic cache failure")

    monkeypatch.setattr(shared.os, "replace", fail)
    result, hit = shared.transcribe(backend, [take_request])[take_request.media]
    assert result.words == backend.words and not hit
    assert shared.lookup(backend, take_request) is None


def test_digest_retries_changes_and_rejects_unstable_file(tmp_path, monkeypatch):
    from yap_studio import receipt
    from yap_studio.errors import InputError

    take = tmp_path / "take"
    take.write_bytes(b"first")
    original = receipt.sha256_file
    calls = []

    def changing(path):
        digest = original(path)
        calls.append(path)
        if len(calls) == 1:
            path.write_bytes(b"second")
        return digest

    monkeypatch.setattr(receipt, "sha256_file", changing)
    assert shared.take_digest(take) == original(take)
    assert len(calls) == 2
    take.write_bytes(b"third")

    def unstable(path):
        digest = original(path)
        path.write_bytes(path.read_bytes() + b"x")
        return digest

    monkeypatch.setattr(receipt, "sha256_file", unstable)
    with pytest.raises(InputError, match="kept changing"):
        shared.take_digest(take)


def test_equal_content_in_one_batch_runs_once(take_request):
    backend = CountingBackend()
    other = take_request.media.with_name("copy.wav")
    other.write_bytes(take_request.media.read_bytes())
    duplicate = shared.Request(take_request.take_sha256, 0, other)
    result = shared.transcribe(backend, [take_request, duplicate])
    assert backend.calls == [[take_request.media]]
    result[take_request.media][0].words.clear()
    assert result[other][0].words


def test_concurrent_callers_transcribe_once(take_request):
    backend = CountingBackend()
    entered, release, second_started = (threading.Event() for _ in range(3))
    original = backend.transcribe_many

    def blocking(paths, *, language=None):
        entered.set()
        assert release.wait(10)
        return original(paths, language=language)

    def second():
        second_started.set()
        return shared.transcribe(backend, [take_request])

    backend.transcribe_many = blocking
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(shared.transcribe, backend, [take_request])
        assert entered.wait(10)
        waiting = pool.submit(second)
        assert second_started.wait(10)
        assert not waiting.done()
        release.set()
        assert first.result()[take_request.media][1] is False
        assert waiting.result()[take_request.media][1] is True
    assert len(backend.calls) == 1


def test_known_digest_never_hashes(tmp_path, monkeypatch):
    from yap_studio import receipt

    take = tmp_path / "take"
    take.write_bytes(b"first")
    assert shared.known_digest(take) is None
    digest = shared.take_digest(take)
    monkeypatch.setattr(receipt, "sha256_file", lambda _: pytest.fail("unexpected hash"))
    assert shared.known_digest(take) == digest
    assert shared.take_digest(take) == digest
    stat = take.stat()
    shared.os.utime(take, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert shared.known_digest(take) is None
    take.write_bytes(b"second")
    assert shared.known_digest(take) is None


@pytest.mark.parametrize("changed", [False, True])
def test_remember_digest_uses_the_recorded_identity(tmp_path, monkeypatch, changed):
    from yap_studio import receipt
    from yap_studio.project.receipts import _identity

    take = tmp_path / "take"
    take.write_bytes(b"first")
    identity = _identity(take)
    digest = receipt.sha256_file(take)
    if changed:
        take.write_bytes(b"second")
    with monkeypatch.context() as recording:
        recording.setattr(shared, "_identity", lambda _: pytest.fail("unexpected fresh identity"))
        shared.remember_digest(take, digest, identity)
    assert shared.known_digest(take) == (None if changed else digest)


@pytest.mark.parametrize("seconds", [3.1, 43.0])
def test_one_full_transcription_per_take(tmp_path, monkeypatch, seconds):
    from test_ingest import make_media, make_project

    from yap_studio.gallery import still
    from yap_studio.stages.ingest import ingest
    from yap_studio.transcript.take_job import run
    from yap_studio.watch import build_pack

    take = make_media(tmp_path / "take.mkv", seconds=seconds, sound="silence", fps=2)
    backend = CountingBackend()
    monkeypatch.setattr(transcript, "resolve_backend", lambda *_: backend)
    monkeypatch.setattr(still, "pick_still", lambda *_args, **_kwargs: type("Still", (), {
        "path": still.sample_still(), "time_s": 1.0, "scores": {"faces": []}, "reasons": (),
    })())
    preview = still.prepare_take(take, tmp_path / "preview", backend=backend)
    assert preview["transcript"] == ("take" if seconds < 40 else "opening")
    if seconds > 40:
        run(take, backend)
    root = make_project(tmp_path / "project", [take])
    ingest(root)
    build_pack(take, out=tmp_path / "pack", backend=backend)
    assert sum(duration > seconds - 0.1 for duration in backend.durations) == 1
    assert sum(39.9 <= duration <= 40.1 for duration in backend.durations) == (seconds > 40)
    assert len(backend.durations) == (2 if seconds > 40 else 1)  # Silent media needs no clips.
