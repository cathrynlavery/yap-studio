"""Source-clock ingest, recording analysis, and safe word replacement."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import numpy as np
import pytest
import synth

from yap_studio import ffmpeg, paths, receipt, transcript
from yap_studio.cli import main
from yap_studio.errors import DependencyError, InputError, OutputPathError, ProcessingError
from yap_studio.project.lock import ConflictError
from yap_studio.project.model import load_words
from yap_studio.project.receipts import AudioStreamMissingError, audio_stream_sha256
from yap_studio.project.status import ProjectContext, ingest_inputs, status
from yap_studio.stages.ingest import (
    IngestRefusal,
    _float_wav,
    _source_words,
    ingest,
    preflight,
    words_listing,
)
from yap_studio.transcript import Transcript, Word
from yap_studio.transcript.backends import ParakeetMlxBackend

FIXTURE_WORDS = [
    Word("Here's", 0.2, 0.5, 0.9),
    Word("the", 0.55, 0.7, 0.9),
    Word("thing", 0.75, 1.0, 0.9),
    Word("nobody", 1.1, 1.4, 0.9),
    Word("tells", 1.5, 1.7, 0.9),
    Word("you.", 1.8, 2.0, 0.9),
]
SENTENCES = [
    "Bright planets circle slowly over quiet hills on calm winter evenings",
    "Seven green boxes wait beside the open door near our garden",
    "Morning light fills the empty kitchen before breakfast begins each day",
]


class FakeBackend:
    """Return fixture words on each analysis WAV's sample clock."""

    name = "parakeet-mlx"
    model = "fixture"

    def __init__(self) -> None:
        self.words = list(FIXTURE_WORDS)
        self.calls: list[list[Path]] = []
        self.languages: list[str | None] = []
        self.on_transcribe: Callable[[], None] | None = None

    def version(self) -> str:
        return "test-version"

    def transcribe_many(
        self,
        files: Sequence[Path],
        *,
        language: str | None = None,
    ) -> dict[Path, Transcript]:
        """Check the WAV contract and return one result for every input."""
        self.calls.append(list(files))
        self.languages.append(language)
        results = {}
        for wav in files:
            info = ffmpeg.probe(wav, count_frames=False)
            assert info.sample_rate == 16_000
            assert info.channels == 1
            assert info.audio_codec == "pcm_f32le"
            assert info.audio_offset_s == 0
            results[wav] = Transcript(
                list(self.words), language, self.name, self.model, info.duration_s
            )
        if self.on_transcribe:
            self.on_transcribe()
        return results


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    """Install a deterministic speech substitute without loading an engine."""
    fake = FakeBackend()
    monkeypatch.setattr(transcript, "resolve_backend", lambda _name="auto", _model=None: fake)
    return fake


def write_json(target: Path, data: dict[str, Any]) -> None:
    """Write a project fixture atomically."""
    paths.atomic_write_text(target, json.dumps(data) + "\n")


def make_project(root: Path, takes: list[Path], *, stream: int | None = None) -> Path:
    """Create the smallest valid project around existing read-only sources."""
    write_json(
        root / "project.json",
        {
            "schema": "yap-studio/project@1",
            "name": "fixture",
            "takes": [
                {
                    "id": f"t{index + 1}",
                    "path": str(take.relative_to(root)) if take.is_relative_to(root) else str(take),
                    **({"audio_stream": stream} if stream is not None else {}),
                }
                for index, take in enumerate(takes)
            ],
        },
    )
    return root


def make_media(
    target: Path,
    *,
    seconds: float = 10,
    sound: str = "voice",
    fps: int = 30,
    transfer: str | None = None,
    offset_s: float = 0,
    second_stream: bool = False,
) -> Path:
    """Build media whose level, colour, rates, and clocks are fully controlled."""
    sounds = {
        "voice": (
            f"anoisesrc=color=white:sample_rate=16000:amplitude=0.1:seed=13:duration={seconds}"
        ),
        "tone": f"aevalsrc=0.08*sin(2*PI*440*t)+0.0005*random(1):s=16000:d={seconds}",
        "silence": f"anullsrc=r=16000:cl=mono:d={seconds}",
    }
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=gray:s=64x64:r={fps}:d={seconds}",
    ]
    if sound != "none":
        command.extend(["-itsoffset", str(offset_s), "-f", "lavfi", "-i", sounds[sound]])
    if second_stream:
        command.extend(["-itsoffset", "0.12", "-f", "lavfi", "-i", sounds["tone"]])
    command.extend(["-map", "0:v:0"])
    if sound != "none":
        command.extend(["-map", "1:a:0"])
    if second_stream:
        command.extend(["-map", "2:a:0"])
    command.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le"])
    if transfer:
        command.extend(
            ["-vf", f"setparams=color_trc={transfer}:color_primaries=bt2020:colorspace=bt2020nc"]
        )
    with paths.atomic_output(target) as temporary:
        ffmpeg.run([*command, "-f", "matroska", str(temporary)], timeout_s=120 + seconds * 1.5)
    return target


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A take inside the project, ready for analysis."""
    return make_project(tmp_path, [make_media(tmp_path / "take.mkv")])


def add_references(root: Path, *, time_only: bool = False) -> None:
    """Anchor an edit and both lanes to words that can later change."""
    hashes = {"t1": receipt.sha256_file(root / "ingest/t1/words.json")}
    write_json(
        root / "edit.json",
        {
            "schema": "yap-studio/edit@1",
            "words_sha256": hashes,
            "segments": [
                {
                    "id": "s1",
                    "take": "t1",
                    "from": {"time": 0} if time_only else {"word": "t1:0"},
                    "to": {"time": 2} if time_only else {"word": "t1:5"},
                },
            ],
        },
    )
    for lane in ("agent", "broll"):
        write_json(
            root / f"timeline/{lane}.json",
            {
                "schema": "yap-studio/timeline@1",
                "words_sha256": hashes,
                "elements": [
                    {
                        "id": f"e-{lane}",
                        "kind": "caption_gap",
                        "start": {"word": "t1:0"},
                        "end": {"word": "t1:5"},
                    },
                ],
            },
        )


@pytest.mark.parametrize(
    ("suffix", "options", "same_audio"),
    [
        (".mkv", ["-vf", "scale=32:32", "-c:v", "libx264", "-c:a", "copy"], True),
        (".mov", ["-c", "copy"], True),
        (".mp4", ["-c", "copy"], True),
        (".mkv", ["-c:v", "copy", "-c:a", "aac", "-b:a", "64k"], False),
    ],
    ids=["picture-reencode", "container-remux", "container-remux-mp4", "audio-reencode"],
)
def test_audio_stream_hash_tracks_encoded_audio(
    tmp_path: Path, suffix: str, options: list[str], same_audio: bool,
) -> None:
    take = make_media(tmp_path / "take.mkv", seconds=2)
    copied = tmp_path / f"copy{suffix}"
    ffmpeg.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(take),
            "-map", "0:v:0", "-map", "0:a:0", *options, str(copied),
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    assert receipt.sha256_file(take) != receipt.sha256_file(copied)
    assert (audio_stream_sha256(take, 0) == audio_stream_sha256(copied, 0)) is same_audio


@pytest.mark.parametrize("suffix", [".mov", ".mp4"])
def test_audio_stream_hash_tolerates_container_timing_rounding(tmp_path: Path, suffix: str) -> None:
    take = tmp_path / "take.mkv"
    # AAC at 44.1 kHz has packet durations that cannot be exact Matroska milliseconds.
    ffmpeg.run(
        [
            "ffmpeg", "-nostdin", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=10:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=1",
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-c:a", "aac", str(take),
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    remuxed = tmp_path / f"copy{suffix}"
    ffmpeg.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(take),
            "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", str(remuxed),
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    assert audio_stream_sha256(take, 0) == audio_stream_sha256(remuxed, 0)


def test_audio_stream_hash_tracks_container_offset(tmp_path: Path) -> None:
    take = make_media(tmp_path / "take.mkv", seconds=2)
    shifted = tmp_path / "shifted.mkv"
    ffmpeg.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(take),
            "-itsoffset", "0.25", "-i", str(take),
            "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", str(shifted),
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    packets = []
    for media in (take, shifted):
        result = ffmpeg.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-i", str(media),
                "-map", "0:a:0", "-c", "copy", "-f", "streamhash", "-hash", "sha256", "-",
            ],
            timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
        )
        packets.append(result.stdout)
    assert packets[0] == packets[1]
    assert ffmpeg.probe(shifted, count_frames=False).audio_offset_s == pytest.approx(0.25)
    assert audio_stream_sha256(take, 0) != audio_stream_sha256(shifted, 0)


@pytest.mark.parametrize("gap_s", [0.5, -0.016], ids=["gap", "overlap"])
def test_audio_stream_hash_tracks_later_packet_timing(tmp_path: Path, gap_s: float) -> None:
    take = make_media(tmp_path / "take.mkv", seconds=2)
    shifted = tmp_path / "shifted.mkv"
    # Change timestamps after packet 16 without decoding or changing any packet bytes.
    shift = rf"gte(N\,16)*{gap_s}/TB"
    ffmpeg.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(take),
            "-map", "0:v:0", "-map", "0:a:0", "-c", "copy",
            "-bsf:a", f"setts=pts=PTS+{shift}:dts=DTS+{shift}", str(shifted),
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    hashes = []
    starts = []
    for media in (take, shifted):
        result = ffmpeg.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-i", str(media),
                "-map", "0:a:0", "-c", "copy", "-f", "framehash", "-hash", "sha256", "-",
            ],
            timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
        )
        lines = result.stdout.splitlines()
        time_base = Fraction(next(line.split(":", 1)[1] for line in lines
                                  if line.startswith("#tb 0:")))
        packets = [line.split(",") for line in lines if line and not line.startswith("#")]
        assert len(packets) > 16
        hashes.append([packet[5].strip() for packet in packets])
        starts.append([int(packet[2]) * time_base for packet in packets])
    assert hashes[0] == hashes[1]
    assert starts[0][:16] == starts[1][:16]
    assert starts[1][16] - starts[0][16] == Fraction(str(gap_s))
    original = ffmpeg.probe(take, count_frames=False)
    replacement = ffmpeg.probe(shifted, count_frames=False)
    assert original.format_start_s + original.audio_offset_s == (
        replacement.format_start_s + replacement.audio_offset_s
    )
    assert audio_stream_sha256(take, 0) != audio_stream_sha256(shifted, 0)


def test_audio_stream_hash_ignores_ffmpeg_software_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    take = make_media(tmp_path / "take.mkv", seconds=1)
    original_hash = audio_stream_sha256(take, 0)
    run = ffmpeg.run
    calls = []

    def different_software(cmd: list[str], *, timeout_s: float) -> CompletedProcess[str]:
        result = run(cmd, timeout_s=timeout_s)
        if "framehash" in cmd:
            assert "#software:" in result.stdout
            result.stdout = "\n".join(
                "#software: another ffmpeg version" if line.startswith("#software:") else line
                for line in result.stdout.splitlines()
            )
            calls.append(cmd)
        return result

    monkeypatch.setattr(ffmpeg, "run", different_software)
    assert audio_stream_sha256(take, 0) == original_hash
    assert len(calls) == 1


@pytest.mark.parametrize("ordinal", [-1, 1])
def test_audio_stream_hash_refuses_missing_stream(tmp_path: Path, ordinal: int) -> None:
    take = make_media(tmp_path / "take.mkv", seconds=1)
    with pytest.raises(
        AudioStreamMissingError, match=f"audio stream {ordinal} is missing; choose another stream",
    ):
        audio_stream_sha256(take, ordinal)


def test_audio_stream_hash_refuses_no_packets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    take = make_media(tmp_path / "take.mkv", seconds=1)
    run = ffmpeg.run

    def no_packets(cmd: list[str], *, timeout_s: float) -> CompletedProcess[str]:
        result = run(cmd, timeout_s=timeout_s)
        if "framehash" in cmd:
            result.stdout = "\n".join(line for line in result.stdout.splitlines()
                                      if line.startswith("#"))
        return result

    monkeypatch.setattr(ffmpeg, "run", no_packets)
    with pytest.raises(AudioStreamMissingError, match="audio stream has no packets"):
        audio_stream_sha256(take, 0)


def test_container_offset_and_verified_hashes(tmp_path: Path, backend: FakeBackend) -> None:
    take = make_media(tmp_path / "shifted.mkv", offset_s=0.06)
    root = make_project(tmp_path, [take])
    offset = ffmpeg.probe(take).audio_offset_s
    assert offset == pytest.approx(0.06)
    source_hash = receipt.sha256_file(take)
    result = ingest(root)
    words = load_words(root / "ingest/t1/words.json")
    assert words.time_base == "source"
    assert words.audio_offset_s == offset
    assert words.words[0].start == pytest.approx(FIXTURE_WORDS[0].start + offset)
    assert words.words[0].end == pytest.approx(FIXTURE_WORDS[0].end + offset)
    assert words.take_sha256 == source_hash == receipt.sha256_file(take)
    raw = json.loads((root / "ingest/t1/transcript.json").read_bytes())
    assert raw["schema"] == "yap-studio/transcript@1"
    assert raw["words"][0]["start"] == FIXTURE_WORDS[0].start
    assert words.transcript_sha256 == receipt.sha256_file(root / "ingest/t1/transcript.json")
    assert result["takes"][0]["words_sha256"] == receipt.sha256_file(root / "ingest/t1/words.json")
    facts = json.loads((root / "ingest/ingest.json").read_bytes())["takes"][0]
    assert facts["audio_sha256"] == audio_stream_sha256(take, facts["audio_stream"])
    identity = json.loads((root / ".yap/identity.json").read_bytes())[str(take.resolve())]
    assert "audio" not in identity
    stage = json.loads((root / "ingest/receipt.json").read_bytes())
    assert stage["inputs"] == ingest_inputs(ProjectContext(root))
    assert stage["status"] == "verified"
    assert "transcribing" in [phase["name"] for phase in stage["timings"]["phases"]]
    for output in stage["outputs"]:
        assert receipt.sha256_file(root / output["path"]) == output["sha256"]
    assert backend.languages == ["en"]
    assert not list((root / ".yap/work").iterdir())


@pytest.mark.parametrize("picture,audio_seconds", [(True, 3.5), (True, 3), (False, 3.5)])
def test_source_words_stop_at_media_end(
    tmp_path: Path, backend: FakeBackend, picture: bool, audio_seconds: float,
) -> None:
    take = tmp_path / ("take.mov" if picture else "take.wav")
    command = ["ffmpeg", "-nostdin", "-v", "error"]
    if picture:
        command += ["-f", "lavfi", "-i", "color=c=gray:s=64x64:r=30:d=3"]
    command += ["-f", "lavfi", "-i", f"sine=sample_rate=16000:duration={audio_seconds}"]
    if picture:
        command += ["-map", "0:v", "-map", "1:a", "-c:v", "libx264"]
    ffmpeg.run([*command, "-c:a", "pcm_s16le", str(take)], timeout_s=120)
    info = ffmpeg.probe(take)
    media_end = info.video_duration_s if picture else info.duration_s
    assert media_end == (3 if picture else audio_seconds)
    if picture:
        assert info.audio_duration_s == audio_seconds
    backend.words = [
        Word("One", 0.2, 0.5, 0.9),
        Word("Strawberry.", 2.7, 3.003 if audio_seconds > 3 else 3, 0.8),
    ]
    if audio_seconds > 3:
        backend.words.append(Word("After", 3.2, 3.4, 0.7))
    if not picture:
        backend.words.append(Word("Beyond", 3.6, 3.7, 0.6))
    root = make_project(tmp_path, [take])
    ingest(root)
    words = load_words(root / "ingest/t1/words.json")
    assert words.to_dict()["words"] == [
        {"id": f"t1:{index}", "text": word.text,
         "start": min(word.start, media_end), "end": min(word.end, media_end),
         "conf": word.conf}
        for index, word in enumerate(backend.words)
    ]
    assert all(0 <= word.start <= word.end <= media_end for word in words.words)
    raw = json.loads((root / "ingest/t1/transcript.json").read_bytes())
    assert raw["words"] == [word.to_dict() for word in backend.words]
    assert words.transcript_sha256 == receipt.sha256_file(root / "ingest/t1/transcript.json")


def test_source_words_round_shifted_and_clamped_times_consistently() -> None:
    raw = Transcript([Word("First", 0.0006, 0.1005), Word("Last", 2.9, 3.01)])
    words = _source_words("t2", raw, 0.06, 3.0004)
    assert [(word.id, word.start, word.end) for word in words] == [
        ("t2:0", 0.061, 0.161), ("t2:1", 2.96, 3.0),
    ]


def test_all_takes_transcribed_in_one_batch(tmp_path: Path, backend: FakeBackend) -> None:
    takes = [
        make_media(tmp_path / "one.mkv", seconds=3), make_media(tmp_path / "two.mkv", seconds=3),
    ]
    root = make_project(tmp_path, takes)
    ingest(root)
    assert len(backend.calls) == 1
    assert len(backend.calls[0]) == 2
    assert [word.id for word in load_words(root / "ingest/t2/words.json").words] == [
        f"t2:{index}" for index in range(len(FIXTURE_WORDS))
    ]
    recorded = json.loads((root / "ingest/ingest.json").read_bytes())["takes"]
    for take, facts in zip(takes, recorded, strict=True):
        assert facts["audio_sha256"] == audio_stream_sha256(take, facts["audio_stream"])


@pytest.mark.parametrize("when", ["decode", "transcribe"])
def test_changed_take_does_not_publish_shared_transcript(tmp_path, backend, monkeypatch, when):
    from yap_studio.stages import ingest as ingest_module
    from yap_studio.transcript import shared

    take = make_media(tmp_path / "take.mkv", seconds=3, sound="silence")
    digest = receipt.sha256_file(take)
    root = make_project(tmp_path / "project", [take])

    def change():
        take.write_bytes(take.read_bytes() + b"changed")

    if when == "decode":
        decode = ingest_module._decode

        def changed_decode(*args):
            change()
            return decode(*args)

        monkeypatch.setattr(ingest_module, "_decode", changed_decode)
    else:
        backend.on_transcribe = change
    with pytest.raises(InputError, match="changed"):
        ingest(root)
    assert shared.lookup(backend, shared.Request(digest, 0, take)) is None
    assert not (root / "ingest/t1/words.json").exists()


def test_second_project_reuses_raw_transcript_and_force_refreshes(tmp_path, backend):
    from yap_studio.transcript import shared

    take = make_media(tmp_path / "take.mkv", seconds=3)
    first = make_project(tmp_path / "first", [take])
    second = make_project(tmp_path / "second", [take])
    ingest(first)
    backend.calls.clear()
    ingest(second)
    assert backend.calls == []
    for name in ("words.json", "transcript.json"):
        assert (first / "ingest/t1" / name).read_bytes() == (
            second / "ingest/t1" / name
        ).read_bytes()
    backend.words = [Word("refreshed", 0.2, 2.8)]
    ingest(second, force=True)
    assert len(backend.calls) == 1
    raw = shared.lookup(backend, shared.Request(receipt.sha256_file(take), 0, take))
    assert raw.words == backend.words


def test_whisper_ingest_reuses_auto_detected_take_job_transcript(tmp_path, monkeypatch):
    from test_transcript_shared import CountingBackend

    from yap_studio.transcript.take_job import run

    take = make_media(tmp_path / "take.mkv", seconds=3, sound="silence")
    backend = CountingBackend()
    backend.name = "faster-whisper"
    monkeypatch.setattr(transcript, "resolve_backend", lambda *_: backend)
    run(take, backend)
    assert backend.durations == [3.0]
    backend.calls.clear()
    root = make_project(tmp_path / "project", [take])
    ingest(root)
    assert backend.calls == []
    assert backend.durations == [3.0]


def test_ingest_ignores_path_memo_store_failure(tmp_path, backend, monkeypatch):
    from yap_studio.transcript import shared

    take = make_media(tmp_path / "take.mkv", seconds=3, sound="silence")
    root = make_project(tmp_path / "project", [take])
    attempted = []

    def fail(path, digest, identity):
        assert (root / "ingest/receipt.json").is_file()
        attempted.append((path, digest, identity))
        raise OSError("synthetic memo store failure")

    monkeypatch.setattr(shared, "remember_digest", fail)
    assert ingest(root)["noop"] is False
    assert len(attempted) == 1
    assert attempted[0][0] == take
    assert attempted[0][1] == receipt.sha256_file(take)
    assert shared.known_digest(take) is None


def test_shared_hit_still_recovers_coverage(tmp_path, monkeypatch):
    from yap_studio.transcript import shared

    root, backend, spans = _dropped_sentence_project(tmp_path, monkeypatch, recover=True)
    ingest(root)
    raw = shared.lookup(backend, shared.Request(
        receipt.sha256_file(tmp_path / "speech.mov"), 0, tmp_path / "speech.mov",
    ))
    assert len(raw.words) < len(spans)
    backend.calls.clear()
    second = make_project(tmp_path / "second", [tmp_path / "speech.mov"])
    ingest(second)
    assert len(backend.calls) == 1
    assert all("-speech-" in path.name for path in backend.calls[0])
    assert (root / "ingest/t1/transcript.json").read_bytes() == (
        second / "ingest/t1/transcript.json"
    ).read_bytes()


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("engine", ["parakeet-mlx", "faster-whisper"])
def test_language_change_and_pickup_reuse_matching_shared_transcripts(
    tmp_path: Path, backend: FakeBackend, legacy: bool, engine: str,
) -> None:
    backend.name = engine
    first = make_media(tmp_path / "one.mkv", seconds=3)
    root = make_project(tmp_path, [first])
    ingest(root)
    manifest_path = root / "ingest/ingest.json"
    if legacy:
        manifest = json.loads(manifest_path.read_bytes())
        manifest.pop("language")
        write_json(manifest_path, manifest)
        receipt_path = root / "ingest/receipt.json"
        prior = json.loads(receipt_path.read_bytes())
        next(output for output in prior["outputs"] if output["path"] == "ingest/ingest.json")[
            "sha256"] = receipt.sha256_file(manifest_path)
        write_json(receipt_path, prior)
    second = make_media(tmp_path / "two.mkv", seconds=3)
    make_project(root, [first, second])
    project_data = json.loads((root / "project.json").read_bytes())
    project_data["language"] = "en" if legacy else "fr"
    write_json(root / "project.json", project_data)
    backend.calls.clear()
    ingest(root)
    expected = ["t1.wav", "t2.wav"] if engine == "faster-whisper" and not legacy else ["t2.wav"]
    assert [[path.name for path in call] for call in backend.calls] == [expected]
    assert json.loads(manifest_path.read_bytes())["language"] == project_data["language"]


def test_stable_ids_noop_and_identical_forced_ingest(project: Path, backend: FakeBackend) -> None:
    ingest(project)
    before = (project / "ingest/t1/words.json").read_bytes()
    receipt_before = (project / "ingest/receipt.json").read_bytes()
    assert ingest(project)["noop"] is True
    assert len(backend.calls) == 1
    assert (project / "ingest/receipt.json").read_bytes() == receipt_before
    add_references(project)
    result = ingest(project, force=True)
    assert result["orphans"] == []
    assert (project / "ingest/t1/words.json").read_bytes() == before


def test_forced_ingest_binds_proxy_file_hash(tmp_path: Path, backend: FakeBackend) -> None:
    take = make_media(tmp_path / "take.mkv", seconds=3)
    root = make_project(tmp_path, [take])
    original = ingest(root)["takes"][0]
    add_references(root)
    proxy = tmp_path / "proxy.mkv"
    ffmpeg.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(take),
            "-map", "0:v:0", "-map", "0:a:0", "-vf", "scale=32:32",
            "-c:v", "libx264", "-c:a", "copy", str(proxy),
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    proxy.replace(take)
    digest = receipt.sha256_file(take)
    assert digest != original["sha256"]
    assert audio_stream_sha256(take, 0) == original["audio_sha256"]
    assert ingest_inputs(ProjectContext(root))["take:t1"] == original["sha256"]
    assert ingest(root)["noop"] is True
    assert len(backend.calls) == 1
    result = ingest(root, force=True)
    assert result["noop"] is False
    assert len(backend.calls) == 2
    manifest = json.loads((root / "ingest/ingest.json").read_bytes())
    assert result["takes"][0]["sha256"] == manifest["takes"][0]["sha256"] == digest
    assert manifest["takes"][0]["audio_sha256"] == original["audio_sha256"]
    assert load_words(root / "ingest/t1/words.json").take_sha256 == digest
    stage_receipt = json.loads((root / "ingest/receipt.json").read_bytes())
    assert stage_receipt["inputs"]["take:t1"] == digest
    assert ingest(root)["noop"] is True


def test_referenced_replacement_refused_then_forced(project: Path, backend: FakeBackend) -> None:
    ingest(project)
    add_references(project)
    before = (project / "ingest/t1/words.json").read_bytes()
    backend.words = [Word("Changed", 0.2, 0.5)]
    backend.model = "fixture-v2"
    with pytest.raises(IngestRefusal) as caught:
        ingest(project, asr="parakeet-mlx")
    assert caught.value.code == "words_referenced"
    assert "--force" in (caught.value.suggested_action or "")
    assert all(
        relative in str(caught.value)
        for relative in (
            "edit.json",
            "timeline/agent.json",
            "timeline/broll.json",
        )
    )
    assert (project / "ingest/t1/words.json").read_bytes() == before
    result = ingest(project, force=True)
    assert len(result["orphans"]) == 6
    assert {item["file"] for item in result["orphans"]} == {
        "edit.json",
        "timeline/agent.json",
        "timeline/broll.json",
    }
    assert {item["word"] for item in result["orphans"]} == {"t1:0", "t1:5"}
    assert {item["old_word"] for item in result["orphans"]} == {"Here's", "you."}
    old_hash = hashlib.sha256(before).hexdigest()
    digest = receipt.sha256_file(project / "ingest/t1/previous" / f"{old_hash}.json")
    assert digest == old_hash


def test_force_identical_words_does_not_save_snapshot(project: Path, backend: FakeBackend) -> None:
    ingest(project)
    ingest(project, force=True)
    assert not (project / "ingest/t1/previous").exists()


def test_cycling_words_reuses_existing_snapshots(project: Path, backend: FakeBackend) -> None:
    seen = []
    for text in ("alpha", "beta", "alpha", "beta"):
        backend.words = [Word(text, 0.2, 0.5)]
        ingest(project, force=bool(seen))
        seen.append((project / "ingest/t1/words.json").read_bytes())
        assert status(project)["stages"][0]["state"] == "done"
    first, second, third, fourth = seen
    assert first == third and second == fourth
    for payload in (first, second):
        digest = hashlib.sha256(payload).hexdigest()
        assert (project / f"ingest/t1/previous/{digest}.json").read_bytes() == payload


def test_two_reingests_keep_both_snapshots_and_rebase_from_first(
    project: Path, backend: FakeBackend,
) -> None:
    from yap_studio.stages.rebase import rebase_edit

    ingest(project)
    first = (project / "ingest/t1/words.json").read_bytes()
    first_hash = hashlib.sha256(first).hexdigest()
    write_json(project / "edit.json", {
        "schema": "yap-studio/edit@1", "words_sha256": {"t1": first_hash},
        "segments": [{"id": "s1", "take": "t1", "from": {"word": "t1:0"},
                      "to": {"word": "t1:5"}}],
    })
    backend.words = [Word("First", 0.01, 0.08), *backend.words]
    backend.model = "fixture-v2"
    ingest(project, force=True)
    second = (project / "ingest/t1/words.json").read_bytes()
    second_hash = hashlib.sha256(second).hexdigest()
    first_snapshot = project / f"ingest/t1/previous/{first_hash}.json"
    assert first_snapshot.read_bytes() == first
    backend.words.insert(1, Word("Second", 0.09, 0.15))
    backend.model = "fixture-v3"
    ingest(project, force=True)
    assert first_snapshot.read_bytes() == first
    assert (project / f"ingest/t1/previous/{second_hash}.json").read_bytes() == second
    assert status(project)["stages"][0]["state"] == "done"
    assert rebase_edit(project)["status"] == "rebased"
    edit = json.loads((project / "edit.json").read_bytes())
    assert edit["segments"][0]["from"]["word"] == "t1:2"
    assert edit["words_sha256"]["t1"] == receipt.sha256_file(
        project / "ingest/t1/words.json")


def test_time_anchored_edit_still_guards_take(project: Path, backend: FakeBackend) -> None:
    ingest(project)
    add_references(project, time_only=True)
    backend.model = "fixture-v2"
    with pytest.raises(IngestRefusal):
        ingest(project, asr="parakeet-mlx")


def test_timeline_references_alone_guard_changed_words(project: Path, backend: FakeBackend) -> None:
    ingest(project)
    add_references(project)
    (project / "edit.json").unlink()
    backend.words = [Word("Changed", 0.2, 0.5)]
    backend.model = "fixture-v2"
    with pytest.raises(IngestRefusal):
        ingest(project, asr="parakeet-mlx")
    result = ingest(project, force=True)
    assert len(result["orphans"]) == 4
    assert {item["file"] for item in result["orphans"]} == {
        "timeline/agent.json",
        "timeline/broll.json",
    }


def test_missing_words_still_guard_existing_references(project: Path, backend: FakeBackend) -> None:
    ingest(project)
    add_references(project)
    (project / "ingest/t1/words.json").unlink()
    with pytest.raises(IngestRefusal):
        ingest(project)


def test_force_orphans_ignore_case_and_punctuation(project: Path, backend: FakeBackend) -> None:
    ingest(project)
    add_references(project)
    backend.words[0] = Word("HERE'S!", 0.2, 0.5, 0.9)
    result = ingest(project, force=True)
    assert result["orphans"] == []


@pytest.mark.parametrize(("transfer", "expected"), [("arib-std-b67", "hlg"), ("smpte2084", "pq")])
def test_hdr_and_source_rate_warnings(
    tmp_path: Path,
    backend: FakeBackend,
    transfer: str,
    expected: str,
) -> None:
    root = make_project(tmp_path, [make_media(tmp_path / "hdr.mkv", transfer=transfer, fps=25)])
    facts = ingest(root)["takes"][0]
    assert facts["hdr"] == expected
    assert facts["probe"]["color_transfer"] == transfer
    assert facts["probe"]["fps"] == "25/1"
    assert "format_tags" not in facts["probe"]
    assert any("HDR" in warning for warning in facts["warnings"])
    assert any("25 fps" in warning for warning in facts["warnings"])


def test_vfr_warning_from_probe_rates(
    project: Path, backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = ffmpeg.run

    def rates(command: list[str], **kwargs: Any) -> Any:
        result = original(command, **kwargs)
        if command[0] == "ffprobe" and "-show_streams" in command and "-show_format" not in command:
            data = json.loads(result.stdout)
            data["streams"][0].update(avg_frame_rate="299/10", r_frame_rate="30/1")
            result.stdout = json.dumps(data)
        return result

    monkeypatch.setattr(ffmpeg, "run", rates)
    facts = ingest(project)["takes"][0]
    assert facts["vfr"] is True
    assert any("VFR" in warning for warning in facts["warnings"])


def test_tonal_bed_under_speech_and_music_only(tmp_path: Path, backend: FakeBackend) -> None:
    root = make_project(tmp_path, [make_media(tmp_path / "tone.mkv")])
    # Mix the continuous tone with intermittent speech-like bursts.
    source = root / "tone.mkv"
    bed = make_media(root / "bed.mkv", sound="tone")
    with paths.atomic_output(root / "mixed.mkv") as temporary:
        ffmpeg.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(source),
                "-i",
                str(bed),
                "-filter_complex",
                "[0:a]volume='if(lt(mod(t,1),0.2),0.15,0)':eval=frame[s];"
                "[s][1:a]amix=inputs=2:normalize=0[a]",
                "-map",
                "0:v:0",
                "-map",
                "[a]",
                "-c:v",
                "copy",
                "-c:a",
                "pcm_s16le",
                "-f",
                "matroska",
                str(temporary),
            ],
            timeout_s=135,
        )
    make_project(root, [root / "mixed.mkv"])
    backend.words = [
        Word(f"word{index}", index * 0.5 + 0.1, index * 0.5 + 0.3) for index in range(20)
    ]
    facts = ingest(root)["takes"][0]
    assert facts["music_ranges"] == [{"start": 0.0, "end": 10.0, "kind": "music_under_speech"}]
    assert any("music under the voice" in warning for warning in facts["warnings"])
    assert all(
        window["gap20"] < 0.05 and window["quiet_flatness"] < 0.03 and window["floor10_dbfs"] > -45
        for window in facts["music_windows"]
    )
    backend.words = []
    facts = ingest(root, force=True)["takes"][0]
    assert facts["music_ranges"][0]["kind"] == "music_only"


def test_no_audio_fails_before_speech(tmp_path: Path, backend: FakeBackend) -> None:
    root = make_project(tmp_path, [make_media(tmp_path / "silent.mkv", sound="none")])
    with pytest.raises(InputError, match="the take has no audio; record with sound"):
        ingest(root)
    assert backend.calls == []


def test_stream_override_and_clock(tmp_path: Path, backend: FakeBackend) -> None:
    take = make_media(tmp_path / "streams.mkv", second_stream=True)
    root = make_project(tmp_path, [take], stream=1)
    facts = ingest(root)["takes"][0]
    assert facts["audio_stream"] == 1
    recorded = json.loads((root / "ingest/ingest.json").read_bytes())["takes"][0]
    assert recorded["audio_sha256"] == facts["audio_sha256"] == audio_stream_sha256(take, 1)
    assert facts["audio_sha256"] != audio_stream_sha256(take, 0)
    assert facts["audio_offset_s"] == pytest.approx(0.12)
    assert load_words(root / "ingest/t1/words.json").words[0].start == pytest.approx(0.32)
    assert any("non-default audio stream 1" in warning for warning in facts["warnings"])


def test_skips_undecodable_first_stream(
    tmp_path: Path, backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    take = make_media(tmp_path / "streams.mkv", second_stream=True)
    root = make_project(tmp_path, [take])
    original = ffmpeg.run

    def reject(command: list[str], **kwargs: Any) -> Any:
        if "-t" in command and "0:a:0" in command:
            raise InputError(take.name, "unsupported decoder")
        return original(command, **kwargs)

    monkeypatch.setattr(ffmpeg, "run", reject)
    assert ingest(root)["takes"][0]["audio_stream"] == 1


def test_preflight_silence_fails_and_recording_passes(tmp_path: Path, backend: FakeBackend) -> None:
    silence = make_media(tmp_path / "silence.mkv", sound="silence")
    backend.words = []
    failed = preflight(silence)
    assert failed["status"] == "FAIL"
    assert "at_least_five_words" in failed["failed_checks"]
    assert "loudness" in failed["failed_checks"]
    backend.words = list(FIXTURE_WORDS)
    speech = make_media(tmp_path / "recording.mkv")
    passed = preflight(speech)
    assert passed["status"] == "PASS"
    assert passed["words_count"] == 6
    assert -28 <= passed["integrated_lufs"] <= -16
    assert passed["true_peak_dbtp"] <= -0.3
    assert not (tmp_path / ".yap").exists()
    assert not (tmp_path / "ingest").exists()


def test_preflight_music_fails(tmp_path: Path, backend: FakeBackend) -> None:
    result = preflight(make_media(tmp_path / "bed.mkv", sound="tone"))
    assert result["status"] == "FAIL"
    assert "no_music" in result["failed_checks"]


def test_preflight_synthetic_speech_passes(tmp_path: Path, backend: FakeBackend) -> None:
    speech, _ = synth.speech_source(tmp_path / "speech.mov", SENTENCES)
    ready = tmp_path / "ready.mov"
    duration = ffmpeg.probe(speech).duration_s or 10
    with paths.atomic_output(ready) as temporary:
        ffmpeg.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(speech),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-c:v",
                "copy",
                "-af",
                "loudnorm=I=-22:TP=-2:LRA=7",
                "-c:a",
                "pcm_s16le",
                str(temporary),
            ],
            timeout_s=120 + duration * 1.5,
        )
    backend.words = [
        Word(word, 0.3 + index * 0.4, 0.5 + index * 0.4)
        for index, word in enumerate(" ".join(SENTENCES).split())
    ]
    result = preflight(ready)
    assert result["status"] == "PASS"
    assert result["music_ranges"] == []


@pytest.mark.parametrize(
    ("peak", "passed", "hot"),
    [(-1.2, True, False), (-0.8, True, True), (-0.3, True, True), (-0.2, False, False)],
)
def test_preflight_peak_boundary(
    project: Path,
    backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
    peak: float,
    passed: bool,
    hot: bool,
) -> None:
    monkeypatch.setattr("yap_studio.stages.ingest._loudness", lambda _stderr: (-22.0, 0.0, peak))
    result = preflight(project / "take.mkv")
    assert result["checks"]["true_peak"] is passed
    assert bool(result["notes"]) is hot
    if hot:
        assert result["notes"] == ["hot peak: back the mic gain off a little"]


def test_copy_opt_in_and_verified_source_preference(tmp_path: Path, backend: FakeBackend) -> None:
    take = make_media(tmp_path / "external.mkv")
    root = make_project(tmp_path / "project", [take])
    ingest(root)
    assert not (root / "ingest/takes").exists()
    result = ingest(root, copy=True)
    copied = root / result["takes"][0]["copy"]
    assert copied == root / "ingest/takes/t1.mkv"
    assert receipt.sha256_file(copied) == receipt.sha256_file(take)
    assert ProjectContext(root).take_path("t1") == copied
    assert ingest(root)["noop"] is True
    assert ingest(root, force=True)["takes"][0]["copy"] == "ingest/takes/t1.mkv"
    take.unlink()
    assert ingest(root)["noop"] is True
    assert status(root)["stages"][0]["state"] == "done"


def test_copied_take_forced_ingest_and_cut_without_original(
    tmp_path: Path, backend: FakeBackend, capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_cut import time_segment, write_edit

    from yap_studio.cli import main

    monkeypatch.setattr("yap_studio.verify.resolve_backend", lambda *_: None)
    take = synth.calib_source(tmp_path / "external.mov", seconds=4)
    root = make_project(tmp_path / "project", [take])
    assert main(["ingest", str(root), "--copy", "--json"]) == 0
    capsys.readouterr()
    take.rename(tmp_path / "unavailable.mov")
    assert main(["ingest", str(root), "--force", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["noop"] is False
    write_edit(root, [time_segment("story", "t1", 0.1, 2.5)])
    assert main(["cut", str(root), "--force", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    assert {stage["name"]: stage for stage in status(root)["stages"]}["cut"]["state"] == "done"


def test_copy_mismatch_refuses_before_publication(
    tmp_path: Path, backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    take = make_media(tmp_path / "external.mkv")
    root = make_project(tmp_path / "project", [take])
    original = receipt.sha256_file

    def wrong_hash(target: str | Path) -> str:
        return "0" * 64 if "-copy" in Path(target).name else original(target)

    monkeypatch.setattr(receipt, "sha256_file", wrong_hash)
    with pytest.raises(InputError, match="copy hash mismatch"):
        ingest(root, copy=True)
    assert not (root / "ingest/takes/t1.mkv").exists()
    assert not (root / "ingest/t1/words.json").exists()


def test_changed_copy_is_refused(tmp_path: Path, backend: FakeBackend) -> None:
    take = make_media(tmp_path / "external.mkv")
    root = make_project(tmp_path / "project", [take])
    ingest(root, copy=True)
    with (root / "ingest/takes/t1.mkv").open("ab") as output:
        output.write(b"changed")
    with pytest.raises(InputError, match="recorded copy hash mismatch"):
        ProjectContext(root).take_path("t1")


def test_copy_is_bound_to_the_configured_source(tmp_path: Path, backend: FakeBackend) -> None:
    take = make_media(tmp_path / "external.mkv")
    root = make_project(tmp_path / "project", [take])
    ingest(root, copy=True)
    replacement = make_media(tmp_path / "replacement.mkv", sound="tone")
    make_project(root, [replacement])
    assert ProjectContext(root).take_path("t1") == replacement
    assert status(root)["stages"][0]["state"] == "stale"


def test_copy_directory_symlink_cannot_escape_project(
    tmp_path: Path, backend: FakeBackend,
) -> None:
    take = make_media(tmp_path / "external.mkv")
    root = make_project(tmp_path / "project", [take])
    ingest(root, copy=True)
    copies = root / "ingest/takes"
    moved = tmp_path / "moved"
    copies.rename(moved)
    copies.symlink_to(moved, target_is_directory=True)
    with pytest.raises(InputError, match="outside ingest/takes"):
        ProjectContext(root).take_path("t1")


def test_changed_source_during_transcription_refused(project: Path, backend: FakeBackend) -> None:
    def change() -> None:
        with (project / "take.mkv").open("ab") as source:
            source.write(b"changed")

    backend.on_transcribe = change
    with pytest.raises(InputError, match="changed"):
        ingest(project)
    assert not (project / "ingest/t1/words.json").exists()


def test_new_references_during_transcription_refused(project: Path, backend: FakeBackend) -> None:
    ingest(project)
    backend.on_transcribe = lambda: add_references(project)
    with pytest.raises(ConflictError):
        ingest(project, force=True)


def test_missing_engine_and_incomplete_batch_fail(
    project: Path, backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(transcript, "resolve_backend", lambda *_args: None)
    with pytest.raises(DependencyError):
        ingest(project)
    monkeypatch.setattr(transcript, "resolve_backend", lambda *_args: backend)
    monkeypatch.setattr(backend, "transcribe_many", lambda *_args, **_kwargs: {})
    with pytest.raises(ProcessingError, match="missing result for t1"):
        ingest(project)
    assert not (project / "ingest/t1/words.json").exists()


def test_take_cannot_be_overwritten_by_ingest(tmp_path: Path, backend: FakeBackend) -> None:
    take = make_media(tmp_path / "ingest/t1/words.json")
    source_hash = receipt.sha256_file(take)
    make_project(tmp_path, [take])
    with pytest.raises(OutputPathError):
        ingest(tmp_path)
    assert receipt.sha256_file(take) == source_hash


def test_words_listing_sentences_gaps_and_cut(project: Path, backend: FakeBackend) -> None:
    backend.words = [
        *FIXTURE_WORDS,
        Word("Again!", 3.2, 3.5),
        Word("Long", 5.0, 5.2),
        Word("pause?", 6.3, 6.6),
    ]
    ingest(project)
    result = words_listing(project)
    assert result["lines"] == [
        "t1:0-t1:5  0.200-2.000  Here's the thing nobody tells you.",
        "t1:6-t1:6  3.200-3.500  Again!",
        "t1:7-t1:7  5.000-5.200  Long",
        "t1:8-t1:8  6.300-6.600  pause?",
    ]
    assert result["words"][0]["id"] == "t1:0"
    with pytest.raises(InputError, match="run yap-studio cut first"):
        words_listing(project, cut=True)
    write_json(
        project / "cut/cut-words.json",
        {
            "schema": "yap-studio/cut-words@1",
            "time_base": "output",
            "words_sha256": {"t1": receipt.sha256_file(project / "ingest/t1/words.json")},
            "words": [{"id": "t1:5", "segment": "s1", "text": "you.", "start": 0, "end": 0.2}],
        },
    )
    result = words_listing(project, cut=True)
    assert result["time_base"] == "output"
    assert result["lines"] == ["t1:5-t1:5  0.000-0.200  you."]


def test_mlx_timeout_scales_with_total_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager
    from types import SimpleNamespace

    from yap_studio.transcript import backends

    converted = tmp_path / "converted.wav"
    seen = []

    @contextmanager
    def converted_wavs(_files: Sequence[Path]) -> Any:
        yield [converted], [converted], [1200.0]

    def spawn(command: list[str], **kwargs: Any) -> Any:
        seen.append(kwargs["timeout_s"])
        output = Path(command[command.index("--output-dir") + 1]) / "converted.json"
        write_json(output, {"schema": "yap-studio/transcript@1", "words": [], "text": ""})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(backends, "_converted_wavs", converted_wavs)
    monkeypatch.setattr(
        ParakeetMlxBackend, "executable", staticmethod(lambda: Path("parakeet-mlx"))
    )
    monkeypatch.setattr(backends.process, "spawn", spawn)
    ParakeetMlxBackend().transcribe_many([converted])
    assert seen == [1950.0]


class DroppedSentenceBackend(FakeBackend):
    """Drop the middle sentence on the take and optionally recover it in clips."""

    def __init__(self, spans: list[synth.WordSpan], *, recover: bool) -> None:
        super().__init__()
        self.spans = spans
        self.recover = recover
        self.full_samples: np.ndarray | None = None

    def transcribe_many(
        self, files: Sequence[Path], *, language: str | None = None,
    ) -> dict[Path, Transcript]:
        """Return words using the sample offset of each exact WAV clip."""
        self.calls.append(list(files))
        results = {}
        for wav in files:
            info = ffmpeg.probe(wav, count_frames=False)
            assert (info.sample_rate, info.channels, info.audio_codec, info.audio_offset_s) == (
                16_000, 1, "pcm_f32le", 0,
            )
            samples = np.asarray(_float_wav(wav))
            if self.full_samples is None:
                self.full_samples = samples.copy()
            if len(samples) == len(self.full_samples):
                first = [span for span in self.spans if span.sentence == 0]
                third = [span for span in self.spans if span.sentence == 2]
                words = [Word(span.text, span.start, span.end) for span in [*first, *third]]
                last = len(first) - 1
                words[last] = Word(words[last].text, words[last].start, third[0].start)
            else:
                pivot = int(np.flatnonzero(samples)[0])
                matching = np.flatnonzero(self.full_samples == samples[pivot])
                offsets = [int(index) - pivot for index in matching]
                start = next(
                    offset for offset in offsets
                    if offset >= 0 and np.array_equal(
                        self.full_samples[offset : offset + len(samples)], samples
                    )
                )
                offset_s = start / 16_000
                words = [
                    Word(span.text, span.start - offset_s, span.end - offset_s)
                    for span in self.spans
                    if self.recover and offset_s <= (span.start + span.end) / 2 <
                    offset_s + len(samples) / 16_000
                ]
            results[wav] = Transcript(words, language, self.name, self.model, info.duration_s)
        return results


def _dropped_sentence_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, recover: bool,
) -> tuple[Path, DroppedSentenceBackend, list[synth.WordSpan]]:
    """Build one synthetic take with a scripted missing middle sentence."""
    take, spans = synth.speech_source(tmp_path / "speech.mov", SENTENCES)
    fake = DroppedSentenceBackend(spans, recover=recover)
    monkeypatch.setattr(transcript, "resolve_backend", lambda *_args: fake)
    return make_project(tmp_path, [take]), fake, spans


def test_ingest_recovers_dropped_speech_in_one_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, fake, spans = _dropped_sentence_project(tmp_path, monkeypatch, recover=True)
    result = ingest(root)
    words = load_words(root / "ingest/t1/words.json").words
    middle = [span.text for span in spans if span.sentence == 1]
    texts = [word.text for word in words]
    assert texts == [span.text for span in spans]
    saved = Transcript.load(root / "ingest/t1/transcript.json")
    assert [word.text for word in saved.words] == texts
    assert all(
        abs(source.start - word.start) <= 0.001 and abs(source.end - word.end) <= 0.001
        for source, word in zip(saved.words, words, strict=True)
    )
    assert [word.id for word in words] == [f"t1:{i}" for i in range(len(words))]
    assert all(word in texts for word in middle)
    assert not any("speech with no words" in warning for warning in result["warnings"])
    assert result["takes"][0]["speech_check"]["untranscribed"] == []
    assert result["takes"][0]["speech_check"]["words_added"] == len(middle)
    assert len(fake.calls) == 2
    assert len(fake.calls[0]) == len(fake.calls[1]) == 1
    stage = json.loads((root / "ingest/receipt.json").read_text())
    assert stage["status"] == "verified"
    assert {item["id"]: item["pass"] for item in stage["asserts"]}["speech_covered"]


def test_ingest_warns_when_dropped_speech_stays_untranscribed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    root, fake, _ = _dropped_sentence_project(tmp_path, monkeypatch, recover=False)
    assert main(["ingest", str(root)]) == 0
    output = capsys.readouterr().out
    assert output.startswith("Ingest finished: some speech has no words.\n")
    assert "warning: t1: speech with no words at " in output
    manifest = json.loads((root / "ingest/ingest.json").read_text())
    stretches = manifest["takes"][0]["speech_check"]["untranscribed"]
    assert stretches and stretches[0]["seconds"] > 0.6
    assert any("with no words" in item for item in manifest["warnings"])
    stage = json.loads((root / "ingest/receipt.json").read_text())
    assert stage["status"] == "verified"
    assert not {item["id"]: item["pass"] for item in stage["asserts"]}["speech_covered"]
    assert stage["untranscribed"][0]["take"] == "t1"
    assert status(root)["warnings"] == manifest["warnings"][:len(stretches)]
    assert main(["status", str(root)]) == 0
    assert "warning: t1: speech with no words at " in capsys.readouterr().out
    assert len(fake.calls) == 2


def test_ingest_batches_recovery_windows_across_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, fake, spans = _dropped_sentence_project(tmp_path, monkeypatch, recover=True)
    make_project(root, [root / "speech.mov", root / "speech.mov"])
    result = ingest(root)
    assert [take["speech_check"]["words_added"] for take in result["takes"]] == [
        len([span for span in spans if span.sentence == 1])
    ] * 2
    assert len(fake.calls) == 2
    assert [len(call) for call in fake.calls] == [1, 2]


def test_missing_recovery_result_raises_speech_engine_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, fake, _ = _dropped_sentence_project(tmp_path, monkeypatch, recover=False)
    original = fake.transcribe_many

    def incomplete(files: Sequence[Path], *, language: str | None = None) -> dict[Path, Transcript]:
        if fake.calls:
            fake.calls.append(list(files))
            return {}
        return original(files, language=language)

    monkeypatch.setattr(fake, "transcribe_many", incomplete)
    with pytest.raises(ProcessingError) as caught:
        ingest(root)
    assert caught.value.code == "speech_engine_failed"
    assert "missing result for t1" in caught.value.message


@pytest.mark.asr
def test_real_engine_espeak_recovery(tmp_path: Path) -> None:
    backend = transcript.resolve_backend("auto")
    if backend is None:
        pytest.skip("speech engine not installed")
    take, _ = synth.speech_source(tmp_path / "speech.mov", SENTENCES)
    root = make_project(tmp_path, [take])
    ingest(root)
    expected = Counter(transcript.norm_words(" ".join(SENTENCES)))
    recovered = Counter(
        transcript.norm_words(
            " ".join(word.text for word in load_words(root / "ingest/t1/words.json").words)
        )
    )
    assert sum((expected & recovered).values()) / sum(expected.values()) >= 0.9


@pytest.mark.asr
def test_twenty_minute_take_ingests_with_scaled_timeout(tmp_path: Path) -> None:
    backend = transcript.resolve_backend("auto")
    if backend is None:
        pytest.skip("speech engine not installed")
    seed, _ = synth.speech_source(tmp_path / "seed.mov", SENTENCES)
    take = tmp_path / "long.mkv"
    with paths.atomic_output(take) as temporary:
        ffmpeg.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=gray:s=64x64:r=30:d=1200",
                "-stream_loop",
                "-1",
                "-i",
                str(seed),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-t",
                "1200",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "pcm_s16le",
                "-f",
                "matroska",
                str(temporary),
            ],
            timeout_s=1920,
        )
    root = make_project(tmp_path / "project", [take])
    started = time.monotonic()
    result = ingest(root)
    assert time.monotonic() - started <= 120 + 30 + 1.5 * 1200
    assert result["engine"] == backend.name
    assert load_words(root / "ingest/t1/words.json").words
    assert result["takes"][0]["probe"]["duration_s"] == pytest.approx(1200, abs=0.1)
