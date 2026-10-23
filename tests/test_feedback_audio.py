"""Voice envelopes measure delivery independently of words and pauses."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tracemalloc
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_feedback_versions import _render_media

from yap_studio.errors import InputError, ProcessingError, ValidationError
from yap_studio.feedback import audio as audio_module
from yap_studio.feedback.audio import (
    ENVELOPE_RATE,
    ENVELOPE_SCHEMA,
    compare_delivery,
    envelope_slice,
    parse_envelope,
    voice_envelope,
)


@pytest.mark.parametrize("gain,expected", [(0.0, "kept"), (1.0, "kept"), (4.0, "broken"),
                                          (-4.2, "broken")])
def test_compare_delivery_gain(gain: float, expected: str) -> None:
    kept = [-23.0, -25.0, -30.0, -24.0, -28.0] * 10
    status, detail = compare_delivery(kept, [value + gain for value in kept])
    assert status == expected
    if gain > 3:
        assert detail == "speech is 4.0 dB louder"
    if gain < -3:
        assert detail == "speech is 4.2 dB quieter"


def test_compare_delivery_one_word_louder_changes_contour() -> None:
    status, detail = compare_delivery([-25.0] * 30, [-25.0] * 20 + [-17.0] * 10)
    assert status == "broken"
    assert detail == "delivery changed: level contour differs by 2.7 dB on average"


@pytest.mark.parametrize("kept,new", [([-80.0] * 10, [-80.0] * 10), ([-50.0] * 10, [-50.0] * 10),
                                      ([-23.0] * 4, [-23.0] * 5), ([], [])])
def test_compare_delivery_too_little_speech(kept: list[float], new: list[float]) -> None:
    assert compare_delivery(kept, new) == ("unmeasured", "too little speech to measure delivery")


def test_compare_delivery_ignores_joint_silence_and_quiet_tail() -> None:
    kept = [-80.0] * 10 + [-23.0] * 5
    new = [-60.0] * 10 + [-22.0] * 5 + [-80.0] * 20
    assert compare_delivery(kept, new)[0] == "kept"
    assert compare_delivery([-50.0] * 5, [-49.9] * 5)[0] == "kept"
    assert compare_delivery([-23.0] * 5, [-20.0] * 5)[0] == "kept"


@pytest.mark.parametrize("channels", [1, 2])
def test_voice_envelope_known_sine_level(tmp_path: Path, channels: int) -> None:
    samples = 0.25 * np.sin(2 * np.pi * 1000 * np.arange(4800) / 48000)
    if channels == 2:
        samples = np.stack((samples, samples * 0.5), axis=1)
    pcm = (samples * 32768).astype("<i2")
    path = tmp_path / "voice.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(pcm.tobytes())
    envelope = voice_envelope(path)
    expected = 20 * math.log10(0.25 * (0.75 if channels == 2 else 1) / math.sqrt(2))
    assert envelope["schema"] == ENVELOPE_SCHEMA and envelope["rate"] == ENVELOPE_RATE
    assert len(envelope["db"]) == 10
    assert envelope["db"] == pytest.approx([expected] * 10, abs=0.2)
    assert parse_envelope(envelope) == envelope


def test_voice_envelope_mixes_channels_before_rms_and_includes_partial_window(
    tmp_path: Path,
) -> None:
    samples = np.full(500, 1000, dtype="<i2")
    pcm = np.stack((samples, -samples), axis=1)
    path = tmp_path / "voice.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(pcm.tobytes())
    assert voice_envelope(path)["db"] == [-80.0, -80.0]


def test_voice_envelope_refuses_non_16_bit_pcm(tmp_path: Path) -> None:
    path = tmp_path / "voice.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(1)
        wav.setframerate(48000)
        wav.writeframes(b"\x80" * 480)
    with pytest.raises(InputError, match="16-bit PCM"):
        voice_envelope(path)


@pytest.mark.parametrize("bad", [None, [], {}, {"schema": "wrong", "rate": 100, "db": []},
                               {"schema": ENVELOPE_SCHEMA, "rate": True, "db": []},
                               {"schema": ENVELOPE_SCHEMA, "rate": 100.0, "db": []},
                               {"schema": ENVELOPE_SCHEMA, "rate": 99, "db": []},
                               {"schema": ENVELOPE_SCHEMA, "rate": 100, "db": (), "extra": 1}])
def test_parse_envelope_rejects_bad_shape(bad: object) -> None:
    with pytest.raises(ValidationError):
        parse_envelope(bad)


@pytest.mark.parametrize("value", [True, None, "-20", -81, 1, float("nan"), float("inf"),
                                 float("-inf")])
def test_parse_envelope_rejects_invalid_levels(value: object) -> None:
    with pytest.raises(ValidationError):
        parse_envelope({"schema": ENVELOPE_SCHEMA, "rate": 100, "db": [value]})


def test_envelope_slice_uses_output_time_and_clips() -> None:
    envelope = {"schema": ENVELOPE_SCHEMA, "rate": 100, "db": list(range(100))}
    assert envelope_slice(envelope, 0.2, 0.5) == list(range(20, 50))
    assert envelope_slice(envelope, 0.201, 0.499) == list(range(20, 50))
    assert envelope_slice(envelope, -1, 0.01) == [0]
    assert envelope_slice(envelope, 0.9, 2) == list(range(90, 100))


@pytest.mark.parametrize("shorter", [False, True])
def test_regression_compare_delivery_detects_speech_tail(shorter: bool) -> None:
    kept = [-23.0] * 30
    new = kept + [-10.0] * 20
    if shorter:
        kept, new = new, kept
    direction = "shorter" if shorter else "longer"
    assert compare_delivery(kept, new) == (
        "broken", f"the kept speech is 0.20 s {direction} in the new version",
    )


@pytest.mark.parametrize("windows", [1, 2])
@pytest.mark.parametrize("shorter", [False, True])
def test_compare_delivery_tolerates_two_windows(windows: int, shorter: bool) -> None:
    kept, new = [-23.0] * 30, [-23.0] * 30 + [-10.0] * windows
    assert compare_delivery(*(new, kept) if shorter else (kept, new))[0] == "kept"


@pytest.mark.parametrize("speech_windows", [0, 2, 3])
@pytest.mark.parametrize("shorter", [False, True])
def test_compare_delivery_tail_speech_threshold(speech_windows: int, shorter: bool) -> None:
    kept = [-23.0] * 30
    new = kept + [-50.0] * (20 - speech_windows) + [-49.9] * speech_windows
    result = compare_delivery(*(new, kept) if shorter else (kept, new))
    assert result[0] == ("broken" if speech_windows > 2 else "kept")


def test_compare_delivery_quiet_tail_still_checks_overlap() -> None:
    assert compare_delivery([-23.0] * 30, [-15.0] * 30 + [-80.0] * 20) == (
        "broken", "speech is 8.0 dB louder",
    )


def test_media_envelope_measures_render_audio(tmp_path: Path) -> None:
    media = tmp_path / "render.mp4"
    _render_media(media, 30, amplitude=0.25)
    envelope = audio_module.media_envelope(media)
    assert parse_envelope(envelope) == envelope
    assert 100 <= len(envelope["db"]) <= 103
    expected = 20 * math.log10(0.25 / math.sqrt(2))
    assert envelope["db"][10:90] == pytest.approx([expected] * 80, abs=0.3)


def test_media_envelope_without_audio_is_none(tmp_path: Path) -> None:
    media = tmp_path / "silent.mp4"
    _render_media(media, 3)
    assert audio_module.media_envelope(media) is None


def test_regression_media_envelope_streams_ten_minutes_with_bounded_memory(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg")
    media = tmp_path / "long.flac"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
        "sine=frequency=1000:sample_rate=48000:duration=600", "-c:a", "flac", str(media),
    ], stdin=subprocess.DEVNULL, capture_output=True, timeout=60, check=True)
    tracemalloc.start()
    try:
        envelope = audio_module.media_envelope(media)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(envelope["db"]) in (60000, 60001)
    assert peak < 32 * 1024 * 1024, f"peak traced memory was {peak / 1024**2:.1f} MiB"


def test_regression_media_envelope_aligns_delayed_audio_to_video(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("needs ffmpeg and ffprobe")
    media = tmp_path / "delayed.mkv"
    try:
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
            "color=c=black:size=32x32:rate=30:duration=2", "-itsoffset", "0.12",
            "-f", "lavfi", "-i",
            "aevalsrc=if(between(t\\,0.50\\,0.70)\\,0.5*sin(2*PI*1000*t)\\,0):s=48000:d=2",
            "-c:v", "libx264", "-c:a", "pcm_s16le", str(media),
        ], stdin=subprocess.DEVNULL, capture_output=True, timeout=60, check=True)
    except subprocess.CalledProcessError:
        pytest.skip("local ffmpeg cannot produce the delayed audio fixture")
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time",
        "-of", "json", str(media),
    ], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10, check=True)
    streams = {stream["codec_type"]: stream for stream in json.loads(result.stdout)["streams"]}
    if (not math.isclose(float(streams["audio"].get("start_time", "nan")), 0.12, abs_tol=0.005)
            or float(streams["video"].get("start_time", "nan")) != 0):
        pytest.skip("local ffmpeg did not preserve audio start at 0.12 s and video start at 0 s")
    envelope = audio_module.media_envelope(media)
    assert len(envelope["db"]) >= 82
    assert all(level > -30 for level in envelope["db"][63:82])
    assert all(level < -60 for level in envelope["db"][50:59])


@pytest.mark.parametrize("chunk_size", [1, 3, 479, 959, 960, 961, 65537])
def test_pcm_accumulator_matches_one_shot_across_sample_and_window_boundaries(
    tmp_path: Path, chunk_size: int,
) -> None:
    samples = np.random.default_rng(42).integers(-32768, 32768, 480 * 150 + 317, dtype="<i2")
    samples[:480] = -32768
    samples[480:960] = 1
    samples[960:1440] = 0
    pcm = samples.tobytes()
    expected = audio_module._rms_envelope(samples.astype(np.float64) / 32768, 48000)
    accumulator = audio_module._PCMEnvelope()
    for start in range(0, len(pcm), chunk_size):
        accumulator.update(pcm[start:start + chunk_size])
        assert len(accumulator._remainder) < 960
    assert accumulator.finish() == expected
    wav_path = tmp_path / "same.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(pcm)
    assert voice_envelope(wav_path) == expected


def _mock_probe(monkeypatch: pytest.MonkeyPatch, **offsets) -> None:
    info = SimpleNamespace(has_audio=True, has_video=True, audio_offset_s=0.0, video_offset_s=0.0)
    for name, value in offsets.items():
        setattr(info, name, value)

    def probe(path, *, count_frames):
        assert count_frames is False
        return info

    monkeypatch.setattr(audio_module.ffmpeg, "probe", probe)


@pytest.mark.parametrize("exists", [False, True])
def test_media_envelope_refuses_unreadable_media(tmp_path: Path, exists: bool) -> None:
    media = tmp_path / "unreadable.mp4"
    if exists:
        if shutil.which("ffmpeg") is None:
            pytest.skip("needs ffmpeg")
        media.write_bytes(b"invalid media")
    with pytest.raises(InputError):
        audio_module.media_envelope(media)


@pytest.mark.parametrize("chunk_size", [65536, 17])
@pytest.mark.parametrize("size,timeout", [(0, 60), (16 * 1024 * 1024, 80)])
def test_media_envelope_decode_command_and_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: int, timeout: float, chunk_size: int,
) -> None:
    media = tmp_path / "render.mp4"
    with media.open("wb") as output:
        output.truncate(size)
    pcm = np.full(500, 3277, dtype="<i2").tobytes()
    _mock_probe(monkeypatch)
    expected_chunk_size = chunk_size

    def decode(command, *, timeout_s, chunk_size):
        assert command == [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(media), "-map", "0:a:0",
            "-af", "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0",
            "-ac", "1", "-ar", "48000", "-f", "s16le", "-",
        ]
        assert timeout_s == timeout and chunk_size == expected_chunk_size
        for start in range(0, len(pcm), chunk_size):
            yield pcm[start:start + chunk_size]

    monkeypatch.setattr(audio_module.ffmpeg, "run_stream", decode)
    assert audio_module.media_envelope(media, chunk_size=chunk_size)["db"] == [-20.0, -20.0]


@pytest.mark.parametrize("partial", [False, True])
def test_media_envelope_decode_failure_is_input_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, partial: bool,
) -> None:
    media = tmp_path / "render.mp4"
    media.touch()
    _mock_probe(monkeypatch)
    closed = []

    def decode(command, **kwargs):
        try:
            if partial:
                yield b"\0" * 960
            raise ProcessingError("ffmpeg", 1, "Error while decoding audio")
        finally:
            closed.append(True)

    monkeypatch.setattr(audio_module.ffmpeg, "run_stream", decode)
    with pytest.raises(InputError, match="cannot decode render audio"):
        audio_module.media_envelope(media)
    assert closed == [True]


@pytest.mark.parametrize("exit_kind", ["success", "accumulator_error", "odd_pcm"])
def test_media_envelope_closes_stream_on_accumulator_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_kind: str,
) -> None:
    media = tmp_path / "render.mp4"
    media.touch()
    _mock_probe(monkeypatch)
    closed = []

    def decode(command, **kwargs):
        try:
            yield b"\0" * 960
            yield b"\0" if exit_kind == "odd_pcm" else b"\0" * 2
        finally:
            closed.append(True)

    def fail(chunk):
        raise ValueError("cannot accumulate")

    monkeypatch.setattr(audio_module.ffmpeg, "run_stream", decode)
    if exit_kind == "accumulator_error":
        monkeypatch.setattr(audio_module._PCMEnvelope, "update", lambda self, chunk: fail(chunk))
    if exit_kind != "success":
        with pytest.raises(InputError, match="cannot decode render audio"):
            audio_module.media_envelope(media)
    else:
        assert audio_module.media_envelope(media)["db"] == [-80.0, -80.0]
    assert closed == [True]


@pytest.mark.parametrize("offset", [0.12, -0.12])
@pytest.mark.parametrize("has_video", [False, True])
def test_media_envelope_offsets_and_negative_sample_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offset: float, has_video: bool,
) -> None:
    media = tmp_path / "render.mkv"
    media.touch()
    _mock_probe(monkeypatch, has_video=has_video, audio_offset_s=0.25 + offset, video_offset_s=0.25)
    samples = np.concatenate((np.full(5760, 3277, dtype="<i2"), np.full(497, 16384, dtype="<i2")))
    pcm = samples.tobytes()

    def decode(command, **kwargs):
        for start in range(0, len(pcm), 137):
            yield pcm[start:start + 137]

    monkeypatch.setattr(audio_module.ffmpeg, "run_stream", decode)
    levels = audio_module.media_envelope(media)["db"]
    if not has_video:
        assert levels == [-20.0] * 12 + [-6.0] * 2
    elif offset > 0:
        assert levels == [-80.0] * 12 + [-20.0] * 12 + [-6.0] * 2
    else:
        assert levels == [-6.0] * 2
