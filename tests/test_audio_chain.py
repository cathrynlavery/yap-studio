"""Synthetic acceptance, timing and spectral checks for fixed speech filters."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
import synth

from yap_studio import ffmpeg
from yap_studio.audio_chain import VOICE_FILTERS, voice_filters
from yap_studio.errors import YapError
from yap_studio.stages.assemble_audio import RATE


@pytest.mark.parametrize(("name", "expected"), [
    ("none", ""),
    ("rumble", "highpass=f=80:p=2"),
    ("proximity", "highpass=f=100:p=2,equalizer=f=200:t=q:w=1:g=-3"),
    ("proximity_strong", "highpass=f=100:p=2,equalizer=f=200:t=q:w=0.8:g=-5"),
    ("room", "agate=threshold=0.02:ratio=1.3:range=0.5:attack=10:release=200:makeup=1"),
    ("harshness", "deesser=i=0.2:m=0.25:f=0.5,highshelf=f=5500:g=-2"),
    ("clean", "highpass=f=80:p=2,apad=pad_len=1200,afftdn=nr=6:nf=-50,"
     "atrim=start_sample=1200,asetpts=N/SR/TB,"
     "acompressor=threshold=0.125:ratio=1.5:attack=15:release=150:makeup=1"),
    ("broadcast", "highpass=f=80:p=2,apad=pad_len=1200,afftdn=nr=6:nf=-50,"
     "atrim=start_sample=1200,asetpts=N/SR/TB,equalizer=f=3000:t=q:w=0.7:g=1.5,"
     "acompressor=threshold=0.1:ratio=2.5:attack=10:release=150:makeup=1"),
    ("warm", "highpass=f=80:p=2,equalizer=f=200:t=q:w=0.7:g=1.5,highshelf=f=5500:g=-2,"
     "acompressor=threshold=0.125:ratio=1.5:attack=15:release=150:makeup=1"),
])
def test_fixed_filter_table(name: str, expected: str) -> None:
    assert VOICE_FILTERS[name] == expected
    if name in {"none", "clean", "broadcast", "warm"}:
        assert voice_filters(name) == expected
    elif name.startswith("proximity"):
        strength = "strong" if name == "proximity_strong" else "gentle"
        assert voice_filters("none", proximity=strength) == expected
    else:
        assert voice_filters("none", **{name: True}) == expected


@pytest.mark.parametrize("preset", ["none", "clean", "broadcast", "warm"])
def test_repairs_precede_preset_without_duplicate_filters(preset: str) -> None:
    parts = voice_filters(preset, rumble=True, room=True, harshness=True).split(",")
    repairs = ",".join(VOICE_FILTERS[name] for name in ("rumble", "room", "harshness"))
    assert parts[:4] == repairs.split(",")
    assert len(parts) == len(set(parts))
    assert parts.count("highpass=f=80:p=2") == 1
    assert parts.count("highshelf=f=5500:g=-2") == 1
    assert all("makeup=1" in part for part in parts if part.startswith(("agate", "acompressor")))


@pytest.mark.parametrize("preset", ["none", "clean", "broadcast", "warm"])
@pytest.mark.parametrize("proximity", ["gentle", "strong"])
def test_proximity_order_and_supersedes_rumble(preset: str, proximity: str) -> None:
    parts = voice_filters(preset, rumble=True, proximity=proximity, room=True,
                          harshness=True).split(",")
    name = "proximity_strong" if proximity == "strong" else "proximity"
    repairs = ",".join(VOICE_FILTERS[item] for item in (name, "room", "harshness"))
    assert parts[:5] == repairs.split(",")
    assert "highpass=f=80:p=2" not in parts
    assert parts.count("highpass=f=100:p=2") == 1
    assert len(parts) == len(set(parts))


def test_unknown_preset_has_structured_refusal() -> None:
    with pytest.raises(YapError) as caught:
        voice_filters("unknown")
    assert caught.value.code == "voice_chain_invalid"


@pytest.mark.parametrize("proximity", ["medium", True])
def test_invalid_proximity_has_structured_refusal(proximity: object) -> None:
    with pytest.raises(YapError) as caught:
        voice_filters("none", proximity=proximity)
    assert caught.value.error_type == "audio_error"
    assert caught.value.code == "voice_chain_invalid"
    assert caught.value.suggested_action


def _filtered(source: Path, chain: str) -> np.ndarray:
    """Decode a chain directly, without padding that could mask a length error."""
    result = ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(source), "-map", "0:a:0",
                         "-af", chain or "anull", "-f", "f32le", "-"],
                        timeout_s=60, binary=True)
    return np.frombuffer(result.stdout, dtype="<f4")


@pytest.mark.parametrize("preset", ["none", "clean", "broadcast", "warm"])
@pytest.mark.parametrize(("rumble", "room", "harshness"), list(itertools.product(
    (False, True), repeat=3)))
def test_every_chain_keeps_samples_and_zero_lag(tmp_path: Path, preset: str,
                                               rumble: bool, room: bool,
                                               harshness: bool) -> None:
    chain = voice_filters(preset, rumble=rumble, room=room, harshness=harshness)
    _assert_samples_and_zero_lag(tmp_path, chain)


@pytest.mark.parametrize("preset", ["none", "clean", "broadcast", "warm"])
@pytest.mark.parametrize("proximity", ["gentle", "strong"])
def test_proximity_keeps_samples_and_zero_lag(tmp_path: Path, preset: str,
                                              proximity: str) -> None:
    chain = voice_filters(preset, rumble=True, proximity=proximity, room=True, harshness=True)
    _assert_samples_and_zero_lag(tmp_path, chain)


def _assert_samples_and_zero_lag(tmp_path: Path, chain: str) -> None:
    count = RATE + 137  # Deliberately not an FFT block or frame multiple.
    time = np.arange(count) / RATE
    signal = (0.1 * np.sin(2 * np.pi * 220 * time) +
              np.random.default_rng(76).normal(0, 0.06, count)).astype(np.float32)
    # Include a separate marker near the end to exercise the denoiser's flush.
    signal[-100:] += np.random.default_rng(77).normal(0, 0.1, 100).astype(np.float32)
    source = synth.float_audio(tmp_path / "input.wav", signal)
    output = _filtered(source, chain)
    assert len(output) == count
    assert np.isfinite(output).all()
    size = 1 << (2 * count - 1).bit_length()
    correlation = np.fft.irfft(np.fft.rfft(output, size) *
                               np.conj(np.fft.rfft(signal, size)), size)
    assert int(np.argmax(correlation)) == 0
    tail_scores = [np.dot(signal[-100:-20], output[count - 100 + lag:count - 20 + lag])
                   for lag in range(-10, 11)]
    assert np.argmax(tail_scores) - 10 == 0


@pytest.mark.parametrize(("repair", "frequency", "maximum_db"), [
    ("rumble", 40, -12), ("harshness", 7000, -1),
])
def test_repairs_lower_unwanted_tone_energy(tmp_path: Path, repair: str,
                                           frequency: float, maximum_db: float) -> None:
    time = np.arange(2 * RATE) / RATE
    signal = (0.15 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
    source = synth.float_audio(tmp_path / "tone.wav", signal)
    output = _filtered(source, voice_filters("none", **{repair: True}))
    # Measure the settled second half so filter startup does not affect the ratio.
    reduction = 10 * np.log10(np.mean(output[RATE:] ** 2) / np.mean(signal[RATE:] ** 2))
    assert reduction <= maximum_db


@pytest.mark.parametrize(("proximity", "frequency", "maximum_db"), [
    ("gentle", 200, -2.5), ("strong", 200, -4.5),
    ("gentle", 60, -6), ("gentle", 1000, 1), ("strong", 1000, 1),
])
def test_proximity_tone_response(tmp_path: Path, proximity: str,
                                frequency: float, maximum_db: float) -> None:
    time = np.arange(2 * RATE) / RATE
    signal = (0.15 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
    source = synth.float_audio(tmp_path / "tone.wav", signal)
    output = _filtered(source, voice_filters("none", proximity=proximity))
    change = 10 * np.log10(np.mean(output[RATE:] ** 2) / np.mean(signal[RATE:] ** 2))
    if frequency == 1000:
        assert abs(change) < maximum_db
    else:
        assert change <= maximum_db
