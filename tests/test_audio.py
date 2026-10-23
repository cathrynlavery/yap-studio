"""Calibration tests for audio primitives."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import synth

from yap_studio import audio, ffmpeg
from yap_studio.errors import InputError, ProcessingError, ValidationError


def _sample(seconds: float, sr: int) -> int:
    return math.floor(seconds * sr + 0.5)


def test_decode_selected_audio_ordinal_uses_its_rate_channels_and_clock(tmp_path: Path) -> None:
    source = tmp_path / "streams.mkv"
    ffmpeg.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:d=1",
        "-itsoffset", "0.12", "-f", "lavfi", "-i",
        "aevalsrc=0.1*sin(2*PI*880*t)|0.2*sin(2*PI*660*t):s=48000:d=1",
        "-map", "0:a:0", "-map", "1:a:0", "-c:a", "pcm_s16le", str(source),
    ], timeout_s=60)
    rate, selected = audio.decode(source, stream=1, mono=False)
    assert rate == 48_000
    assert selected.shape == (53_760, 2)
    assert not np.any(selected[:5760])
    assert np.max(np.abs(selected[5760:])) > 0.1
    assert audio.decode(source)[0] == 16_000
    _, first = audio.decode(source, stream=0)
    np.testing.assert_array_equal(first, audio.decode(source)[1])
    with pytest.raises(InputError, match="no audio stream"):
        audio.decode(source, stream=2)


@pytest.fixture(scope="module")
def oracle_audio_shifts(synth_media: object) -> dict[str, float]:
    names = ("clean", "clean_shift_samples", "clean_shift_start")
    mean_offsets_ms: dict[str, float] = {}
    for name in names:
        measured = synth.oracle(
            synth_media.path(name),  # type: ignore[attr-defined]
            synth_media.spans,  # type: ignore[attr-defined]
        )
        assert measured.events
        mean_offsets_ms[name] = float(
            np.mean([event.offset_ms for event in measured.events])
        )
    clean_offset_ms = mean_offsets_ms["clean"]
    return {
        name: (mean_offset_ms - clean_offset_ms) / 1000.0
        for name, mean_offset_ms in mean_offsets_ms.items()
    }


@pytest.mark.parametrize("sr", [48_000, 16_000])
def test_locate_calibrated_audio_offsets(
    synth_media: object, oracle_audio_shifts: dict[str, float], sr: int
) -> None:
    source_path = synth_media.path("calib")  # type: ignore[attr-defined]
    _, source = audio.decode(source_path, sr=sr)
    renders = ("clean", "clean_shift_samples", "clean_shift_start")
    for name in renders[1:]:
        assert oracle_audio_shifts[name] == pytest.approx(0.1, abs=0.001)
    decoded = {
        name: audio.decode(synth_media.path(name), sr=sr)[1]  # type: ignore[attr-defined]
        for name in renders
    }
    for segment, (start_s, end_s) in enumerate(synth.default_spans()):
        if end_s - start_s < 1.0:
            continue
        middle_s = (start_s + end_s) / 2.0
        first = _sample(middle_s, sr)
        needle = source[first : first + _sample(0.5, sr)]
        expected = segment * 2.4 + (middle_s - start_s)
        for name in renders:
            true_shift_s = oracle_audio_shifts[name]
            located = audio.locate(
                needle,
                decoded[name],
                sr,
                expect_s=expected + true_shift_s,
                search_s=0.35,
            )
            assert located.time_s - expected == pytest.approx(
                true_shift_s, abs=0.0001
            )
            assert located.ambiguity < 0.5


def test_periodic_audio_is_ambiguous() -> None:
    sr = 16_000
    signal = np.zeros(3 * sr, dtype=np.float32)
    length = _sample(0.04, sr)
    beep = np.sin(2 * np.pi * 1000 * np.arange(length) / sr).astype(np.float32)
    for first in range(0, len(signal) - length, _sample(0.1, sr)):
        signal[first : first + length] = beep
    needle = signal[_sample(0.5, sr) : _sample(1.0, sr)]
    located = audio.locate(needle, signal, sr, expect_s=1.5, search_s=0.45)
    assert located.ambiguity > 0.8


def test_click_and_raised_cosine_splice(synth_media: object) -> None:
    sr, source = audio.decode(synth_media.path("calib"))  # type: ignore[attr-defined]
    spans, planted_step = synth._click_adjusted_spans(  # noqa: SLF001
        synth_media.path("calib"),  # type: ignore[attr-defined]
        synth.default_spans(),
        3,
    )
    assert planted_step >= 0.2
    left_end = _sample(spans[2][1], sr)
    right_start = _sample(spans[3][0], sr)
    width = _sample(0.15, sr)
    pieces = [(left_end - width, left_end), (right_start, right_start + width)]
    hard, hard_joins = audio.splice(source, pieces, fade=0)
    smooth, smooth_joins = audio.splice(source, pieces, fade=_sample(0.005, sr))
    assert audio.click_db(hard, hard_joins[0], sr) >= 6.0
    assert audio.click_db(smooth, smooth_joins[0], sr) <= 0.0


def test_pause_at_inserted_silence() -> None:
    sr = 48_000
    rng = np.random.default_rng(20260928)
    signal = (0.1 * rng.standard_normal(sr)).astype(np.float32)
    first = _sample(0.5, sr)
    silence = np.zeros(_sample(0.021, sr), dtype=np.float32)
    signal = np.concatenate((signal[:first], silence, signal[first:]))
    middle = first + len(silence) // 2
    assert audio.pause_at(signal, middle, sr) == pytest.approx(0.021, abs=0.005)


def test_decode_applies_stream_start_offset(
    synth_media: object, oracle_audio_shifts: dict[str, float]
) -> None:
    sr, clean = audio.decode(synth_media.path("clean"))  # type: ignore[attr-defined]
    shifted_sr, shifted = audio.decode(  # type: ignore[attr-defined]
        synth_media.path("clean_shift_start"), sr=sr
    )
    assert shifted_sr == sr
    assert np.max(np.abs(shifted[: _sample(0.075, sr)])) < 1e-6
    needle = clean[_sample(1.0, sr) : _sample(1.5, sr)]
    expected = 1.0 + oracle_audio_shifts["clean_shift_start"]
    located = audio.locate(needle, shifted, sr, expect_s=expected, search_s=0.2)
    assert located.time_s == pytest.approx(expected, abs=0.0001)


def test_decode_maps_selected_absolute_audio_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "selected-audio.mp4"
    path.write_bytes(b"media")
    monkeypatch.setattr(
        ffmpeg,
        "probe",
        lambda *args, **kwargs: SimpleNamespace(
            has_audio=True,
            audio_index=7,
            sample_rate=48_000,
            channels=1,
            audio_offset_s=0.0,
            format_start_s=0.0,
        ),
    )
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(command)
        if command[0] == "ffprobe":
            return SimpleNamespace(stdout="frame,0,1\nstream,1/48000\n")
        return SimpleNamespace(stdout=np.zeros(1, dtype="<f4").tobytes())

    monkeypatch.setattr(ffmpeg, "run", run)
    _, decoded = audio.decode(path)

    assert decoded.shape == (1,)
    assert commands[0][commands[0].index("-map") + 1] == "0:7"
    assert commands[1][commands[1].index("-select_streams") + 1] == "7"


GAP_AT_SAMPLE = math.ceil(96_000 / 1024) * 1024


def _on_timestamps(reference: np.ndarray, at: int, gap: int) -> np.ndarray:
    """Place continuous reference audio where the gapped render's timestamps put it."""
    return np.concatenate((reference[:at], np.zeros(gap), reference[at:]))


def test_decode_fills_timestamp_gap_with_silence(synth_media: object, aac_tail: int) -> None:
    render = synth_media.path("audio_gap")  # type: ignore[attr-defined]
    sr, decoded = audio.decode(render)
    _, reference = synth.read_wav(render.with_suffix(".wav"))
    assert sr == 48_000
    # The timestamp span: 4 s of content plus the 1600-sample gap (and any encoder tail).
    assert len(decoded) == 4 * sr + 1600 + aac_tail
    assert not decoded[GAP_AT_SAMPLE : GAP_AT_SAMPLE + 1600].any()
    placed = _on_timestamps(reference, GAP_AT_SAMPLE, 1600)
    for start_s in (0.5, 2.5, 3.2):
        window = slice(_sample(start_s, sr), _sample(start_s + 0.6, sr))
        lag_s, corr = audio.lag(placed[window], decoded[window], sr, max_lag_s=0.05)
        assert lag_s == 0, start_s
        assert corr > 0.9, start_s
    clock = audio.timeline(render)
    assert clock.fills == ((GAP_AT_SAMPLE, 1600),)
    assert clock.fill_samples == 1600


def test_resampled_and_sliced_decodes_keep_the_gap(synth_media: object, aac_tail: int) -> None:
    render = synth_media.path("audio_gap")  # type: ignore[attr-defined]
    sr, decoded = audio.decode(render, sr=16_000)
    assert sr == 16_000
    assert abs(len(decoded) - round((4 * 48_000 + 1600 + aac_tail) / 3)) <= 2
    first = round(GAP_AT_SAMPLE / 3)
    assert not decoded[first + 2 : first + 531].any()
    assert np.abs(decoded[first + 540 : first + 1000]).max() > 0.01
    _, whole = audio.decode(render)
    _, tail = audio.decode(render, start_s=3.0, dur_s=0.5)
    np.testing.assert_array_equal(tail, whole[_sample(3.0, 48_000) : _sample(3.5, 48_000)])


def test_decode_refuses_audio_it_cannot_place(
    synth_media: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    render = synth_media.path("audio_gap")  # type: ignore[attr-defined]
    real = audio._clock_for(render, ffmpeg.probe(render, count_frames=False))
    wrong = replace(real, decoded_samples=real.decoded_samples + 1024)
    monkeypatch.setattr(audio, "_clock_for", lambda *args: wrong)
    with pytest.raises(ProcessingError) as error:
        audio.decode(render)
    assert error.value.code == "audio_clock_mismatch"


def test_overlapping_timestamps_keep_every_decoded_sample(synth_media: object) -> None:
    # Per-clip AAC joins step the timestamps back; the decoded sound plays in order.
    render = synth_media.path("perclip")  # type: ignore[attr-defined]
    clock = audio.timeline(render)
    assert clock.overlap_samples > 0 and clock.gap_samples == 0
    assert clock.fills == ()
    _, decoded = audio.decode(render)
    assert len(decoded) == clock.decoded_samples > clock.span_samples


def test_decode_slice_pads_past_end(synth_media: object) -> None:
    path = synth_media.path("calib")  # type: ignore[attr-defined]
    sr, whole = audio.decode(path, sr=16_000)
    _, tail = audio.decode(path, sr=sr, start_s=29.9, dur_s=0.2)
    assert len(tail) == _sample(0.2, sr)
    assert np.array_equal(tail[: _sample(0.1, sr)], whole[_sample(29.9, sr) :])
    assert np.all(tail[_sample(0.1, sr) :] == 0)


def test_audio_argument_and_stream_errors(synth_media: object, tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        audio.decode(synth_media.path("calib"), sr=0)  # type: ignore[attr-defined]
    video_only = synth_media.cached(  # type: ignore[attr-defined]
        "audio-video-only",
        lambda out: synth._run(  # noqa: SLF001
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=size=16x16:duration=0.1",
                "-map",
                "0:v:0",
                "-an",
                str(out),
            ]
        ),
    )
    with pytest.raises(InputError, match="no audio stream"):
        audio.decode(video_only)
