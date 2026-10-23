"""Calibration tests for the synthetic fixture generator."""

from __future__ import annotations

import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import synth


def _run_ffmpeg(args: list[str]) -> bytes:
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=300,
        check=True,
    )
    return result.stdout


def _probe(path: Path) -> dict[str, object]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return json.loads(result.stdout)


def _video_stream(path: Path) -> dict[str, object]:
    streams = _probe(path)["streams"]
    assert isinstance(streams, list)
    return next(stream for stream in streams if stream["codec_type"] == "video")


def _audio(path: Path) -> np.ndarray:
    raw = _run_ffmpeg(
        [
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-f",
            "f32le",
            "-",
        ]
    )
    return np.frombuffer(raw, dtype="<f4")


def _click_db(media: np.ndarray, sample: int, sr: int = 48000) -> float:
    """Compare join HF sharpness with the surrounding 99.5th percentile.

    The peak covers plus or minus 2 ms. The reference covers plus or minus 100 ms,
    excluding the central 5 ms. Values at or below 0 dB are no sharper than nearby audio.
    """
    differences = np.abs(np.diff(media.astype(np.float64)))
    window_2 = int(0.002 * sr)
    window_100 = int(0.1 * sr)
    window_5 = int(0.005 * sr)
    near = differences[max(0, sample - window_2) : sample + window_2].max()
    low = max(0, sample - window_100)
    high = min(len(differences), sample + window_100)
    mask = np.ones(high - low, dtype=bool)
    mask[
        max(0, sample - window_5 - low) : max(0, sample + window_5 - low)
    ] = False
    reference = np.percentile(differences[low:high][mask], 99.5) + 1e-12
    return float(20 * np.log10(near / reference + 1e-12))


def _first_gray_frame(path: Path, width: int, height: int) -> np.ndarray:
    raw = _run_ffmpeg(
        [
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-vf",
            f"scale={width}:{height},format=gray",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-",
        ]
    )
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width)


def test_counter_source_without_drawtext(tmp_path: Path, monkeypatch) -> None:
    original = synth._run

    def without_drawtext(command, **kwargs):
        assert not any("drawtext" in arg for arg in command)
        return original(command, **kwargs)

    monkeypatch.setattr(synth, "_run", without_drawtext)
    movie = synth.counter_source(tmp_path / "counter.mov")
    frames = [synth.picture_pixels(movie, n, box=(40, 40, 128, 80)) for n in range(36)]
    assert len({frame.tobytes() for frame in frames}) == 36
    # Read the six binary bars independently of the source filter expression.
    for n, frame in enumerate(frames):
        bits = [frame[40, bit * 16 + 6].mean() > 128 for bit in range(6)]
        assert sum(int(on) << bit for bit, on in enumerate(bits)) == n


def test_default_spans_are_safe() -> None:
    spans = synth.default_spans()
    flashes = synth.flash_times(30, "30/1")
    assert len(spans) == 8
    for index, (start, end) in enumerate(spans):
        assert math.isclose(start * 30, round(start * 30), abs_tol=1e-9)
        assert math.isclose(end * 30, round(end * 30), abs_tol=1e-9)
        assert 2.3 <= end - start <= 2.5
        assert any(start <= float(flash) < end for flash in flashes)
        for edge in (start, end):
            assert all(
                not float(flash) - 0.05 <= edge <= float(flash) + 0.04 + 0.05
                for flash in flashes
            )
        if index:
            gap = start - spans[index - 1][1]
            assert 0.8 <= gap <= 1.2


def test_clean_render_calibrates_to_one_ms(synth_media) -> None:
    measured = synth.oracle(synth_media.path("clean"), synth_media.spans)
    assert measured.events
    assert max(abs(event.offset_ms) for event in measured.events) <= 1.0


def test_perclip_render_has_measurable_drift(synth_media) -> None:
    measured = synth.oracle(synth_media.path("perclip"), synth_media.spans)
    assert measured.slope_ms_per_cut is not None
    assert measured.slope_ms_per_cut >= 10.0
    assert measured.cut_jumps_ms
    assert all(jump >= 10.0 for jump in measured.cut_jumps_ms.values())
    assert measured.events[0].offset_ms == pytest.approx(0.0, abs=1.0)
    assert all(
        jump == pytest.approx(32.0, abs=1.0)
        for jump in measured.cut_jumps_ms.values()
    ), "each join should expose the 1536-sample (32 ms at 48 kHz) AAC overlap"
    assert max(event.offset_ms for event in measured.events) > 60.0


@pytest.mark.parametrize("name", ["clean_shift_samples", "clean_shift_start"])
def test_shift_audio_reads_one_hundred_ms(synth_media, name: str) -> None:
    measured = synth.oracle(synth_media.path(name), synth_media.spans)
    offsets = [event.offset_ms for event in measured.events]
    assert offsets
    assert np.mean(offsets) == pytest.approx(100.0, abs=1.0)
    assert all(offset == pytest.approx(100.0, abs=1.0) for offset in offsets)


def test_rendered_flash_count_matches_source_spans(synth_media) -> None:
    expected = sum(
        1
        for flash in synth.flash_times(30, "30/1")
        if any(start <= float(flash) < end for start, end in synth_media.spans)
    )
    assert len(synth.oracle(synth_media.path("clean")).events) == expected


def test_calib_neighbouring_frames_are_unique(synth_media) -> None:
    raw = _run_ffmpeg(
        [
            "-i",
            str(synth_media.path("calib")),
            "-map",
            "0:v:0",
            "-vf",
            "scale=64:36,format=gray",
            "-frames:v",
            "60",
            "-f",
            "rawvideo",
            "-",
        ]
    )
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 36, 64).astype(np.float32)
    differences = np.mean(np.abs(np.diff(frames, axis=0)), axis=(1, 2))
    assert np.all(differences > 1.0)


def test_calib_background_stays_below_beep_threshold() -> None:
    audio = synth._calib_audio(30, Fraction(30), 48000, beeps=False)
    assert float(np.max(np.abs(audio))) < 0.3


def test_vfr_source_has_nominal_rate_and_audio_offset(synth_media) -> None:
    path = synth_media.cached("vfr", synth.vfr_source, suffix=".mov")
    streams = _probe(path)["streams"]
    picture = next(stream for stream in streams if stream["codec_type"] == "video")
    sound = next(stream for stream in streams if stream["codec_type"] == "audio")
    nominal = Fraction(picture["r_frame_rate"])
    average = Fraction(picture["avg_frame_rate"])
    assert nominal == 30
    assert average != nominal
    assert float(picture["duration"]) * (nominal - average) >= 2
    assert float(sound["start_time"]) - float(picture["start_time"]) == pytest.approx(
        0.06, abs=0.001
    )
    frames = json.loads(synth._process([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path),
    ]))["frames"]
    timestamps = np.asarray([float(frame["best_effort_timestamp_time"]) for frame in frames])
    steps = np.diff(timestamps)
    assert np.all(steps > 0)
    gaps = steps[steps > 1.5 / 30]
    assert len(gaps) == 4
    np.testing.assert_allclose(gaps, 2 / 30, atol=0.001)
    jitter = timestamps - np.round(timestamps * 30) / 30
    assert 0.001 < max(abs(jitter)) <= 0.0035


def test_speech_layout_marks_audible_words_and_silent_gap(synth_media) -> None:
    sentences = ["Bright planets circle slowly", "Gentle rivers carry silver stones"]
    layout: list[synth.WordSpan] = []

    def build(out: Path) -> None:
        _, generated = synth.speech_source(out, sentences)
        layout.extend(generated)

    path = synth_media.cached("speech_layout", build, suffix=".mov")
    audio = _audio(path)
    for word in layout:
        start = max(0, int(word.start * 48000))
        end = min(len(audio), int(word.end * 48000))
        assert end > start
        assert float(np.sqrt(np.mean(audio[start:end] ** 2))) > 0.01
    left = max(word.end for word in layout if word.sentence == 0)
    right = min(word.start for word in layout if word.sentence == 1)
    margin = min(0.15, (right - left) / 3.0)
    gap = audio[int((left + margin) * 48000) : int((right - margin) * 48000)]
    assert len(gap) > 0
    assert float(np.sqrt(np.mean(gap * gap))) < 0.002


def test_reframes_decode_at_expected_shapes(synth_media) -> None:
    center = _video_stream(synth_media.path("crop_center"))
    offcenter = _video_stream(synth_media.path("crop_offcenter"))
    fitted = _video_stream(synth_media.path("fit_vertical"))
    assert (center["width"], center["height"]) == (180, 320)
    assert (offcenter["width"], offcenter["height"]) == (180, 320)
    assert (fitted["width"], fitted["height"]) == (568, 320)
    assert synth.crop_window(320, 180, 0.5) == (108, 0, 102, 180)
    assert synth.crop_window(320, 180, 0.2) == (44, 0, 102, 180)


def test_rotation_metadata_round_trips_upright_picture(synth_media) -> None:
    if not synth.display_rotation_supported():
        pytest.skip("ffmpeg lacks -display_rotation")
    source = synth_media.path("calib_vertical")
    rotated = synth_media.path("rotated_vertical")
    stream = _video_stream(rotated)
    assert int(stream["width"]) > int(stream["height"])
    side_data = stream.get("side_data_list", [])
    assert any(abs(float(item.get("rotation", 0))) == 90 for item in side_data)
    original_frame = _first_gray_frame(source, 18, 32)
    rotated_frame = _first_gray_frame(rotated, 18, 32)
    difference = np.mean(
        np.abs(original_frame.astype(np.float32) - rotated_frame.astype(np.float32))
    )
    assert difference < 8.0


def test_click_is_planted_only_at_cut_three(synth_media) -> None:
    perclip_audio = _audio(synth_media.path("perclip"))
    click_path = synth_media.path("perclip_click3")
    click_audio = _audio(click_path)
    measured = synth.oracle(click_path, synth_media.spans)
    planned_cut_time = sum(end - start for start, end in synth_media.spans[:3])
    expected_click_time = planned_cut_time + measured.segment_offsets_ms[3] / 1000.0

    differences = np.abs(np.diff(click_audio))
    peak_index = int(np.argmax(differences))
    peak_time = (peak_index + 1) / 48000.0
    assert peak_time == pytest.approx(expected_click_time, abs=0.25)
    assert _click_db(click_audio, peak_index) >= 6.0
    assert _click_db(perclip_audio, peak_index) <= 0.0

    window_start = max(0, int((expected_click_time - 0.25) * 48000))
    window_end = min(len(differences), int((expected_click_time + 0.25) * 48000))
    candidates = differences.copy()
    candidates[window_start:window_end] = -np.inf
    separated_peaks: list[int] = []
    separation = int(0.01 * 48000)
    for _ in range(20):
        sample = int(np.argmax(candidates))
        assert np.isfinite(candidates[sample])
        separated_peaks.append(sample)
        low = max(0, sample - separation + 1)
        high = min(len(candidates), sample + separation)
        candidates[low:high] = -np.inf
    assert all(_click_db(click_audio, sample) < 6.0 for sample in separated_peaks)


def test_click_plant_moves_source_edges_by_at_most_five_ms(synth_media) -> None:
    spans = synth_media.spans
    adjusted, step = synth._click_adjusted_spans(synth_media.path("calib"), spans, 3)
    assert step >= 0.2
    assert abs(adjusted[2][1] - spans[2][1]) <= 0.005
    assert abs(adjusted[3][0] - spans[3][0]) <= 0.005


def test_tests_never_need_drawtext() -> None:
    """Homebrew's ffmpeg on CI has no drawtext; fixtures stamp pictures with drawbox instead."""
    tests = Path(__file__).parent
    users = [path.name for path in tests.rglob("*.py")
             if path.name != "test_synth.py" and "drawtext=" in path.read_text(encoding="utf-8")]
    assert users == []
