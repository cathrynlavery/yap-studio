"""Synthetic sample clocks, mix automation and audio-policy acceptance."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import synth
from test_assemble import project  # noqa: F401
from test_assemble import refuse_unexpected_matte_work as refuse_unexpected_matte_work
from test_cli_captions import fake_layers  # noqa: F401

from yap_studio import ffmpeg, receipt
from yap_studio.audio_chain import voice_filters
from yap_studio.project.model import AUDIO_DEFAULTS
from yap_studio.project.receipts import projection_sha256
from yap_studio.project.status import ProjectContext
from yap_studio.stages import assemble as assembly
from yap_studio.stages import assemble_audio as stage
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline
from yap_studio.style import resolve


@pytest.mark.parametrize(("container_delta", "padding", "passed"), [
    (0, 0, True), (0, 896, True), (-1, 896, False), (1, 896, False),
    (0, 1024, False), (0, -1, False),
])
def test_delivery_samples_allow_only_terminal_decoder_padding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    container_delta: int, padding: int, passed: bool,
) -> None:
    frames = 121
    expected = frames * stage.SAMPLES_PER_FRAME
    monkeypatch.setattr(ffmpeg, "probe", lambda _: SimpleNamespace(
        channels=2, audio_duration_s=(expected + container_delta) / stage.RATE,
        audio_codec="aac", sample_rate=stage.RATE, audio_start_s=0,
    ))
    monkeypatch.setattr(stage, "decode", lambda source, commands, *, channels:
                        np.zeros((expected + padding, channels), dtype=np.float32))
    monkeypatch.setattr(stage, "packet_samples", lambda *args: [1024, 64])
    monkeypatch.setattr(stage, "measure", lambda *args: stage.Loudness(None, None, None))
    checks, _ = stage.delivery_asserts(tmp_path / "final.mp4", frames=frames,
                                      policy={"mode": "preserve"}, commands=[])
    samples = next(check for check in checks if check["id"] == "audio.samples")
    assert samples["pass"] is passed, samples
    assert samples["expected"] == expected
    assert samples["container_samples"] == expected + container_delta
    assert samples["actual"] == expected + padding
    assert samples["terminal_padding"] == padding


@pytest.mark.parametrize("audible", [False, True])
def test_card_clip_audio_window_and_muted_inputs(tmp_path: Path, audible: bool) -> None:
    count = 120 * stage.SAMPLES_PER_FRAME
    voice = np.full(count, 0.01, dtype=np.float32)
    synth.float_audio(tmp_path / "cut/voice.wav", voice)
    clip = synth.float_audio(tmp_path / "clip.wav", np.full(count, 0.1, dtype=np.float32))
    entry = {"id": "agent-video:clip:0", "path": clip.name, "in_s": 0.2, "speed": 2,
             "start_frame": 30, "frames": 30}
    if audible:
        entry["gain_db"] = -6.020599913
    result = stage.render_audio(
        tmp_path, tmp_path / "audio.m4a", work_dir=tmp_path / "work",
        timeline=ResolvedTimeline((), (), (), 120, 3), policy={"mode": "preserve"},
        music=resolve("preset:clean_minimal").data["music"], clips=[entry],
    )
    pcm = stage.decode(tmp_path / "work/mix.wav", [], channels=1)[:, 0]
    np.testing.assert_allclose(pcm[:30 * 1600], voice[:30 * 1600], atol=1e-6)
    np.testing.assert_allclose(pcm[60 * 1600:120 * 1600], voice[60 * 1600:], atol=1e-6)
    if audible:
        np.testing.assert_allclose(pcm[35 * 1600:55 * 1600], 0.06, atol=5e-5)
        assert any("atempo=2" in arg for command in result.commands for arg in command)
    else:
        np.testing.assert_array_equal(pcm[:count], voice)
        assert not any(str(clip) in command for command in result.commands)
    assert not pcm[count:].any()


@pytest.mark.parametrize("hold", [0, 17, 45])
def test_preserve_voice_pcm_is_sample_exact_with_zero_lag(tmp_path: Path, hold: int) -> None:
    synth.loudness_master(tmp_path / "cut/voice.wav")
    timeline = ResolvedTimeline((), (), (), 120, hold)
    out = tmp_path / "audio.m4a"
    result = stage.render_audio(tmp_path, out, work_dir=tmp_path / "work", timeline=timeline,
                                policy={"mode": "preserve"},
                                music=resolve("preset:clean_minimal").data["music"])
    before = stage.decode(tmp_path / "cut/voice.wav", [], channels=1)
    pcm = stage.decode(tmp_path / "work/mix.wav", [], channels=1)
    np.testing.assert_array_equal(pcm[:len(before)], before)
    assert not np.any(pcm[len(before):])
    assert result.samples == (120 + hold) * 1600
    assert result.input_voice.integrated_lufs == pytest.approx(-16, abs=0.1)
    checks, output = stage.delivery_asserts(out, frames=120 + hold,
                                          policy={"mode": "preserve"}, commands=[])
    assert all(check["pass"] for check in checks)
    assert output.integrated_lufs == pytest.approx(result.pcm_output.integrated_lufs, abs=0.1)
    if hold == 0:
        assert output.integrated_lufs == pytest.approx(-16, abs=0.1)
    decoded = stage.decode(out, [], channels=1)[:len(before)]
    # A search around zero measures the lag independently of loudness or PCM equality.
    scores = [np.dot(before[100:10000, 0], decoded[100 + shift:10000 + shift, 0])
              for shift in range(-20, 21)]
    assert np.argmax(scores) - 20 == 0
    assert not any("loudnorm=" in arg or "alimiter=" in arg
                   for command in result.commands for arg in command)


@pytest.mark.parametrize("target", [-14, -18])
def test_constant_gain_loudnorm_target_and_true_peak(tmp_path: Path, target: float) -> None:
    synth.loudness_master(tmp_path / "cut/voice.wav")
    timeline = ResolvedTimeline((), (), (), 120, 45)
    policy = {**AUDIO_DEFAULTS, "integrated_lufs": target}
    out = tmp_path / "audio.m4a"
    result = stage.render_audio(tmp_path, out, work_dir=tmp_path / "work", timeline=timeline,
                                policy=policy, music=resolve("preset:clean_minimal").data["music"])
    checks, output = stage.delivery_asserts(out, frames=165, policy=policy, commands=[])
    assert all(check["pass"] for check in checks), checks
    assert output.integrated_lufs == pytest.approx(target, abs=0.5)
    assert output.true_peak_dbtp <= -1.7
    passes = [arg for command in result.commands for arg in command if "loudnorm=" in arg]
    assert len(passes) == 1 and "print_format=json" in passes[0]
    applications = [arg for command in result.commands for arg in command if "volume=" in arg]
    assert applications and all("alimiter=" in application for application in applications)
    assert all("asetpts=N/SR/TB" in application and "atrim=end_sample=264000" in application
               for application in applications)
    assert result.normalization["mode"] == "constant_gain_limiter"
    assert result.normalization["limiter_dbtp"] == -2.2
    assert f"volume={result.normalization['gain_db']:.12f}dB" in applications[-1]
    pcm = stage.decode(tmp_path / f"work/normalised-{result.normalization['iterations'] - 1}.wav",
                       [], channels=1)
    assert len(pcm) == 264000 and not pcm[192000:].any()
    packets = stage.packet_samples(out, [])
    assert all(value == 1024 for value in packets[:-1])


def test_music_dry_open_windows_ducking_and_hold(tmp_path: Path) -> None:
    count = 6 * stage.RATE
    time = np.arange(count) / stage.RATE
    voice = np.zeros(count, dtype=np.float32)
    voice[stage.RATE:2 * stage.RATE] = 0.4 * np.sin(2 * np.pi * 220 * time[:stage.RATE])
    synth.float_audio(tmp_path / "cut/voice.wav", voice)
    bed = np.column_stack([0.1 * np.sin(2 * np.pi * 880 * time)] * 2)
    bed = np.vstack([bed, np.zeros((stage.RATE, 2))])
    source = synth.float_audio(tmp_path / "bed.wav", bed)
    window = ResolvedElement("agent-level", "music_level", "agent", 120, 150, "front",
                             {"level": 0.25})
    timeline = ResolvedTimeline((window,), (), (), 180, 30)
    music = {**resolve("preset:clean_minimal").data["music"], "level": 0.5,
             "dry_open_s": 0.7}
    result = stage.render_audio(tmp_path, tmp_path / "audio.m4a", work_dir=tmp_path / "work",
                                timeline=timeline, policy={"mode": "preserve"}, music=music,
                                bed=source)
    mix = stage.decode(tmp_path / "work/mix.wav", [], channels=2)
    assert result.click_stem == tmp_path / "work/voice-stem.wav"
    stem = stage.decode(result.click_stem, [], channels=2)
    assert len(stem) == result.samples == 7 * stage.RATE
    np.testing.assert_allclose(stem[:count], np.repeat(voice[:, None], 2, axis=1), atol=1e-6)
    assert not stem[count:].any()
    music_only = mix[:count] - voice[:, None]
    assert not music_only[:round(0.7 * stage.RATE)].any()
    assert not mix[count:].any()

    def rms(first: float, last: float) -> float:
        return float(np.sqrt(np.mean(music_only[round(first * stage.RATE):
                                               round(last * stage.RATE)] ** 2)))

    assert rms(1.5, 1.9) < rms(3, 3.5) * 0.5
    assert rms(4.1, 4.9) == pytest.approx(rms(3, 3.5) * 0.25, rel=0.01)


def test_click_stem_absent_without_bed(tmp_path: Path) -> None:
    synth.float_audio(tmp_path / "cut/voice.wav", np.zeros(stage.RATE, dtype=np.float32))
    result = stage.render_audio(tmp_path, tmp_path / "audio.m4a", work_dir=tmp_path / "work",
                                timeline=ResolvedTimeline((), (), (), 30, 0),
                                policy={"mode": "preserve"},
                                music=resolve("preset:clean_minimal").data["music"])
    assert result.click_stem is None
    assert not (tmp_path / "work/voice-stem.wav").exists()


def test_click_stem_uses_delivery_loudnorm_gain(tmp_path: Path) -> None:
    synth.loudness_master(tmp_path / "cut/voice.wav")
    samples = 120 * stage.SAMPLES_PER_FRAME
    bed = synth.float_audio(tmp_path / "bed.wav", np.zeros((samples, 2), dtype=np.float32))
    result = stage.render_audio(tmp_path, tmp_path / "audio.m4a", work_dir=tmp_path / "work",
                                timeline=ResolvedTimeline((), (), (), 120, 0),
                                policy=AUDIO_DEFAULTS,
                                music=resolve("preset:clean_minimal").data["music"], bed=bed)
    stem = stage.decode(result.click_stem, [], channels=2)
    normalised = tmp_path / "work" / f"normalised-{result.normalization['iterations'] - 1}.wav"
    delivered = stage.decode(normalised, [], channels=2)
    np.testing.assert_allclose(stem, delivered, atol=1e-5)


@pytest.mark.parametrize("audible", [False, True])
def test_insert_gain_in_point_and_speed(tmp_path: Path, audible: bool) -> None:
    synth.float_audio(tmp_path / "cut/voice.wav", np.zeros(144000, dtype=np.float32))
    source = synth.voiced_source(tmp_path / "insert.mov", [(0.5, 1.5)], seconds=3)
    from yap_studio import receipt

    item = ResolvedElement("broll-one", "insert", "broll", 30, 60, "front",
                           {"media": {"path": source.name, "sha256": receipt.sha256_file(source),
                                      "in_s": 0.5, "speed": 2},
                            **({"audio": {"gain_db": -6}} if audible else {})})
    timeline = ResolvedTimeline((item,), (), (), 90, 5)
    stage.render_audio(tmp_path, tmp_path / "audio.m4a", work_dir=tmp_path / "work",
                       timeline=timeline, policy={"mode": "preserve"},
                       music=resolve("preset:clean_minimal").data["music"])
    pcm = stage.decode(tmp_path / "work/mix.wav", [], channels=1)[:, 0]
    assert not pcm[:48000].any() and not pcm[96000:].any()
    if audible:
        assert np.max(np.abs(pcm[48000:68000])) == pytest.approx(0.1 * 10 ** (-6 / 20), abs=0.002)
        assert not pcm[80000:].any()
    else:
        assert not pcm.any()


def test_background_clip_audio_never_enters_mix(tmp_path: Path) -> None:
    synth.loudness_master(tmp_path / "cut/voice.wav")
    source = synth.voiced_source(tmp_path / "background.mov", [(0, 3)], seconds=4)
    assert ffmpeg.probe(source).has_audio
    assert np.max(np.abs(stage.decode(source, [], channels=1))) > 0.05
    item = ResolvedElement("broll-page", "background", "broll", 0, 120, "front", {
        "media": {"path": source.name, "sha256": receipt.sha256_file(source),
                  "in_s": 0, "speed": 1},
    })
    result = stage.render_audio(
        tmp_path, tmp_path / "audio.m4a", work_dir=tmp_path / "work",
        timeline=ResolvedTimeline((item,), (), (), 120, 0), policy={"mode": "preserve"},
        music=resolve("preset:clean_minimal").data["music"],
    )
    np.testing.assert_array_equal(stage.decode(tmp_path / "work/mix.wav", [], channels=1),
                                  stage.decode(tmp_path / "cut/voice.wav", [], channels=1))
    assert all(str(source) not in command for command in result.commands)


def test_short_voice_and_non_silent_bed_refused(tmp_path: Path) -> None:
    synth.float_audio(tmp_path / "cut/voice.wav", np.zeros(2000, dtype=np.float32))
    timeline = ResolvedTimeline((), (), (), 90, 5)
    with pytest.raises(stage.AudioError, match="sample count"):
        stage.render_audio(tmp_path, tmp_path / "audio.m4a", work_dir=tmp_path / "work",
                           timeline=timeline, policy={"mode": "preserve"}, music={})


def test_default_style_is_byte_identical_to_original_level_mix(tmp_path: Path) -> None:
    """Compare to the original sidechain graph and its once-encoded delivery."""
    count, samples = 4 * stage.RATE, 5 * stage.RATE
    time = np.arange(count) / stage.RATE
    voice = (0.2 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    voice[stage.RATE:2 * stage.RATE] = 0
    voice_path = synth.float_audio(tmp_path / "cut/voice.wav", voice)
    bed = np.vstack([np.column_stack([0.1 * np.sin(2 * np.pi * 880 * time)] * 2),
                     np.zeros((stage.RATE, 2))]).astype(np.float32)
    bed_path = synth.float_audio(tmp_path / "bed.wav", bed)
    window = ResolvedElement("agent-level", "music_level", "agent", 60, 90, "front",
                             {"level": 0.25})
    timeline = ResolvedTimeline((window,), (), (), 120, 30)
    style = resolve("preset:clean_minimal").data
    out = tmp_path / "audio.m4a"
    result = stage.render_audio(tmp_path, out, work_dir=tmp_path / "work", timeline=timeline,
                                policy={"mode": "preserve"}, music=style["music"],
                                voice=style["voice"], bed=bed_path)
    legacy_bed = bed * np.float32(0.06)
    legacy_bed[:33600] = 0
    legacy_bed[96000:144000] *= 0.25
    raw = tmp_path / "legacy-bed.f32le"
    raw.write_bytes(legacy_bed.tobytes())
    legacy_pcm = tmp_path / "legacy.wav"
    filters = (
        f"[0:a:0]pan=stereo|c0=c0|c1=c0,asetpts=N/SR/TB,apad=whole_len={samples},"
        f"atrim=end_sample={samples}[voice];[voice]asplit=2[voice_mix][key];"
        "[1:a:0]asetpts=N/SR/TB[bed];"
        "[bed][key]sidechaincompress=threshold=0.03:ratio=6:attack=10:"
        "release=320:makeup=1[ducked];"
        f"[ducked]atrim=end_sample={count},apad=whole_len={samples},"
        f"atrim=end_sample={samples}[music];"
        "[voice_mix][music]amix=inputs=2:normalize=0:dropout_transition=0,"
        f"atrim=end_sample={count},apad=whole_len={samples},"
        f"atrim=end_sample={samples},asetpts=N/SR/TB[mix]"
    )
    ffmpeg.run(["ffmpeg", "-v", "error", "-filter_complex_threads", "1", "-i", str(voice_path),
                "-f", "f32le", "-ar", str(stage.RATE), "-ac", "2", "-i", str(raw),
                "-filter_complex", filters, "-map", "[mix]", "-c:a", "pcm_f32le",
                "-ar", str(stage.RATE), str(legacy_pcm)], timeout_s=60)
    np.testing.assert_array_equal(stage.decode(tmp_path / "work/mix.wav", [], channels=2),
                                  stage.decode(legacy_pcm, [], channels=2))
    legacy_out = tmp_path / "legacy.m4a"
    ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(legacy_pcm), "-map", "0:a:0",
                "-c:a", "aac", "-b:a", "256k", "-ar", str(stage.RATE),
                "-movie_timescale", str(stage.RATE), str(legacy_out)], timeout_s=60)
    assert out.read_bytes() == legacy_out.read_bytes()
    assert result.to_dict()["voice"] == {"preset": "none", "repairs": [], "filters": ""}
    assert result.to_dict()["music"] == {"duck_mode": "level"}


@pytest.mark.parametrize("voice", [{"preset": "clean"}, {"preset": "broadcast"},
                                   {"preset": "warm"}, {"repair_rumble": True},
                                   {"repair_proximity": "gentle"}, {"repair_room": True},
                                   {"repair_harshness": True}])
def test_preserve_refuses_voice_chain_before_media_work(tmp_path: Path, voice: dict,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_probe(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("No media commands should run for a refused voice chain.")

    monkeypatch.setattr(ffmpeg, "probe", unexpected_probe)
    monkeypatch.setattr(ffmpeg, "run", unexpected_probe)
    with pytest.raises(stage.AudioError) as caught:
        stage.render_audio(tmp_path, tmp_path / "audio.m4a", work_dir=tmp_path / "work",
                           timeline=ResolvedTimeline((), (), (), 120, 30),
                           policy={"mode": "preserve"}, music={}, voice=voice)
    assert caught.value.code == "voice_chain_preserve"
    assert "loudnorm" in caught.value.suggested_action
    assert "voice preset none" in caught.value.suggested_action
    assert not (tmp_path / "work").exists()
    assert not (tmp_path / "audio.m4a").exists()


def test_broadcast_runs_before_loudnorm_and_passes_delivery(tmp_path: Path) -> None:
    synth.loudness_master(tmp_path / "cut/voice.wav")
    out = tmp_path / "audio.m4a"
    voice = {"preset": "broadcast", "repair_rumble": True, "repair_room": True,
             "repair_proximity": "strong", "repair_harshness": True}
    result = stage.render_audio(tmp_path, out, work_dir=tmp_path / "work",
                                timeline=ResolvedTimeline((), (), (), 120, 45),
                                policy=AUDIO_DEFAULTS,
                                music=resolve("preset:clean_minimal").data["music"], voice=voice)
    checks, output = stage.delivery_asserts(out, frames=165, policy=AUDIO_DEFAULTS, commands=[])
    assert all(check["pass"] for check in checks), checks
    assert output.integrated_lufs == pytest.approx(-14, abs=0.5)
    assert output.true_peak_dbtp <= -1.7
    chain = voice_filters("broadcast", rumble=True, proximity="strong", room=True,
                          harshness=True)
    assert result.to_dict()["voice"] == {"preset": "broadcast",
                                        "repairs": ["rumble", "proximity_strong", "room",
                                                    "harshness"],
                                        "filters": chain}
    graph_index = next(i for i, command in enumerate(result.commands)
                       if "-filter_complex" in command)
    graph = result.commands[graph_index]
    expression = graph[graph.index("-filter_complex") + 1]
    assert f"{chain},asetpts=N/SR/TB,apad=" in expression
    assert all(i > graph_index for i, command in enumerate(result.commands)
               if any("loudnorm=" in arg for arg in command))
    pcm = stage.decode(tmp_path / "work/mix.wav", [], channels=1)
    original = stage.decode(tmp_path / "cut/voice.wav", [], channels=1)
    assert len(pcm) == 264000 and not pcm[192000:].any()
    assert not np.array_equal(original, pcm[:len(original)])


@pytest.mark.parametrize("depth", [-12, -24, 0])
def test_word_ducking_depth_pause_ramps_and_existing_envelope(tmp_path: Path, depth: float) -> None:
    count, hold = 6 * stage.RATE, stage.RATE
    time = np.arange(count) / stage.RATE
    voice = (0.05 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    synth.float_audio(tmp_path / "cut/voice.wav", voice)
    # A constant bed permits per-sample gain measurement, including ramp boundaries.
    bed = np.vstack([np.full((count, 2), 0.1, dtype=np.float32),
                     np.zeros((hold, 2), dtype=np.float32)])
    bed_path = synth.float_audio(tmp_path / "bed.wav", bed)
    window = ResolvedElement("agent-level", "music_level", "agent", 120, 150, "front",
                             {"level": 0.25})
    music = {**resolve("preset:clean_minimal").data["music"], "duck_mode": "words",
             "duck_depth_db": depth, "duck_attack_ms": 100, "duck_release_ms": 200,
             "level": 0.5, "dry_open_s": 0.5}
    result = stage.render_audio(tmp_path, tmp_path / "audio.m4a", work_dir=tmp_path / "work",
                                timeline=ResolvedTimeline((window,), (), (), 180, 30),
                                policy={"mode": "preserve"}, music=music, bed=bed_path,
                                cut_words={"time_base": "output", "words": [
                                    {"start": 1.0, "end": 2.0}]})
    mix = stage.decode(tmp_path / "work/mix.wav", [], channels=2)
    actual = mix[:count] - voice[:, None]
    np.testing.assert_allclose(actual[:24000], 0, atol=2e-8)
    np.testing.assert_allclose(actual[48000:96000], 0.05 * 10 ** (depth / 20), atol=2e-8)
    # Even continuous voice in the pause must not trigger level-driven ducking.
    np.testing.assert_allclose(actual[120000:168000], 0.05, atol=2e-8)
    np.testing.assert_allclose(actual[192000:240000], 0.0125, atol=2e-8)
    attack_samples, release_samples = 4800, 9600
    attack_db = depth * np.arange(attack_samples) / attack_samples
    release_db = depth * (1 - np.arange(release_samples) / release_samples)
    np.testing.assert_allclose(actual[43200:48000, 0], 0.05 * 10 ** (attack_db / 20), atol=2e-8)
    np.testing.assert_allclose(actual[96000:105600, 0], 0.05 * 10 ** (release_db / 20), atol=2e-8)
    assert actual[43200, 0] == pytest.approx(0.05)
    assert actual[105600, 0] == pytest.approx(0.05)
    assert not mix[count:].any() and len(mix) == count + hold
    assert result.to_dict()["music"] == {"duck_mode": "words", "ducked_spans": 1}
    assert not any("sidechaincompress=" in arg for command in result.commands for arg in command)


@pytest.mark.parametrize(("gap_ms", "spans"), [(250, 1), (200, 2), (0, 2)])
def test_word_ducking_merges_only_gaps_below_limit(tmp_path: Path, gap_ms: int, spans: int) -> None:
    synth.float_audio(tmp_path / "cut/voice.wav", np.zeros(3 * stage.RATE, dtype=np.float32))
    bed = synth.float_audio(tmp_path / "bed.wav", np.full((3 * stage.RATE, 2), 0.1))
    music = {**resolve("preset:clean_minimal").data["music"], "duck_mode": "words",
             "duck_depth_db": -12, "duck_merge_gap_ms": gap_ms,
             "duck_attack_ms": 10, "duck_release_ms": 10, "level": 1, "dry_open_s": 0}
    result = stage.render_audio(tmp_path, tmp_path / "audio.m4a", work_dir=tmp_path / "work",
                                timeline=ResolvedTimeline((), (), (), 90, 0),
                                policy={"mode": "preserve"}, music=music, bed=bed,
                                cut_words={"time_base": "output", "words": [
                                    {"start": 1.2, "end": 2}, {"start": 0.5, "end": 1}]})
    pcm = stage.decode(tmp_path / "work/mix.wav", [], channels=2)
    expected = 0.1 * (10 ** (-12 / 20) if spans == 1 else 1)
    np.testing.assert_allclose(pcm[52800:53280], expected, atol=1e-8)
    assert result.to_dict()["music"]["ducked_spans"] == spans


def test_word_ducking_clips_ramps_overlaps_and_empty_spans() -> None:
    music = {"duck_depth_db": -12, "duck_attack_ms": 100, "duck_release_ms": 200,
             "duck_merge_gap_ms": 0}
    envelope, spans = stage._word_ducking([
        {"start": 0.05, "end": 0.1}, {"start": 0.15, "end": 0.2},
        {"start": 0.9, "end": 1.2}], stage.RATE, music)
    assert spans == 3 and len(envelope) == stage.RATE
    assert 20 * np.log10(envelope[0]) == pytest.approx(-6)
    assert 20 * np.log10(envelope[6000]) == pytest.approx(-10.5)
    np.testing.assert_allclose(envelope[43200:], 10 ** (-12 / 20), atol=1e-7)
    assert envelope.min() >= 10 ** (-12 / 20) - 1e-7
    empty, spans = stage._word_ducking([], stage.RATE, music)
    assert spans == 0
    np.testing.assert_array_equal(empty, np.ones(stage.RATE))
    _, spans = stage._word_ducking([{"start": 0.1, "end": 0.3},
                                    {"start": 0.2, "end": 0.4}], stage.RATE, music)
    assert spans == 1


@pytest.mark.parametrize("words", [None, {"time_base": "source", "words": []},
                                   {"time_base": "output"}])
def test_word_ducking_refuses_missing_output_timings(tmp_path: Path, words: dict | None) -> None:
    with pytest.raises(stage.AudioError, match="output time base"):
        stage.render_audio(tmp_path, tmp_path / "audio.m4a", work_dir=tmp_path / "work",
                           timeline=ResolvedTimeline((), (), (), 90, 0),
                           policy={"mode": "preserve"}, music={"duck_mode": "words"},
                           bed=tmp_path / "bed.wav", cut_words=words)
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize(("first", "last"), [(1, 0), (-1, 1),
                                             (float("nan"), 1), (0, float("inf"))])
def test_word_ducking_refuses_invalid_spans(first: float, last: float) -> None:
    music = resolve("preset:clean_minimal").data["music"]
    with pytest.raises(stage.AudioError, match="finite start and end"):
        stage._word_ducking([{"start": first, "end": last}], stage.RATE, music)


def test_assembly_retains_word_timings_and_tracks_voice_style(project: Path) -> None:  # noqa: F811
    context = ProjectContext(project)
    style = context.render_style()
    voice = {**style.data["voice"], "preset": "broadcast"}
    context._render_style = replace(style, data={**style.data, "voice": voice})
    snapshot = assembly.assembly_snapshot(context)
    assert snapshot.inputs["style:voice"] == projection_sha256(voice)
    assert "cut/cut-words.json" not in snapshot.small_files
    # The snapshot must include the exact cut document when words mode consumes it.
    data = json.loads((project / "project.json").read_bytes())
    data["style"] = "style.json"
    (project / "project.json").write_text(json.dumps(data), encoding="utf-8")
    (project / "style.json").write_text(json.dumps({
        "schema_version": 1, "music": {"duck_mode": "words"},
    }), encoding="utf-8")
    from yap_studio.stages.music import music

    music(project)
    snapshot = assembly.assembly_snapshot(ProjectContext(project))
    original = (project / "cut/cut-words.json").read_bytes()
    assert snapshot.small_files["cut/cut-words.json"] == original
    retained = project / "retained"
    assembly._retained(snapshot, retained)
    assert (retained / "cut/cut-words.json").read_bytes() == original
    (project / "cut/cut-words.json").write_text("{}", encoding="utf-8")
    assert (retained / "cut/cut-words.json").read_bytes() == original


def test_assembly_refuses_preserve_chain_before_snapshot(project: Path,  # noqa: F811
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    context = ProjectContext(project)
    style = context.render_style()
    context._render_style = replace(style, data={**style.data,
                                                "voice": {"preset": "broadcast"}})
    monkeypatch.setattr(assembly, "ProjectContext", lambda _: context)

    def unexpected_snapshot(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("The stage must refuse the voice chain before taking a snapshot.")

    monkeypatch.setattr(assembly, "assembly_snapshot", unexpected_snapshot)
    with pytest.raises(stage.AudioError) as caught:
        assembly.assemble(project)
    assert caught.value.code == "voice_chain_preserve"
    assert not (project / "final").exists()
