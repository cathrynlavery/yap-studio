"""Patches fit, match their surroundings and blend against the caller's decode."""

import numpy as np
import pytest
import synth

from yap_studio import audio
from yap_studio.checks.joins import click_at, click_peak_dbfs_at
from yap_studio.checks.model import Thresholds
from yap_studio.errors import YapError
from yap_studio.voice import splice
from yap_studio.voice.splice import blend, build_patch


def fixture_sound():
    rate = 48000
    rng = np.random.default_rng(42)
    sound = rng.normal(0, 0.0003, (6 * rate, 1)).astype(np.float32)
    for first, last in [(0.5, 1.5), (2, 3), (3.5, 4.5)]:
        a, b = round(first * rate), round(last * rate)
        sound[a:b, 0] += 0.08 * np.sin(2 * np.pi * 220 * np.arange(b - a) / rate)
    return sound


def test_patch_levels_tone_click_and_decode_drift(tmp_path):
    sound = fixture_sound()
    t = np.arange(48000) / 48000
    generated = synth.float_audio(
        tmp_path / "generated.wav", (0.02 * np.sin(2 * np.pi * 260 * t))[:, None]
    )
    patch, meta, checks = build_patch(
        sound, generated, 2, 3, tmp_path, word_spans=[(0.5, 1.5), (2, 3), (3.5, 4.5)],
        word_range=(1, 1),
    )
    assert 0.8 <= meta["stretch"] <= 1.25
    assert len(patch) == meta["samples"]
    assert abs(next(x for x in checks["items"] if x["id"] == "splice.level")["difference_lu"]) < 1
    assert next(x for x in checks["items"] if x["id"] == "splice.room_tone")["difference_db"] < 3
    drift = np.roll(sound, 4, axis=0)
    actual = blend(drift, patch, meta, source_start_sample=0)
    first, last = meta["start_sample"], meta["start_sample"] + len(patch)
    assert np.array_equal(actual[:first], drift[:first])
    assert np.array_equal(actual[last:], drift[last:])
    n = meta["fade_in_samples"]
    w = audio.raised_cosine(n)[:, None]
    np.testing.assert_allclose(
        actual[first : first + n], drift[first : first + n] * (1 - w) + patch[:n] * w
    )
    mono = actual.mean(axis=1)
    for point in (first, last - 1):
        assert (
            click_at(mono, 48000, point / 48000) <= Thresholds().click_fail_db
            or click_peak_dbfs_at(mono, 48000, point / 48000) < Thresholds().click_floor_dbfs
        )


def test_room_tone_floor_excludes_speech_in_a_dense_neighbourhood(tmp_path):
    time = np.arange(20 * splice.RATE) / splice.RATE
    sound = (np.sqrt(2) * 10 ** (-25 / 20) * np.sin(2 * np.pi * 220 * time))[:, None]
    a, b = round(0.5 * splice.RATE), round(0.6 * splice.RATE)
    sound[a:b, 0] = np.random.default_rng(19).normal(0, 0.001, b - a)
    generated = synth.float_audio(
        tmp_path / "generated.wav", (0.02 * np.sin(2 * np.pi * 260 * time[:24000]))[:, None],
    )
    patch, _, checks = build_patch(
        sound.astype(np.float32), generated, 10, 11, tmp_path,
        word_spans=[(0, 0.5), (0.6, 10), (10, 11), (11, 20)], word_range=(2, 2),
        pace="off",
    )
    tone = next(item for item in checks["items"] if item["id"] == "splice.room_tone")
    assert tone["source_floor_db"] == pytest.approx(-60, abs=1), "speech inflated the room floor"
    assert splice.level(patch[round(0.7 * splice.RATE):]) == pytest.approx(-60, abs=1)
    assert tone["status"] == "pass"


@pytest.mark.parametrize("duration", [0.6, 0.9])
def test_recorded_speech_is_not_measured_as_room_tone(tmp_path, monkeypatch, duration):
    from importlib import import_module

    from yap_studio.voice.store import add_cut_checks

    monkeypatch.setattr(splice, "_snap", lambda _, point: point)
    time = np.arange(round(duration * splice.RATE)) / splice.RATE
    speech = (0.01 + 0.07 * (0.5 + 0.5 * np.sin(2 * np.pi * time / duration))) * np.sin(
        2 * np.pi * 260 * time,
    )
    generated = synth.float_audio(tmp_path / "recorded.wav", speech[:, None])
    original = fixture_sound()
    patch, meta, checks = build_patch(
        original, generated, 2, 2 + duration, tmp_path,
        word_spans=[(0.5, 1.5), (2, 2 + duration), (3.5, 4.5)], word_range=(1, 1),
        pace="off", recorded=True,
    )
    tone = next(item for item in checks["items"] if item["id"] == "splice.room_tone")
    assert tone["status"] == "skipped", "quiet speech was measured as room tone"
    assert not tone["measured"] and tone["reason"]
    monkeypatch.setattr(import_module("yap_studio.voice.make"), "_heard", lambda *_, **kwargs:
                        {"id": "splice.heard", "status": "skipped"})
    first = meta["start_sample"]
    report = add_cut_checks(
        {"status": "pass", "findings": []}, blend(original, patch, meta, source_start_sample=0),
        [{"id": "f1", "start_sample": first, "end_sample": first + len(patch)}],
        [{"id": "f1", "take": "t1", "say": "test", "checks": checks}],
        plan=None, work=tmp_path, levels=None,
    )
    delivered = next(item for item in report["voice_fixes"][0]["checks"]["items"]
                     if item["id"] == "splice.room_tone")
    assert delivered["status"] == "skipped" and not delivered["measured"]


@pytest.mark.parametrize("stage", ["make", "cut"])
@pytest.mark.parametrize("duration,pace,boost,status", [
    (0.7125, "fit", 0, "skipped"), (0.2, "off", 0, "pass"), (0.2, "off", 20, "warn"),
])
def test_clone_room_tone_excludes_fitted_speech(
    tmp_path, monkeypatch, stage, duration, pace, boost, status,
):
    from importlib import import_module

    from yap_studio.voice.store import add_cut_checks

    monkeypatch.setattr(splice, "_snap", lambda _, point: point)
    time = np.arange(round(duration * splice.RATE)) / splice.RATE
    speech = (0.01 + 0.07 * (0.5 + 0.5 * np.cos(2 * np.pi * time / duration))) * np.sin(
        2 * np.pi * 260 * time,
    )
    generated = synth.float_audio(tmp_path / "clone.wav", speech[:, None])
    tone = splice._tone

    def louder_padding(*args):
        padding, floor = tone(*args)
        return padding * 10 ** (boost / 20), floor

    monkeypatch.setattr(splice, "_tone", louder_padding)
    original = fixture_sound()
    patch, meta, checks = build_patch(
        original, generated, 2, 2.57, tmp_path,
        word_spans=[(0.5, 1.5), (2, 2.57), (3.5, 4.5)], word_range=(1, 1), pace=pace,
    )
    if stage == "cut":
        monkeypatch.setattr(import_module("yap_studio.voice.make"), "_heard", lambda *_, **kwargs:
                            {"id": "splice.heard", "status": "skipped"})
        first = meta["start_sample"]
        report = add_cut_checks(
            {"status": "pass", "findings": []},
            blend(original, patch, meta, source_start_sample=0),
            [{"id": "f1", "start_sample": first, "end_sample": first + len(patch)}],
            [{"id": "f1", "take": "t1", "say": "test", "patch": meta, "checks": checks}],
            plan=None, work=tmp_path, levels=None,
        )
        checks = report["voice_fixes"][0]["checks"]
    check = next(item for item in checks["items"] if item["id"] == "splice.room_tone")
    assert check["status"] == status, "generated speech was measured as room tone"
    assert check["measured"] == (status != "skipped")
    if status == "skipped":
        assert meta["stretch"] == pytest.approx(0.8)
        assert check["reason"] == "Speech fills the slot; no quiet frames remain."
    else:
        assert check["patch_floor_db"] == pytest.approx(check["source_floor_db"] + boost, abs=1)


@pytest.mark.parametrize("noise_db", [-60, -40])
def test_recorded_room_tone_measures_word_gaps(noise_db):
    time = np.arange(splice.RATE) / splice.RATE
    patch = (0.08 * np.sin(2 * np.pi * 260 * time))[:, None].astype(np.float32)
    patch[19200:28800, 0] = np.random.default_rng(31).normal(0, 10 ** (noise_db / 20), 9600)
    tone = splice.room_tone_check(patch, -60, speech_spans=[(0, 0.4), (0.6, 1)])
    assert tone["measured"]
    assert tone["patch_floor_db"] == pytest.approx(noise_db, abs=1)
    assert tone["status"] == ("pass" if noise_db == -60 else "warn")


@pytest.mark.parametrize("spans", [{"start": 0}, [None], [["word", 0.2]],
                                   [[0, float("inf")]], [[0.5, 0.2]]])
def test_room_tone_rejects_invalid_speech_spans(spans):
    with pytest.raises(YapError) as caught:
        splice.room_tone_check(np.full((4800, 1), 0.001), -60, speech_spans=spans)
    assert caught.value.code == "voice_record_invalid"


def test_too_long(tmp_path):
    generated = synth.float_audio(tmp_path / "long.wav", np.full((3 * 48000, 1), 0.1))
    with pytest.raises(YapError, match=r"3.000 s.*1\.[0-9]+ s"):
        build_patch(fixture_sound(), generated, 2, 3, tmp_path,
                    word_spans=[(2, 3), (3.1, 4)], word_range=(0, 0))


@pytest.mark.parametrize("recorded", [False, True])
def test_pace_off_keeps_speech_length_and_pause(tmp_path, recorded):
    generated = synth.float_audio(tmp_path / "short.wav", np.full((9600, 1), 0.05))
    patch, meta, checks = build_patch(
        fixture_sound(), generated, 2, 3, tmp_path,
        word_spans=[(0.5, 1.5), (2, 3), (3.5, 4.5)], word_range=(1, 1),
        pace="off", recorded=recorded,
    )
    assert meta["stretch"] == 1
    pace = next(item for item in checks["items"] if item["id"] == "splice.pace")
    assert pace["status"] == "warn" and pace["gap_s"] > 0.7
    assert len(patch) == meta["samples"]


def test_recorded_room_is_not_added_under_speech(tmp_path, monkeypatch):
    generated = synth.float_audio(tmp_path / "own.wav", np.full((48000, 1), 0.05))

    def no_tone(*args):
        pytest.fail("Recorded speech fills the slot and already has its room tone.")

    monkeypatch.setattr(splice, "_tone", no_tone)
    monkeypatch.setattr(splice, "_snap", lambda sound, nominal: nominal)
    build_patch(fixture_sound(), generated, 2, 3, tmp_path,
                word_spans=[(0.5, 1.5), (2, 3), (3.5, 4.5)], word_range=(1, 1),
                pace="off", recorded=True)


def test_recorded_padding_handoff_is_crossfaded_and_checked(tmp_path, monkeypatch):
    sound = fixture_sound()
    time = np.arange(len(sound)) / splice.RATE
    sound += (0.005 * np.cos(2 * np.pi * 50 * time))[:, None]
    generated = synth.float_audio(tmp_path / "own.wav", np.full((9600, 1), 0.05))
    patch, meta, checks = build_patch(
        sound, generated, 2, 3, tmp_path,
        word_spans=[(0.5, 1.5), (2, 3), (3.5, 4.5)], word_range=(1, 1),
        pace="off", recorded=True,
    )
    actual = blend(sound, patch, meta, source_start_sample=0).mean(axis=1)
    handoff = meta["start_sample"] + 9600
    assert (click_at(actual, splice.RATE, handoff / splice.RATE) <= Thresholds().click_fail_db
            or click_peak_dbfs_at(actual, splice.RATE, handoff / splice.RATE)
            <= Thresholds().click_floor_dbfs), "recorded padding clicks inside the patch"
    click = next(item for item in checks["items"] if item["id"] == "splice.click")
    assert handoff in {edge["sample"] for edge in click["edges"]}
    assert click["status"] == "pass"
    from importlib import import_module

    from yap_studio.voice.store import add_cut_checks

    monkeypatch.setattr(import_module("yap_studio.voice.make"), "_heard", lambda *_, **kwargs:
                        {"id": "splice.heard", "status": "skipped"})
    first = meta["start_sample"]
    applied = [{"id": "f1", "start_sample": first, "end_sample": first + len(patch)}]
    fix = {"id": "f1", "take": "t1", "say": "test", "checks": checks}
    report = add_cut_checks({"status": "pass", "findings": []}, actual[:, None], applied,
                            [fix], plan=None, work=tmp_path, levels=None)
    click = next(item for item in report["voice_fixes"][0]["checks"]["items"]
                 if item["id"] == "splice.click")
    assert handoff in {edge["sample"] for edge in click["edges"]}


@pytest.mark.parametrize("next_start", [2.5, 2.4995, 2.45, 1.95, 2.69])
def test_extension_cannot_cover_the_following_word(tmp_path, monkeypatch, next_start):
    monkeypatch.setattr(splice, "_snap", lambda _, point: point)
    generated = synth.float_audio(tmp_path / "long.wav", np.full((41000, 1), 0.1))
    with pytest.raises(YapError) as caught:
        build_patch(
            fixture_sound(), generated, 2, 2.5, tmp_path,
            word_spans=[(0.5, 1.5), (2, 2.5), (next_start, 3), (3.5, 4.5)],
            word_range=(1, 1),
        )
    assert caught.value.code == "voice_speech_too_long"


def test_extension_keeps_ten_ms_before_the_following_word(tmp_path, monkeypatch):
    monkeypatch.setattr(splice, "_snap", lambda _, point: point)
    generated = synth.float_audio(tmp_path / "long.wav", np.full((41000, 1), 0.1))
    _, meta, _ = build_patch(
        fixture_sound(), generated, 2, 2.5, tmp_path,
        word_spans=[(0.5, 1.5), (2, 2.5), (2.7, 3), (3.5, 4.5)], word_range=(1, 1),
    )
    end = (meta["start_sample"] + meta["samples"]) / 48000
    assert 2.5 < end <= 2.69


def test_snapped_edges_cannot_cover_other_words(tmp_path, monkeypatch):
    monkeypatch.setattr(splice, "_snap", lambda _, point: point - 480 if point == 96000
                        else point + 480)
    generated = synth.float_audio(tmp_path / "short.wav", np.full((9600, 1), 0.1))
    _, meta, _ = build_patch(
        fixture_sound(), generated, 2, 2.5, tmp_path,
        word_spans=[(0.5, 2.01), (2, 2.5), (2.49, 3)], word_range=(1, 1),
    )
    assert meta["start_sample"] >= round(2.01 * 48000)
    assert meta["start_sample"] + meta["samples"] <= round(2.49 * 48000)


def test_container_clock_rounds_after_offset():
    from fractions import Fraction
    from types import SimpleNamespace

    from yap_studio.voice.plan import source_sample
    from yap_studio.voice.store import apply_piece

    plan = SimpleNamespace(media={"t1": SimpleNamespace(video_offset_s=1 / 96000)})
    piece = SimpleNamespace(take="t1", src_t0=Fraction(1, 96000))
    assert source_sample(plan, "t1", piece.src_t0) == 1
    original = np.zeros((100, 1), dtype=np.float32)
    patch = np.ones((10, 1), dtype=np.float32)
    fix = {
        "take": "t1",
        "patch": {"start_sample": 11, "samples": 10, "fade_in_samples": 0, "fade_out_samples": 0},
    }
    actual = apply_piece(original, piece, [(fix, patch)], plan)
    np.testing.assert_array_equal(actual[10:20], patch)
    assert not np.any(actual[:10]) and not np.any(actual[20:])
