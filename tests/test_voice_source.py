"""Verifier patch audio uses the same resampling as the rendered audio."""

from types import SimpleNamespace

import numpy as np
import pytest

from yap_studio import audio, receipt
from yap_studio.voice.splice import write_wav
from yap_studio.voice.store import source_transform


def stored_patch(root, fix_id, sound, *, take="t1", start=12345, fade_in=0, fade_out=0):
    target = root / f"voice/{fix_id}/patch.wav"
    write_wav(target, sound)
    return {"id": fix_id, "take": take, "patch": {
        "path": str(target.relative_to(root)), "sha256": receipt.sha256_file(target),
        "samples": len(sound), "channels": sound.shape[1], "start_sample": start,
        "fade_in_samples": fade_in, "fade_out_samples": fade_out,
    }}


def tones(count):
    time = np.arange(count) / 48000
    return (0.01 * np.sin(2 * np.pi * 1000 * time)
            + 0.1 * np.sin(2 * np.pi * 10000 * time)).astype(np.float32)


@pytest.mark.parametrize("rate", [16000, 48000])
def test_source_transform_matches_verifier_decode(tmp_path, rate):
    patch = tones(48000)[:, None]
    fix = stored_patch(tmp_path, "f1", patch)
    plan = SimpleNamespace(context=SimpleNamespace(project_dir=tmp_path))
    original = np.zeros(2 * rate, dtype=np.float32)
    transformed = source_transform(plan, [(fix, patch)])("t1", original, rate)
    first = round(fix["patch"]["start_sample"] * rate / 48000)
    actual = transformed[first:first + rate]
    if rate == 48000:
        np.testing.assert_array_equal(actual, patch[:, 0])
    else:
        _, expected = audio.decode(tmp_path / fix["patch"]["path"], sr=rate)
        assert np.corrcoef(actual, expected)[0, 1] > 0.99
        assert np.sqrt(np.mean((actual - expected) ** 2)) < 1e-6
    assert not np.any(transformed[:first])
    assert not np.any(transformed[first + rate:])
    assert not np.any(original)


@pytest.mark.parametrize("rate", [16000, 48000])
def test_source_transform_stereo_matches_verifier_decode(tmp_path, rate):
    patch = np.full((48000, 2), 0.1, dtype=np.float32)
    fix = stored_patch(tmp_path, "f1", patch)
    plan = SimpleNamespace(context=SimpleNamespace(project_dir=tmp_path))
    _, expected = audio.decode(tmp_path / fix["patch"]["path"], sr=rate)
    original = np.zeros(2 * rate, dtype=np.float32)
    actual = source_transform(plan, [(fix, patch)])("t1", original, rate)
    first = round(fix["patch"]["start_sample"] * rate / 48000)
    assert np.sqrt(np.mean((actual[first:first + rate] - expected) ** 2)) < 1e-6
    assert len(actual) == len(original)
    assert not np.any(actual[:first])
    assert not np.any(actual[first + rate:])
    assert not np.any(original)


@pytest.mark.parametrize("rate", [16000, 48000])
def test_source_transform_keeps_scaled_fades_and_length(tmp_path, rate):
    mono = tones(48001)
    patch = np.column_stack((mono, mono * 0.5))
    fix = stored_patch(tmp_path, "f1", patch, fade_in=721, fade_out=899)
    meta = fix["patch"]
    count = round(len(patch) * rate / 48000)
    _, signal = audio.decode(tmp_path / meta["path"], sr=rate, dur_s=count / rate)
    original = np.random.default_rng(42).normal(0, 0.02, 2 * rate).astype(np.float32)
    before = original.copy()
    first = round(meta["start_sample"] * rate / 48000)
    fade_in = round(meta["fade_in_samples"] * rate / 48000)
    fade_out = round(meta["fade_out_samples"] * rate / 48000)
    weight = np.ones(count, dtype=np.float32)
    weight[:fade_in] = audio.raised_cosine(fade_in)
    weight[-fade_out:] = audio.raised_cosine(fade_out)[::-1]
    expected = original.copy()
    expected[first:first + count] = (
        original[first:first + count] * (1 - weight) + signal * weight
    )
    plan = SimpleNamespace(context=SimpleNamespace(project_dir=tmp_path))
    actual = source_transform(plan, [(fix, patch)])("t1", original, rate)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(original, before)


def test_source_transform_caches_each_fix_and_rate(tmp_path, monkeypatch):
    patch = tones(48000)[:, None]
    first = stored_patch(tmp_path, "f1", patch, start=0)
    second = stored_patch(tmp_path, "f2", patch * 2, take="t2", start=0)
    plan = SimpleNamespace(context=SimpleNamespace(project_dir=tmp_path))
    transform = source_transform(plan, [(first, patch), (second, patch * 2)])
    decode = audio.decode
    calls = []

    def record(path, **kwargs):
        calls.append((path, kwargs["sr"]))
        return decode(path, **kwargs)

    monkeypatch.setattr(audio, "decode", record)
    for fix, rate in ((first, 16000), (second, 16000), (first, 8000), (first, 48000)):
        _, expected = decode(tmp_path / fix["patch"]["path"], sr=rate)
        for value in (0, 0.25):
            original = np.full(2 * rate, value, dtype=np.float32)
            actual = transform(fix["take"], original, rate)
            np.testing.assert_array_equal(actual[:rate], expected)
            np.testing.assert_array_equal(actual[rate:], original[rate:])
    assert calls == [(tmp_path / fix["patch"]["path"], rate)
                     for fix, rate in ((first, 16000), (second, 16000), (first, 8000),
                                       (first, 48000))]


@pytest.mark.parametrize("difference", [-1, 1])
def test_source_transform_normalizes_decoded_length(tmp_path, monkeypatch, difference):
    patch = np.full((480, 1), 0.125, dtype=np.float32)
    fix = stored_patch(tmp_path, "f1", patch, start=96)
    plan = SimpleNamespace(context=SimpleNamespace(project_dir=tmp_path))
    calls = []

    def decode(path, **kwargs):
        calls.append(path)
        return 16000, np.full(160 + difference, 0.125, dtype=np.float32)

    monkeypatch.setattr(audio, "decode", decode)
    transform = source_transform(plan, [(fix, patch)])
    for value in (0.75, 0.5):
        original = np.full(400, value, dtype=np.float32)
        expected = original.copy()
        expected[32:192] = 0.125
        if difference < 0:
            expected[191] = 0
        actual = transform("t1", original, 16000)
        assert len(actual) == len(original)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(original, np.full(400, value, dtype=np.float32))
    assert calls == [tmp_path / fix["patch"]["path"]]
