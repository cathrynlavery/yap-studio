"""Synthetic motion, joins, noise and measurements for person matte refinement."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_matte_make import HEIGHT, WIDTH, FakeHelper, decode_matte, make_clip
from test_matte_project import synthetic_matte

from yap_studio.errors import ProcessingError, ValidationError
from yap_studio.matte import make, project, refine
from yap_studio.matte.model import Cleanup, MatteData, load_matte

RAW = Cleanup(low=0, high=255, feather_sigma=0)


def refined(masks, guides, joins=()):
    return list(refine.refine_frames(iter(masks), iter(guides), cleanup=RAW, joins=joins))


def test_moving_disc_has_no_trail_and_still_rim_is_quieter() -> None:
    rng = np.random.default_rng(7)
    y, x = np.indices((96, 128))
    texture = rng.integers(25, 65, x.shape)
    masks, guides = [], []
    for t in range(24):
        distance = np.hypot(x - (40 + max(0, t - 12) * 4), y - 48)
        alpha = np.clip((23 - distance) / 5, 0, 1)
        jitter = rng.normal(0, 12, x.shape) * ((alpha > 0) & (alpha < 1))
        masks.append(np.clip(alpha * 255 + jitter, 0, 255).astype(np.uint8))
        guides.append(np.clip(texture * (1 - alpha) + 190 * alpha
                              + rng.normal(0, 2, x.shape), 0, 255).astype(np.uint8))
    result = refined(masks, guides)
    rim = (masks[0] > 20) & (masks[0] < 235)
    before = np.abs(np.diff(np.array(masks[:12], dtype=float), axis=0))[:, rim].mean()
    after = np.abs(np.diff(np.array(result[:12], dtype=float), axis=0))[:, rim].mean()
    assert after < before * 0.7
    assert max(frame[0, 0] for frame in result) == 0
    assert all(frame[48, 40] == 255 for frame in result[:12])
    assert result[-1][48, 40] <= 1
    assert np.array_equal(result, refined(masks, guides))


def test_join_matches_independent_shots_and_test_detects_blending() -> None:
    p = np.full((32, 48), 70, dtype=np.uint8)
    q = np.full_like(p, 185)
    guides = [np.full_like(p, 100)] * 6 + [np.full_like(p, 104)] * 6
    joined = refined([p] * 6 + [q] * 6, guides, (6,))
    separate = refined([p] * 6, guides[:6]) + refined([q] * 6, guides[6:])
    assert np.array_equal(joined[5:7], separate[5:7])
    blended = refined([p] * 6 + [q] * 6, guides)
    assert not np.array_equal(blended[5:7], separate[5:7])


def test_still_threshold_accepts_noise_and_rejects_motion() -> None:
    rng = np.random.default_rng(19)
    base = rng.uniform(70, 170, (192, 108)).astype(np.float32)
    a, b = [np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8)
            for _ in range(2)]
    still = refine.still_pixels(a, b)
    assert still.mean() > 0.99
    b[60:130, 30:80] = 220
    assert refine.still_pixels(a, b)[70:120, 40:70].mean() < 0.01


@pytest.mark.parametrize("median", [False, True])
def test_refinement_equals_temporal_mean_inside_and_outside_band(median: bool) -> None:
    rng = np.random.default_rng(47)
    y, x = np.indices((96, 128))
    ramp = np.clip((x - 55) / 8, 0, 1)
    texture = np.where((x // 3 + y // 3) % 2, 60, 190)
    masks, guides = [], []
    for t in range(7):
        jitter = rng.normal(0, 12, ramp.shape) * ((ramp > 0) & (ramp < 1))
        masks.append(np.clip(ramp * 255 + jitter, 0, 255).astype(np.uint8))
        guide = np.clip(texture + rng.normal(0, 2, ramp.shape), 0, 255).astype(np.uint8)
        if t == 3:
            guide[25:65, 50:75] = 230
        guides.append(guide)
    medians = [np.median([masks[max(0, t - 1)], masks[t], masks[min(6, t + 1)]], axis=0)
               .astype(np.uint8) for t in range(7)] if median else masks
    result = list(refine.refine_frames(iter(masks), iter(guides),
                                      cleanup=replace(RAW, median=median)))
    for t, actual in enumerate(result):
        total = medians[t].astype(np.float32)
        weights = np.ones(ramp.shape, dtype=np.uint8)
        for n in range(max(0, t - 2), min(6, t + 2) + 1):
            if n != t:
                eligible = refine.still_pixels(guides[t], guides[n])
                total += np.where(eligible, medians[n], 0)
                weights += eligible
        expected = np.round(total / weights).astype(np.uint8)
        band = refine.edge_band(expected)
        assert band.any() and (~band).any()
        assert np.array_equal(actual[band], expected[band])
        assert np.array_equal(actual[~band], expected[~band])


def test_refined_matte_keeps_hard_edge_byte_for_byte(tmp_path: Path, monkeypatch) -> None:
    clip = make_clip(tmp_path / "clip.mp4")
    y, x = np.indices((HEIGHT, WIDTH))
    mask = np.where(x >= 6 + y % 5, 255, 0).astype(np.uint8)
    frames = [mask.tobytes()] * 3
    FakeHelper(tmp_path, monkeypatch).install(frames)
    data = make.make_matte(clip, tmp_path / "out")
    assert data.cleanup == Cleanup()
    assert decode_matte(tmp_path / "out/matte.mkv") == b"".join(frames)


def test_sparse_temporal_mean_matches_full_picture_mean() -> None:
    rng = np.random.default_rng(31)
    center = np.zeros((64, 96), dtype=np.uint8)
    center[:, 40:60] = 180
    guide = rng.integers(70, 170, center.shape, dtype=np.uint8)
    neighbours, total, weights = [], center.astype(np.float32), np.ones(center.shape)
    for i in range(4):
        alpha = center.copy()
        alpha[:, 38:63] = rng.integers(0, 256, (64, 25), dtype=np.uint8)
        picture = guide.copy()
        picture[20:40, 30 + i:70] = 240
        eligible = refine.still_pixels(guide, picture)
        total += np.where(eligible, alpha, 0)
        weights += eligible
        neighbours.append((alpha, refine._still_guide(picture)))
    expected = np.round(total / weights).astype(np.uint8)
    actual = refine.temporal_mean(center, refine._still_guide(guide), neighbours)
    assert np.array_equal(actual, expected)


def test_shot_starts_reads_continuations_jumps_takes_and_holds(tmp_path: Path) -> None:
    spans = tmp_path / "spans.json"
    spans.write_text(json.dumps({
        "sources": {"a": "a.mov", "b": "b.mov"}, "fps": "30/1",
        "segments": [
            {"source": "a", "start": "0", "end": "1"},
            {"source": "a", "start": "1", "end": "2"},
            {"source": "a", "start": "3", "end": "4"},
            {"source": "b", "start": "0", "end": "1"},
            {"source": "b", "start": "1", "end": "2"},
        ],
    }))
    assert refine.shot_starts(spans) == (60, 90)
    spans.write_text(json.dumps({"sources": {"a": "a.mov"}, "ranges": [
        {"source": "a", "start": 0, "end": 1}, {"source": "a", "start": 1, "end": 2}],
        "holds": [{"start_frame": 30, "end_frame": 36, "audio": "silence"}]}))
    assert refine.shot_starts(spans) == (30, 36)


def test_measurements_window_flicker_width_and_join(tmp_path: Path) -> None:
    from test_matte_edge_colour import write_planes

    rng = np.random.default_rng(5)
    ramp = np.tile(np.clip((np.arange(96) - 40) / 10, 0, 1), (64, 1))
    masks = [np.clip(ramp * 255 + rng.normal(0, 8, ramp.shape)
                     * ((ramp > 0) & (ramp < 1)), 0, 255).astype(np.uint8) for _ in range(20)]
    guides = [np.full((64, 96), 100, dtype=np.uint8)] * 20
    video = write_planes(tmp_path / "video.mkv", guides, "gray")
    matte = write_planes(tmp_path / "matte.mkv", masks, "gray")
    clean = write_planes(tmp_path / "clean.mkv", refined(masks, guides), "gray")
    before = refine.measure_edges(matte, video, first=3, count=12)
    after = refine.measure_edges(clean, video, first=3, count=12)
    assert before.frames == after.frames == 12
    assert after.flicker < before.flicker
    known = write_planes(tmp_path / "known.mkv", [(ramp * 255).astype(np.uint8)] * 20, "gray")
    assert refine.measure_edges(known, video, first=4, count=3).edge_width_px == pytest.approx(
        9, abs=0.5)
    stepped = write_planes(tmp_path / "step.mkv", [masks[0]] * 10 + [masks[-1]] * 10, "gray")
    assert refine.measure_edges(stepped, video, joins=(10,)).flicker == 0
    assert refine.measure_edges(stepped, video).flicker > 0


def test_make_default_records_joins_and_loads_legacy(tmp_path: Path, monkeypatch) -> None:
    clip = make_clip(tmp_path / "clip.mp4")
    FakeHelper(tmp_path, monkeypatch).install([bytes([127]) * (16 * 24)] * 3)
    data = make.make_matte(clip, tmp_path / "out", joins=(1,))
    assert data.cleanup.refine and data.joins == (1,)
    assert load_matte(tmp_path / "out/matte.json") == data
    again = make.make_matte(clip, tmp_path / "again", joins=(1,))
    assert data.matte["sha256"] == again.matte["sha256"]
    legacy = data.to_dict()
    del legacy["cleanup"]["refine"]
    del legacy["joins"]
    old = MatteData.from_dict(legacy)
    assert not old.cleanup.refine and old.joins == ()


@pytest.mark.parametrize("sigma", [0, 0.125, 1, 8])
def test_refined_encoder_has_no_spatial_cleanup(tmp_path: Path, sigma: float) -> None:
    from fractions import Fraction

    from yap_studio.faces.model import VideoIdentity

    video = VideoIdentity("test", "a" * 64, 3, Fraction(30), 16, 24)
    command = make._encoder_command(tmp_path / "out.mkv", video,
                                    Cleanup(feather_sigma=sigma), refined=True)
    assert command[command.index("-vf") + 1] == "null"


@pytest.mark.parametrize("joins", [(0,), (3,), (True,), (2, 1), (1, 1)])
def test_invalid_joins_rejected(tmp_path: Path, joins: tuple) -> None:
    clip = make_clip(tmp_path / "clip.mp4")
    with pytest.raises(ValidationError, match="joins"):
        make.make_matte(clip, tmp_path / "out", joins=joins)


@pytest.mark.parametrize("damage", ["legacy", "joins", "spans"])
def test_vision_cache_rebuilds_old_refinement_or_joins(tmp_path: Path, monkeypatch, damage) -> None:
    cut = make_clip(tmp_path / "cut.mp4")
    (cut.parent / "spans.json").write_text(json.dumps({
        "sources": {"a": "a.mov", "b": "b.mov"}, "segments": [
            {"source": "a", "start": 0, "end": "1/30"},
            {"source": "b", "start": 0, "end": "2/30"},
        ],
    }))
    calls = []

    def build(source, folder, *, joins=()):
        calls.append(joins)
        data = synthetic_matte(source, folder)
        data = replace(data, cleanup=Cleanup(), joins=tuple(joins), backend={
            "name": "vision-person-segmentation", "quality": "accurate", "revision": 1,
            "os": "test", "helper_sha256": "a" * 64, "mask_width": 8, "mask_height": 8,
        })
        (folder / "matte.json").write_text(json.dumps(data.to_dict()))
        return data

    monkeypatch.setattr(project, "make_matte", build)
    matte = project.ensure_cut_matte(tmp_path / "project", cut)
    payload = json.loads((matte.parent / "matte.json").read_text())
    if damage == "legacy":
        payload["cleanup"]["refine"] = False
    elif damage == "joins":
        payload["joins"] = [2]
    else:
        spans = json.loads((cut.parent / "spans.json").read_text())
        spans["segments"][0]["end"] = "2/30"
        spans["segments"][1]["end"] = "1/30"
        (cut.parent / "spans.json").write_text(json.dumps(spans))
    (matte.parent / "matte.json").write_text(json.dumps(payload))
    assert project.ensure_cut_matte(tmp_path / "project", cut) == matte
    assert project.ensure_cut_matte(tmp_path / "project", cut) == matte
    assert calls == [(1,), (2,) if damage == "spans" else (1,)]


def test_refined_helper_short_output_fails_without_publication(tmp_path: Path, monkeypatch) -> None:
    clip = make_clip(tmp_path / "clip.mp4")
    FakeHelper(tmp_path, monkeypatch).install([bytes(384)] * 3, raw=bytes(384 * 2 + 1))
    with pytest.raises(ProcessingError) as caught:
        make.make_matte(clip, tmp_path / "out")
    assert caught.value.code in {"matte_frame_count", "person_matte_output"}
    assert list((tmp_path / "out").iterdir()) == []


def test_helper_reason_survives_stdout_closing_before_exit(tmp_path: Path, monkeypatch) -> None:
    clip = make_clip(tmp_path / "clip.mp4")
    fake = FakeHelper(tmp_path, monkeypatch).install([bytes(384)] * 3, raw=bytes(384 * 2))
    script = fake.helper.read_text()
    fake.helper.write_text(script + "\nos.close(1)\ntime.sleep(0.1)\n"
                           "sys.stderr.write('deliberate late helper failure')\n"
                           "sys.stderr.flush()\nos._exit(4)\n")
    with pytest.raises(ProcessingError) as caught:
        make.make_matte(clip, tmp_path / "out")
    assert caught.value.code == "person_matte_exit_4"
    assert "deliberate late helper failure" in caught.value.message
    assert list((tmp_path / "out").iterdir()) == []
    fake.assert_reaped()
