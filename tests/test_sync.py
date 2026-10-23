"""Independent calibration and finding behaviour for segment sync."""

from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import synth

from yap_studio import audio, ffmpeg, video
from yap_studio.checks import sync
from yap_studio.checks.model import CHECK_IDS, Finding, Thresholds, format_time, status_of
from yap_studio.edl.model import EditList, PictureRegion, Segment, Source
from yap_studio.plan import Plan


def _plan(source: Path, spans: list[tuple[float, float]], fps: Fraction = Fraction(30)) -> Plan:
    edits = EditList(
        "yap",
        source.with_suffix(".json"),
        {"main": Source("main", source)},
        [
            Segment(i, "main", Fraction(str(a)), Fraction(str(b)))
            for i, (a, b) in enumerate(spans, 1)
        ],
        fps,
        [],
        [],
    )
    return Plan.build(edits, fps=fps, sample_rate=48000)


@pytest.fixture(scope="module")
def decoded_once():
    """Share real decodes across cases without weakening the measurement code."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(audio, "decode", lru_cache(None)(audio.decode))
        patch.setattr(video, "grey_frames", lru_cache(None)(video.grey_frames))
        yield


@pytest.fixture(scope="module")
def calibration(synth_media, decoded_once):
    source = synth_media.path("calib")
    vertical = synth_media.path("calib_vertical")
    plan = _plan(source, synth_media.spans)
    vertical_plan = _plan(vertical, synth_media.spans)
    vertical_clean = synth_media.cached(
        "t7-vertical-clean",
        lambda out: synth.render_clean(vertical, synth_media.spans, out),
    )
    fitted = synth_media.cached(
        "t7-clean-fit",
        lambda out: synth.reframe(vertical_clean, out, "fit"),
    )
    crop = synth_media.path("clean_crop")

    def drop_first(out: Path) -> None:
        ffmpeg.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(synth_media.path("clean")),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-vf",
                "trim=start_frame=1,setpts=PTS-STARTPTS",
                "-c:v",
                "libx264",
                "-c:a",
                "copy",
                str(out),
            ],
            timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
        )

    paths = {
        name: synth_media.path(name)
        for name in (
            "clean",
            "clean_shift_samples",
            "clean_shift_start",
            "perclip",
            "clean_crop",
        )
    }
    paths.update(
        {
            "missing_first": synth_media.cached("t7-missing-first", drop_first),
            "crop_shift": synth_media.cached(
                "t7-crop-shift",
                lambda out: synth.shift_audio(crop, out, 100, "samples"),
            ),
            "fit": fitted,
            "fit_shift": synth_media.cached(
                "t7-fit-shift",
                lambda out: synth.shift_audio(fitted, out, 100, "samples"),
            ),
        }
    )
    results, oracles, reports = {}, {}, {}
    for name, path in paths.items():
        is_fit = name.startswith("fit")
        results[name] = sync.check_sync(
            path,
            vertical_plan if is_fit else plan,
            {"main": vertical if is_fit else source},
        )
        oracles[name] = synth.oracle(path, synth_media.spans)
        offsets = [row.av_offset_ms for row in results[name].segments if row.measured]
        deltas = [
            row.av_offset_ms - oracles[name].segment_offsets_ms[row.index]
            for row in results[name].segments
            if row.measured and row.index in oracles[name].segment_offsets_ms
        ]
        reports[name] = {
            "min_ms": min(offsets) if offsets else None,
            "max_ms": max(offsets) if offsets else None,
            "drift_ms_per_cut": results[name].drift_ms_per_cut,
            "oracle_delta_min_ms": min(deltas) if deltas else None,
            "oracle_delta_max_ms": max(deltas) if deltas else None,
            "oracle_drift_ms_per_cut": oracles[name].slope_ms_per_cut,
            "corr_min": min(row.corr for row in results[name].segments),
            "corr_max": max(row.corr for row in results[name].segments),
            "whitened_corr_min": min(row.whitened_corr for row in results[name].segments),
            "whitened_corr_max": max(row.whitened_corr for row in results[name].segments),
            "segments": [row.to_dict() for row in results[name].segments],
        }
    # Retain calibration evidence for the task's final report, not test stdout.
    (synth_media.root / "t7-sync-report.json").write_text(json.dumps(reports, indent=2))
    return results, oracles, paths, plan


@pytest.mark.parametrize(
    "name",
    [
        "clean",
        "clean_shift_samples",
        "clean_shift_start",
        "missing_first",
        "clean_crop",
        "crop_shift",
        "fit",
        "fit_shift",
        "perclip",
    ],
)
def test_sync_calibrates_against_independent_oracle(calibration, name: str) -> None:
    results, oracles, _, plan = calibration
    result, oracle = results[name], oracles[name]
    assert len(result.segments) == len(plan.segments) == 8
    assert set(oracle.segment_offsets_ms) == set(range(1, 9))
    for row in result.segments:
        assert row.measured, row
        high = (
            row.corr >= 0.8
            and row.ambiguity < 0.8
            and row.frame_distance < 0.8 * row.frame_second_best
        )
        assert row.confidence == ("high" if high else "low"), row
        assert row.confidence == "high", row
        assert row.whitened_corr is not None
        assert row.to_dict()["whitened_corr"] == row.whitened_corr
        assert row.av_offset_ms == pytest.approx(oracle.segment_offsets_ms[row.index], abs=2)
        assert row.head_s is not None and row.tail_s is not None
        assert row.head_corr is not None and row.tail_corr is not None
        assert row.video_landing_s == pytest.approx(
            row.audio_landing_s - row.av_offset_ms / 1000,
            abs=1e-9,
        )
        if name == "perclip":
            continue
        expected = 100 if "shift" in name else 1000 / 30 if name == "missing_first" else 0
        assert row.av_offset_ms == pytest.approx(expected, abs=1)
        assert row.video_vs_plan_frames == (-1 if name == "missing_first" else 0)
    assert result.drift_ms_per_cut == pytest.approx(oracle.slope_ms_per_cut, abs=2)
    if name != "perclip":
        assert abs(result.drift_ms_per_cut) <= 0.5
    else:
        assert result.drift_ms_per_cut == pytest.approx(32, abs=2)
    if "shift" in name or name == "perclip":
        assert any(f.check == "sync.av_offset" and f.severity == "fail" for f in result.findings)
    else:
        assert not any(f.severity == "fail" for f in result.findings)
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "name,base",
    [
        ("clean_shift_samples", "clean"),
        ("clean_shift_start", "clean"),
        ("crop_shift", "clean_crop"),
        ("fit_shift", "fit"),
    ],
)
def test_shift_tracks_oracle_measured_shift(calibration, name: str, base: str) -> None:
    results, oracles, _, _ = calibration
    for shifted, clean in zip(results[name].segments, results[base].segments, strict=True):
        shift = (
            oracles[name].segment_offsets_ms[shifted.index]
            - oracles[base].segment_offsets_ms[shifted.index]
        )
        assert shifted.av_offset_ms - clean.av_offset_ms == pytest.approx(shift, abs=0.2)


def test_plain_correlations_match_independent_measurements(calibration, synth_media) -> None:
    results, _, paths, plan = calibration
    for name, result in results.items():
        source_name = "calib_vertical" if name.startswith("fit") else "calib"
        sr, source = audio.decode(synth_media.path(source_name).resolve(), sr=16000)
        _, rendered = audio.decode(paths[name], sr=sr)
        for row, segment in zip(result.segments, plan.segments, strict=True):
            start = float(segment.src_f0 / plan.fps)
            end = float(segment.src_f1 / plan.fps)
            middle = float((segment.src_f0 + segment.n_frames // 2) / plan.fps)
            windows = [
                (middle - 0.25, middle + 0.25, row.audio_landing_s - 0.25, row.corr),
                (start + 0.03, start + 0.33, row.head_s + 0.03, row.head_corr),
                (end - 0.33, end - 0.03, row.tail_s - 0.33, row.tail_corr),
            ]
            for a, b, landing, actual in windows:
                first, last, landed = (int(np.floor(t * sr + 0.5)) for t in (a, b, landing))
                needle = source[first:last]
                matched = rendered[landed : landed + len(needle)]
                assert len(matched) == len(needle)
                expected = float(np.corrcoef(needle, matched)[0, 1])
                assert actual == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("gain", [2.0, -2.0])
def test_plain_correlation_removes_means_and_rounds_the_landing(monkeypatch, gain: float) -> None:
    source = np.array([-3, -1, 2, 4], dtype=np.float32)
    rendered = np.concatenate((np.zeros(2), source * gain + 17))
    original_source, original_rendered = source.copy(), rendered.copy()
    # At 10 Hz the halfway location rounds up to render sample 2.
    monkeypatch.setattr(audio, "locate", lambda *a, **kw: audio.Located(0.15, 0.42, 0.31))
    match = sync._locate_audio(source, rendered, 0.0, 0.4, 0.15, 10)
    assert match.time_s == 0.15
    assert match.corr == pytest.approx(np.sign(gain))
    assert match.whitened_corr == 0.42
    assert match.ambiguity == 0.31
    assert np.array_equal(source, original_source)
    assert np.array_equal(rendered, original_rendered)


def test_plain_correlation_handles_missing_slices_and_zero_energy() -> None:
    needle = np.array([1, 3, -2, 4], dtype=np.float32)
    assert sync._plain_corr(needle, needle, 10, 0) == pytest.approx(1)
    assert sync._plain_corr(needle, needle, 10, 0.1) == 0
    assert sync._plain_corr(needle, needle, 10, -0.1) == 0
    assert sync._plain_corr(np.ones(4), needle, 10, 0) == 0
    assert sync._plain_corr(needle, np.ones(4), 10, 0) == 0
    assert sync._plain_corr(np.array([]), needle, 10, 0) == 0


def test_geometry_and_perclip_clock(calibration) -> None:
    results, _, paths, _ = calibration
    for name, kind in (("clean", "full"), ("clean_crop", "crop"), ("fit", "fit")):
        assert all(row.mapping.kind == kind for row in results[name].segments)
    info = ffmpeg.probe(paths["perclip"])
    assert info.video_offset_s > 0
    assert len(video.grey_frames(paths["perclip"], w=64)) == info.video_frames == 576
    for row in results["perclip"].segments:
        assert row.video_landing_s == pytest.approx(info.video_offset_s + row.video_frame / 30)
    findings = [f for f in results["perclip"].findings if f.check == "sync.av_offset"]
    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity == "fail" and finding.segment == 2
    assert finding.evidence["segments"] == list(range(2, 9))
    assert "sound runs late" in finding.message
    assert "+32.0 ms per cut" in finding.message
    assert finding.suggested_action == (
        "Audio was encoded per clip and joined without re-encoding. "
        "Encode the audio once for the whole video."
    )
    drift = [f for f in results["perclip"].findings if f.check == "sync.drift"]
    assert len(drift) == 1 and drift[0].severity == "warn"
    assert drift[0].evidence["segments"] == list(range(1, 9))
    assert drift[0].evidence["drift_ms_per_cut"] == pytest.approx(32, abs=2)


def test_heads_and_tails_follow_source_edges(calibration) -> None:
    results, _, _, plan = calibration
    for name, result in results.items():
        for row, segment in zip(result.segments, plan.segments, strict=True):
            landing_shift = row.audio_vs_plan_ms / 1000
            assert row.head_s == pytest.approx(float(segment.out_time) + landing_shift, abs=0.001)
            if row.index == 8 and name in {"clean_shift_samples", "crop_shift", "fit_shift"}:
                # The samples method delays and then trims to the original length.
                # The final tail window no longer exists in full in this fixture.
                continue
            assert row.tail_s == pytest.approx(
                float(segment.out_time + segment.n_frames / plan.fps) + landing_shift,
                abs=0.001,
            )


@pytest.mark.parametrize("name", ["clean_shift_samples", "crop_shift", "fit_shift"])
def test_shifted_tail_window_is_truncated(calibration, synth_media, name: str) -> None:
    results, _, paths, plan = calibration
    row = results[name].segments[-1]
    sr, samples = audio.decode(paths[name], sr=16000)
    expected_tail = float(plan.total_frames / plan.fps) + row.audio_vs_plan_ms / 1000
    assert expected_tail - 0.03 > len(samples) / sr
    assert abs(row.tail_s - expected_tail) > 0.001
    source_name = "calib_vertical" if name.startswith("fit") else "calib"
    _, source = audio.decode(synth_media.path(source_name).resolve(), sr=sr)
    end = float(plan.segments[-1].src_f1 / plan.fps)
    first, last = (int(np.floor(t * sr + 0.5)) for t in (end - 0.33, end - 0.03))
    assert sync._plain_corr(source[first:last], samples, sr, expected_tail - 0.33) == 0
    # Plain correlation at a different, valid slice can be high for periodic audio.
    # Its exact value is checked separately; it cannot prove this tail survived.


def test_skipped_middle_still_measures_edges(calibration, synth_media) -> None:
    _, _, paths, plan = calibration
    segments = list(plan.segments)
    segments[0] = replace(segments[0], source_id=None)
    segments[1] = replace(segments[1], source_id="missing")
    segments[2] = replace(segments[2], n_frames=9, src_f1=segments[2].src_f0 + 9)
    changed = replace(plan, segments=segments)
    # Cover all of segment 4 (7.2 s to 9.6 s) so no window inside it is unmasked.
    masks = [(Fraction(7), Fraction(10))]
    result = sync.check_sync(
        paths["clean"], changed, {"main": synth_media.path("calib")}, masks=masks
    )
    assert [row.reason for row in result.segments[:4]] == [
        "gap",
        "source not found",
        "shorter than 0.4 s",
        "picture masked",
    ]
    for row in result.segments[2:4]:
        assert row.head_s is not None and row.tail_s is not None
    masked = result.segments[3]
    assert masked.audio_landing_s is not None
    assert masked.video_landing_s is None and masked.mapping is None
    finding = next(f for f in result.findings if f.check == "sync.unmeasured")
    assert len(finding.evidence["segments"]) == 4
    assert "segments 2, 3." in finding.message


def test_masked_middle_probes_the_nearest_unmasked_window(calibration, synth_media) -> None:
    results, _, paths, plan = calibration
    segment = plan.segments[3]
    # 8.0 s to 9.0 s hides segment 4's middle (8.4 s) but not the rest of it.
    masks = [(Fraction(8), Fraction(9))]
    result = sync.check_sync(paths["clean"], plan, {"main": synth_media.path("calib")}, masks=masks)
    row = result.segments[3]
    midpoint = segment.src_f0 + segment.n_frames // 2
    assert row.measured and row.reason is None
    assert row.probe_frame is not None and row.probe_frame != midpoint
    # The nearest clear window ends at 8.0 s: its last frame is before the mask.
    window_end = (segment.out_f0 + row.probe_frame - segment.src_f0 + 5) / 30
    assert window_end <= 8.0 + 1e-9
    assert window_end > 8.0 - 2 / 30
    assert row.video_vs_plan_frames == 0
    assert row.confidence == "high"
    assert row.av_offset_ms == pytest.approx(results["clean"].segments[3].av_offset_ms, abs=1)
    assert not any(f.check == "sync.unmeasured" for f in result.findings)


@pytest.fixture(scope="module")
def caption_band(synth_media):
    """The clean render with a moving band burned over its lower third for most of it."""

    def build(out: Path) -> None:
        ffmpeg.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(synth_media.path("clean")),
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x60:rate=30:duration=19.2",
                "-filter_complex",
                # between() is inclusive; stop before 18.7 s so the declared end is clear.
                "[0:v][1:v]overlay=0:100:enable='between(t,0.5,18.65)'[v]",
                "-map",
                "[v]",
                "-map",
                "0:a:0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                str(out),
            ],
            timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
        )

    return synth_media.cached("sync-caption-band-v2", build)


def test_declared_region_keeps_sync_measured_under_a_burned_band(
    calibration, synth_media, caption_band
) -> None:
    _, _, _, plan = calibration
    source = {"main": synth_media.path("calib")}
    region = PictureRegion(Fraction(1, 2), Fraction(187, 10), 0, 100, 320, 60)
    result = sync.check_sync(caption_band, plan, source, regions=[region])
    for row in result.segments:
        assert row.measured, row
        assert row.masked_region is True
        assert row.confidence == "high", row
        assert row.video_vs_plan_frames == 0
        assert row.av_offset_ms == pytest.approx(0, abs=1)
    assert not any(f.check == "sync.unmeasured" for f in result.findings)
    assert result.drift_ms_per_cut == pytest.approx(0, abs=0.5)

    # The same overlay without a region hides the whole picture for its duration:
    # only segments 1 and 8 find an unmasked window, at the render's edges.
    timed = sync.check_sync(caption_band, plan, source, masks=[(region.start, region.end)])
    reasons = [row.reason for row in timed.segments]
    assert reasons[1:7] == ["picture masked"] * 6
    assert all(row.video_landing_s is None for row in timed.segments[1:7])
    for row, segment in zip(
        (timed.segments[0], timed.segments[7]), (plan.segments[0], plan.segments[7]), strict=True
    ):
        assert row.measured and row.video_vs_plan_frames == 0
        assert row.probe_frame != segment.src_f0 + segment.n_frames // 2


def test_render_without_audio_still_measures_picture(calibration, synth_media) -> None:
    _, _, paths, plan = calibration

    def remove_audio(out: Path) -> None:
        ffmpeg.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(paths["clean"]),
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "copy",
                str(out),
            ],
            timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
        )

    render = synth_media.cached("sync-clean-no-audio", remove_audio)
    result = sync.check_sync(render, plan, {"main": synth_media.path("calib")})

    assert all(row.measured for row in result.segments)
    assert all(row.video_landing_s is not None for row in result.segments)
    assert all(row.video_vs_plan_frames == 0 for row in result.segments)
    audio_fields = (
        "audio_landing_s",
        "audio_vs_plan_ms",
        "corr",
        "whitened_corr",
        "ambiguity",
        "av_offset_ms",
        "head_s",
        "head_corr",
        "tail_s",
        "tail_corr",
        "confidence",
    )
    assert all(getattr(row, field) is None for row in result.segments for field in audio_fields)
    assert {row.reason for row in result.segments} == {"no audio in the render"}
    assert not any(f.check in {"sync.av_offset", "sync.drift"} for f in result.findings)
    unmeasured = [f for f in result.findings if f.check == "sync.unmeasured"]
    assert len(unmeasured) == 1
    assert unmeasured[0].message.count("no audio in the render") == 1


def test_source_without_audio_still_measures_picture(calibration, synth_media) -> None:
    _, _, paths, plan = calibration

    def remove_audio(out: Path) -> None:
        ffmpeg.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(synth_media.path("calib")),
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "copy",
                str(out),
            ],
            timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
        )

    source = synth_media.cached("sync-calib-no-audio", remove_audio, suffix=".mov")
    result = sync.check_sync(paths["clean"], plan, {"main": source})

    assert all(row.measured for row in result.segments)
    assert all(row.video_landing_s is not None for row in result.segments)
    assert all(row.video_vs_plan_frames == 0 for row in result.segments)
    assert all(row.audio_landing_s is None and row.av_offset_ms is None for row in result.segments)
    assert {row.reason for row in result.segments} == {"no audio in the source"}


def test_decode_is_cached_by_source_path(calibration, synth_media, monkeypatch) -> None:
    _, _, paths, plan = calibration
    calls = []
    decode = sync._decode

    def counted(path, sr, **kwargs):
        calls.append(Path(path).resolve())
        return decode(path, sr, **kwargs)

    monkeypatch.setattr(sync, "_decode", counted)
    segments = [replace(s, source_id="alias" if s.index % 2 else "main") for s in plan.segments]
    source = synth_media.path("calib")
    sync.measure_segments(
        paths["clean"], replace(plan, segments=segments), {"main": source, "alias": source}
    )
    assert calls.count(source.resolve()) == 1
    assert calls.count(paths["clean"].resolve()) == 1


@pytest.mark.parametrize("name", ["clean", "perclip", "clean_crop"])
def test_uint8_sync_landings_and_mappings_match_float_decodes(
    calibration, synth_media, monkeypatch, name
) -> None:
    results, _, paths, plan = calibration
    original = sync._decode

    def floating(path, sr, **kwargs):
        decoded = original(path, sr, **kwargs)
        assert decoded.frames.dtype == np.uint8
        return replace(decoded, frames=decoded.frames.astype(np.float32))

    monkeypatch.setattr(sync, "_decode", floating)
    measured = sync.check_sync(paths[name], plan, {"main": synth_media.path("calib")})
    assert measured == results[name]


def test_threshold_boundaries_runs_and_low_confidence(calibration, monkeypatch) -> None:
    results, _, paths, plan = calibration
    offsets = [-40, -41, 60, 61, 0, 100, 100, 0]
    rows = [
        replace(row, av_offset_ms=offset, confidence="low" if row.index == 6 else "high")
        for row, offset in zip(results["clean"].segments, offsets, strict=True)
    ]
    monkeypatch.setattr(sync, "measure_segments", lambda *a, **kw: rows)
    result = sync.check_sync(
        paths["clean"], plan, {}, thresholds=Thresholds(drift_warn_ms_per_cut=999)
    )
    runs = [f for f in result.findings if f.check == "sync.av_offset"]
    assert [f.evidence["segments"] for f in runs] == [[1, 2, 3, 4], [7]]
    assert all(f.severity == "fail" for f in runs)
    assert "-100 ms" in runs[1].suggested_action
    rows[:] = [
        replace(row, av_offset_ms=offset, video_vs_plan_frames=frame)
        for row, offset, frame in zip(
            rows, [-40, 60, 0, 0, 0, 0, 0, 0], [2, 3, 0, -2, -1, 0, 0, 0], strict=True
        )
    ]
    result = sync.check_sync(
        paths["clean"], plan, {}, thresholds=Thresholds(drift_warn_ms_per_cut=999)
    )
    assert next(f for f in result.findings if f.check == "sync.av_offset").severity == "warn"
    assert [f.evidence["segments"] for f in result.findings if f.check == "plan.picture"] == [
        [1, 2],
        [4],
    ]


def test_drift_uses_segment_index_and_excludes_low_confidence(calibration, monkeypatch) -> None:
    results, _, paths, plan = calibration
    rows = [replace(row, av_offset_ms=3 * row.index) for row in results["clean"].segments]
    rows[3] = replace(rows[3], av_offset_ms=1000, confidence="low")
    monkeypatch.setattr(sync, "measure_segments", lambda *a, **kw: rows)
    result = sync.check_sync(paths["clean"], plan, {})
    assert result.drift_ms_per_cut == pytest.approx(3)
    assert sum(f.check == "sync.drift" for f in result.findings) == 1

    # A two-point slope is reportable, but the drift warning needs three points.
    rows[:] = [replace(row, confidence="high" if row.index in {1, 8} else "low") for row in rows]
    result = sync.check_sync(paths["clean"], plan, {})
    assert result.drift_ms_per_cut == pytest.approx(3)
    assert not any(f.check == "sync.drift" for f in result.findings)


@pytest.mark.parametrize("missing", ["picture", "audio", "texture", "signal"])
def test_unmeasurable_media_reports_no_match(calibration, monkeypatch, missing: str) -> None:
    _, _, paths, plan = calibration
    plan = replace(plan, segments=plan.segments[:1], cuts=[])
    info = ffmpeg.probe(paths["clean"])
    rng = np.random.default_rng(73)
    frames = rng.normal(size=(120, 6, 8)).astype(np.float32)
    sound = rng.normal(size=4 * 16000).astype(np.float32)
    source = sync._Media(info, frames, sound)
    if missing == "picture":
        source = replace(source, frames=None)
    elif missing == "audio":
        source = replace(source, sound=None)
    elif missing == "texture":
        source = replace(source, frames=np.zeros_like(frames))
    else:
        source = replace(source, sound=np.zeros_like(sound))

    def decoded(path, sr, **kwargs):
        if path == paths["clean"]:
            return sync._Media(info, frames[12:84], sound[6400:44800])
        return source

    monkeypatch.setattr(sync, "_decode", decoded)
    result = sync.check_sync(paths["clean"], plan, {"main": paths["perclip"]})
    row = result.segments[0]
    assert row.av_offset_ms is None
    assert [f.check for f in result.findings] == ["sync.unmeasured"]
    if missing == "audio":
        assert row.measured and row.reason == "no audio in the source"
        assert row.video_landing_s is not None
        assert row.audio_landing_s is None and row.head_s is None
    else:
        assert not row.measured and row.reason == "no match"
    if missing in {"picture", "texture"}:
        assert row.audio_landing_s is not None and row.head_s is not None
    elif missing == "signal":
        assert row.corr == row.whitened_corr == 0 and row.ambiguity == 1


def test_model_contract() -> None:
    assert CHECK_IDS == (
        "join.click",
        "join.gap",
        "join.words",
        "join.frames",
        "join.listen",
        "join.relisten",
        "join.second_model",
        "sync.av_offset",
        "sync.drift",
        "sync.unmeasured",
        "plan.picture",
        "plan.frames",
        "plan.samples",
        "plan.holds",
        "edit_list.evidence",
        "structure.av_length",
        "structure.audio_timestamps",
        "loudness.integrated",
        "loudness.true_peak",
        "loudness.parts",
        "picture.black",
        "picture.freeze",
        "picture.flash",
        "picture.transition",
        "picture.grade",
        "captions.timing",
        "captions.cards",
        "captions.contrast",
        "captions.words",
        "faces.clearance",
        "detect.cuts",
        "platform.shape",
        "platform.duration",
        "platform.resolution",
        "platform.fps",
        "taste.rules",
    )
    assert len(CHECK_IDS) == len(set(CHECK_IDS))
    with pytest.raises(ValueError, match="unknown check"):
        Finding("not.a.check", "warn", "Unknown")
    with pytest.raises(ValueError):
        Finding("sync.drift", "warn", "Invalid", evidence={"offset": float("inf")})
    one = Finding("sync.drift", "info", "Measured")
    two = Finding("sync.drift", "warn", "Drifting")
    three = Finding("sync.av_offset", "fail", "Late")
    one.evidence["x"] = 1
    assert two.evidence == {}
    assert status_of([]) == status_of([one]) == "pass"
    assert status_of([one, two]) == "warn"
    assert status_of([two, three]) == "fail"
    original = Thresholds()
    assert original.with_target(-16).loudness_target_lufs == -16
    assert original.loudness_target_lufs == -14
    assert [format_time(t) for t in (7.14, 72.0, 59.99, 7.19)] == [
        "0:07.1",
        "1:12.0",
        "0:59.9",
        "0:07.1",
    ]
    assert json.loads(json.dumps(one.to_dict())) == one.to_dict()


def test_different_source_rate_and_nonzero_container_offsets(monkeypatch, calibration) -> None:
    _, _, paths, plan = calibration
    plan = replace(plan, segments=plan.segments[:1], cuts=[])
    source_info = replace(ffmpeg.probe(paths["clean"]), fps=Fraction(60), video_offset_s=0.2)
    render_info = replace(source_info, fps=Fraction(30), video_offset_s=0.1)
    frames = np.random.default_rng(71).normal(size=(240, 6, 8)).astype(np.float32)
    source_audio = np.random.default_rng(72).normal(size=4 * 16000).astype(np.float32)
    segment = plan.segments[0]
    render_frames = frames[segment.src_f0 * 2 : segment.src_f1 * 2 : 2]
    start = round((segment.src_f0 / 30 + 0.2) * 16000)
    render_audio = np.concatenate((np.zeros(1600), source_audio[start : start + 38400]))

    def decoded(path, sr, **kwargs):
        if path == paths["clean"]:
            return sync._Media(render_info, render_frames, render_audio)
        return sync._Media(source_info, frames, source_audio)

    monkeypatch.setattr(sync, "_decode", decoded)
    rows = sync.measure_segments(paths["clean"], plan, {"main": paths["perclip"]})
    assert rows[0].confidence == "high"
    assert rows[0].video_frame == 36
    assert rows[0].av_offset_ms == pytest.approx(0, abs=0.1)
    assert rows[0].audio_vs_plan_ms == pytest.approx(100, abs=0.1)
