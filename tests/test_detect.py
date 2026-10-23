"""Cut discovery, render-only checks, and calibration margins."""

from __future__ import annotations

import json
import time
import wave
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import synth

from yap_studio import audio, ffmpeg, video
from yap_studio.checks import detect, joins
from yap_studio.checks.model import Thresholds
from yap_studio.errors import ValidationError
from yap_studio.transcript import ScriptedBackend, Transcript, Word

_FRAMES = list(range(72, 505, 72))
_RENDERS = (
    "clean",
    "perclip",
    "perclip_click3",
    "clean_crop",
    "perclip_click3_crop",
    "slow_clean",
    "slow_perclip",
)


@pytest.fixture(scope="module")
def calibration(synth_media):
    paths = {name: synth_media.path(name) for name in (*_RENDERS[:5], "calib", "slowpan")}
    for name, render in (
        ("slow_clean", synth.render_clean),
        ("slow_perclip", synth.render_perclip_aac),
    ):
        paths[name] = synth_media.cached(
            f"t9-{name}",
            lambda out, render=render: render(paths["slowpan"], synth_media.spans, out),
        )
    cuts, diagnostics = {}, {}
    for name, path in paths.items():
        start = time.perf_counter()
        cuts[name] = detect.detect_cuts(path)
        seconds = time.perf_counter() - start
        frames = video.grey_frames(path, w=160)
        evidence = detect._picture_evidence(frames, 30)
        expected = _FRAMES if name in _RENDERS else []
        rejected = [row for row in evidence if row["frame"] not in expected]
        flashes = set(np.flatnonzero(frames.mean(axis=(1, 2)) > 245).tolist())
        flash_edges = flashes | {frame + 1 for frame in flashes}
        flash_scores = [row["score"] for row in evidence if row["frame"] in flash_edges]
        diagnostics[name] = {
            "cuts": [cut.to_dict() for cut in cuts[name]],
            "runtime_s": seconds,
            "closest_false_positive": max(rejected, key=lambda row: row["score"]),
            "max_flash_score": max(flash_scores, default=None),
            "flash_edges": sorted(flash_edges),
        }
    report = synth_media.root / "t9-detection-report.json"
    report.write_text(json.dumps(diagnostics, indent=2, allow_nan=False))
    return SimpleNamespace(paths=paths, cuts=cuts, diagnostics=diagnostics, report=report)


@pytest.mark.parametrize("name", _RENDERS)
def test_all_cuts_without_flash_or_motion_false_positives(calibration, name):
    cuts = calibration.cuts[name]
    assert len(cuts) == 7, cuts
    assert [cut.index for cut in cuts] == list(range(1, 8))
    info = ffmpeg.probe(calibration.paths[name], count_frames=False)
    for cut, expected in zip(cuts, _FRAMES, strict=True):
        assert abs(cut.frame - expected) <= 1
        assert cut.time_s == pytest.approx(info.video_offset_s + cut.frame / float(info.fps))
        assert 0 <= cut.confidence <= 1
        assert cut.confidence == pytest.approx(1 - np.exp(-cut.evidence["score"] / 5))
        assert cut.evidence["before_z"] >= 5 and cut.evidence["after_z"] >= 5
        assert cut.frame not in calibration.diagnostics[name]["flash_edges"]
    assert calibration.diagnostics[name]["closest_false_positive"]["score"] < 5
    flash_score = calibration.diagnostics[name]["max_flash_score"]
    assert flash_score is None or flash_score < 5


@pytest.mark.parametrize("name", ["calib", "slowpan"])
def test_uncut_sources_have_no_cuts(calibration, name):
    assert calibration.cuts[name] == []
    assert calibration.diagnostics[name]["closest_false_positive"]["score"] < 5


@pytest.mark.parametrize("shape", [(36, 64), (96, 54), (40, 40)])
def test_returning_flash_and_glitch_are_rejected_but_a_new_shot_persists(monkeypatch, shape):
    rng = np.random.default_rng(91)
    first, second, glitch = rng.integers(0, 256, size=(3, *shape)).astype(np.float32)
    pictures = np.repeat(first[None], 150, axis=0)
    pictures[20] = 255
    pictures[40] = glitch
    pictures[80:] = second
    pictures[110] = 0
    monkeypatch.setattr(
        ffmpeg,
        "probe",
        lambda *a, **kw: SimpleNamespace(
            has_video=True,
            fps=Fraction(30),
            video_offset_s=0.125,
        ),
    )
    monkeypatch.setattr(video, "grey_frames", lambda *a, **kw: pytest.fail("decoded twice"))
    cuts = detect.detect_cuts(Path("supplied.mp4"), frames=pictures)
    assert [cut.frame for cut in cuts] == [80]
    assert cuts[0].time_s == pytest.approx(0.125 + 80 / 30)


def test_neighbouring_candidates_merge_to_the_strongest(monkeypatch):
    rng = np.random.default_rng(2)
    a, b, c = rng.integers(0, 256, size=(3, 30, 40)).astype(np.float32)
    pictures = np.repeat(a[None], 100, axis=0)
    pictures[50] = b
    pictures[51:] = c
    monkeypatch.setattr(
        ffmpeg,
        "probe",
        lambda *a, **kw: SimpleNamespace(
            has_video=True,
            fps=Fraction(30),
            video_offset_s=0,
        ),
    )
    rows = [row for row in detect._picture_evidence(pictures, 30) if row["score"] >= 5]
    assert [row["frame"] for row in rows] == [50, 51]
    cuts = detect.detect_cuts(Path("supplied.mp4"), frames=pictures)
    assert len(cuts) == 1
    assert cuts[0].frame == max(rows, key=lambda row: row["score"])["frame"]


def test_empty_short_and_invalid_picture_arrays():
    assert detect._picture_evidence(np.zeros((0, 36, 64)), 30) == []
    assert detect._picture_evidence(np.zeros((3, 36, 64)), 30) == []
    assert not [
        row for row in detect._picture_evidence(np.zeros((60, 36, 64)), 30) if row["score"] >= 5
    ]
    with pytest.raises(ValidationError):
        detect._picture_evidence(np.zeros((60, 36)), 30)
    with pytest.raises(ValidationError):
        detect._picture_evidence(np.full((60, 36, 64), np.nan), 30)


@pytest.fixture(scope="module")
def checked(calibration, tmp_path_factory):
    directory = tmp_path_factory.mktemp("detect-checked")
    results, seconds = {}, {}
    for name in _RENDERS:
        start = time.perf_counter()
        results[name] = detect.check_without_edl(
            calibration.paths[name],
            backend=None,
            work_dir=directory,
        )
        seconds[name] = time.perf_counter() - start
    report = directory / "t9-checks-report.json"
    report.write_text(
        json.dumps(
            {
                "runtime_s": seconds,
                "results": {name: result.to_dict() for name, result in results.items()},
            },
            indent=2,
            allow_nan=False,
        )
    )
    return results


@pytest.fixture(scope="module")
def speech_detection(tmp_path_factory):
    directory = tmp_path_factory.mktemp("detect-speech")
    source, words = synth.speech_source(
        directory / "speech.mov",
        [
            "Bright planets circle slowly over quiet hills while silver rivers carry "
            "morning light through every open valley and garden path near home."
        ],
    )
    middles = [(word.start + word.end) / 2 for word in words]
    spans = [
        (middles[0], middles[4]),
        (middles[6], middles[10]),
        (middles[12], middles[16]),
        (middles[18], middles[21]),
    ]
    clean = synth.render_clean(source, spans, directory / "clean.mp4")
    clean_result = detect.check_without_edl(clean, backend=None, work_dir=directory)
    assert len(clean_result.cuts) == 3

    planted_cut = clean_result.cuts[1]
    sr, sound = audio.decode(clean)
    sample = round(planted_cut.time_s * sr)
    sound[sample] = np.clip(sound[sample - 1] + (0.5 if sound[sample - 1] <= 0 else -0.5), -1, 1)
    wav = directory / "planted.wav"
    with wave.open(str(wav), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sr)
        stream.writeframes((np.clip(sound, -1, 1) * 32767).round().astype("<i2").tobytes())
    planted = directory / "planted.mov"
    ffmpeg.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(clean),
            "-i",
            str(wav),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "pcm_s16le",
            str(planted),
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    planted_result = detect.check_without_edl(planted, backend=None, work_dir=directory)
    return SimpleNamespace(
        clean=clean_result,
        planted=planted_result,
        planted_cut=planted_cut.index,
    )


@pytest.mark.parametrize("name", _RENDERS)
def test_planted_click_is_at_cut_three_only(checked, name):
    result = checked[name]
    assert len(result.results) == 7
    clicks = [f for f in result.findings if f.check == "join.click"]
    assert [f.cut for f in clicks] == ([3] if "click3" in name else [])
    summary = next(f for f in result.findings if f.check == "detect.cuts")
    searches = summary.evidence["audio_searches"]
    for row, search in zip(result.results, searches, strict=True):
        assert search["cut"] == row.cut.index
        assert search["click_db"] == row.click_db
        assert search["join_s"] == row.join_s
        assert search["threshold_db"] == 6
        assert search["outlier_db"] == 6
        assert search["reference_candidates"] >= 0
        assert search["window_s"] == pytest.approx([row.cut.time_s - 0.1, row.cut.time_s + 0.35])
        reference = search["reference_db"]
        expected = row.click_db >= 6 and (reference is None or row.click_db >= reference + 6)
        assert search["found_by_search"] is expected
        if not expected:
            assert row.join_s == row.cut.time_s
        else:
            assert row.join_s == search["candidate_s"]
    for finding in clicks:
        assert finding.severity == "fail"
        assert "Cut 3 at 0:07.2: click at the join (+" in finding.message
        assert "dB above the sound around it)." in finding.message
        assert finding.suggested_action == (
            "Re-render this cut with a short audio fade (5 to 30 ms)."
        )
        assert finding.evidence["found_by_search"] is True
        assert finding.evidence["threshold_db"] == 6
        assert finding.evidence["click_db"] == result.results[finding.cut - 1].click_db


def test_running_speech_does_not_report_clicks(speech_detection):
    result = speech_detection.clean
    assert not [finding for finding in result.findings if finding.check == "join.click"]
    searches = next(
        finding for finding in result.findings if finding.check == "detect.cuts"
    ).evidence["audio_searches"]
    assert len(searches) == 3
    assert all(search["reference_candidates"] >= 10 for search in searches)
    assert all(search["reference_db"] is not None for search in searches)
    assert all(
        search["click_db"] < search["reference_db"] + detect._CLICK_OUTLIER_DB
        for search in searches
    )


def test_planted_click_stands_out_from_running_speech(speech_detection):
    result = speech_detection.planted
    clicks = [finding for finding in result.findings if finding.check == "join.click"]
    assert [finding.cut for finding in clicks] == [speech_detection.planted_cut]
    evidence = clicks[0].evidence
    assert evidence["click_db"] >= 6
    assert evidence["click_db"] >= evidence["reference_db"] + detect._CLICK_OUTLIER_DB
    assert "dB above the sound around it)." in clicks[0].message


def test_structure_parts_and_skipped_checks_are_reported(checked):
    result = checked["clean"]
    assert len(result.structure.parts_lufs) == 8
    assert all(level is not None for level in result.structure.parts_lufs)
    assert result.structure.frames == 576
    assert result.structure.sample_rate == 48000
    assert result.checks["loudness.parts"] == {"ran": True, "reason": None}
    for check in (
        "sync.av_offset",
        "sync.drift",
        "plan.picture",
        "plan.frames",
        "plan.samples",
        "join.words",
        "join.gap",
    ):
        assert result.checks[check] == {"ran": False, "reason": "needs the source and edit list"}
    assert result.checks["join.listen"] == {"ran": False, "reason": "no speech engine"}
    assert result.checks["detect.cuts"] == {
        "ran": True,
        "reason": None,
        "note": "Jump cuts inside one take may be missed; pass the edit list for a full check.",
    }
    assert all(row.listen is None for row in result.results)
    summary = next(f for f in result.findings if f.check == "detect.cuts")
    assert summary.severity == "info"
    assert summary.message == (
        "7 clear cuts found in the picture. Jump cuts inside one take may be missed; "
        "pass the edit list for a full check."
    )
    assert summary.evidence["cuts"] == [cut.to_dict() for cut in result.cuts]
    assert summary.evidence["jump_cuts_may_be_missed"] is True
    assert all(f in result.findings for f in result.structure.findings)
    json.dumps(result.to_dict(), allow_nan=False)


def test_audio_placement_searches_delayed_click_and_falls_back_in_silence():
    sr = 48000
    sound = np.zeros(2 * sr, dtype=np.float32)
    join, evidence = detect._place_join(sound, sr, sound, 1.0)
    assert join == 1.0
    assert evidence["found_by_search"] is False
    assert evidence["candidate_s"] is None
    assert evidence["click_db"] == -120
    sound[int(1.2 * sr)] = 0.9
    difference = np.r_[0.0, abs(np.diff(sound.astype(np.float64)))]
    join, evidence = detect._place_join(sound, sr, difference, 1.0)
    assert join == 1.2
    assert evidence["found_by_search"] is True
    assert evidence["click_db"] == audio.click_db(sound, int(join * sr), sr)
    with pytest.raises(ValidationError):
        detect._place_join(sound, sr, difference, 3.0)


@pytest.mark.parametrize("level,found", [(3.0, False), (5.999, False), (6.0, True)])
def test_search_threshold_keeps_subthreshold_evidence(monkeypatch, level, found):
    sr = 48000
    sound = np.zeros(2 * sr, dtype=np.float32)
    sound[int(1.25 * sr)] = 0.8
    difference = np.r_[0.0, abs(np.diff(sound.astype(np.float64)))]
    monkeypatch.setattr(detect, "_click_ratio", lambda *args: level)
    join, evidence = detect._place_join(sound, sr, difference, 1.0)
    assert join == (1.25 if found else 1.0)
    assert evidence["click_db"] == level
    assert evidence["candidate_s"] == 1.25
    assert evidence["found_by_search"] is found


def test_search_limits_candidates_and_selects_best_ratio(monkeypatch):
    sr = 48000
    sound = np.zeros(2 * sr, dtype=np.float32)
    indices = sr + np.arange(30) * 288
    sound[indices] = np.linspace(0.1, 0.9, 30)
    sound[indices[-1] + 48] = 0.85  # Within 5 ms of the stronger final peak.
    difference = np.r_[0.0, abs(np.diff(sound.astype(np.float64)))]
    evaluated = []

    def ratio(sound, sr, index):
        evaluated.append(index)
        return 8.0 if index == indices[-2] else 3.0

    monkeypatch.setattr(detect, "_click_ratio", ratio)
    join, evidence = detect._place_join(sound, sr, difference, 1.0)
    assert sorted(evaluated) == indices[-20:].tolist()
    assert join == indices[-2] / sr
    assert evidence["click_db"] == 8
    assert evidence["candidates_evaluated"] == 20


def test_local_reference_reports_only_the_larger_step():
    sr = 8000
    sound = np.zeros(10 * sr, dtype=np.float32)
    regular = np.arange(sr // 2, len(sound), sr // 4)
    sound[regular] = 0.05
    sound[3 * sr] = 0.8
    difference = np.r_[0.0, abs(np.diff(sound.astype(np.float64)))]

    found = []
    evidence_by_cut = {}
    for cut_s in (3.0, 7.0):
        _, evidence = detect._place_join(sound, sr, difference, cut_s)
        evidence_by_cut[cut_s] = evidence
        if evidence["found_by_search"]:
            found.append(cut_s)

    assert found == [3.0]
    larger = evidence_by_cut[3.0]
    assert larger["reference_candidates"] == 15
    assert larger["click_db"] >= larger["reference_db"] + detect._CLICK_OUTLIER_DB
    regular = evidence_by_cut[7.0]
    assert regular["reference_candidates"] == 15
    assert regular["click_db"] == pytest.approx(regular["reference_db"])


def test_audio_click_without_picture_cut_is_not_reported(synth_media, tmp_path):
    def build(out):
        sr = 48000
        sound = np.zeros(5 * sr, dtype=np.float32)
        sound[int(2.4 * sr)] = 0.9
        wav = tmp_path / "click.wav"
        with wave.open(str(wav), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(sr)
            stream.writeframes((sound * 32767).astype("<i2").tobytes())
        ffmpeg.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(synth_media.path("calib")),
                "-i",
                str(wav),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-t",
                "5",
                "-c:v",
                "copy",
                "-c:a",
                "pcm_s16le",
                str(out),
            ],
            timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
        )

    path = synth_media.cached("t9-audio-only", build, suffix=".mov")
    result = detect.check_without_edl(path, backend=None, work_dir=tmp_path)
    assert result.cuts == []
    assert not [f for f in result.findings if f.check == "join.click"]
    assert len(result.structure.parts_lufs) == 1


def test_render_without_audio_still_checks_picture_and_structure(
    synth_media, calibration, tmp_path
):
    def build(out):
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
                "-c:v",
                "copy",
                "-an",
                str(out),
            ],
            timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
        )

    path = synth_media.cached("t9-clean-no-audio", build)
    result = detect.check_without_edl(path, backend=None, work_dir=tmp_path)
    assert [cut.frame for cut in result.cuts] == [cut.frame for cut in calibration.cuts["clean"]]
    assert len(result.results) == 7
    assert all(row.join_s == row.cut.time_s for row in result.results)
    assert all(row.click_db is None and row.pause_s is None for row in result.results)
    assert result.checks["join.click"] == {
        "ran": False,
        "reason": "no audio in the render",
    }
    assert result.checks["join.frames"] == {"ran": True, "reason": None}
    assert not [finding for finding in result.findings if finding.check == "join.click"]
    assert result.structure.frames == 576
    assert result.structure.sample_rate is None
    assert result.structure.findings


@pytest.mark.parametrize("confidence,expected", [(0.2, True), (0.95, False)])
def test_speech_confidence_with_one_batch(calibration, tmp_path, confidence, expected):
    batches = []

    def scripted(path):
        with wave.open(str(path), "rb") as stream:
            sr = stream.getframerate()
            assert stream.getnframes() == int(5.5 * sr)
            samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
        assert not samples[:sr].any() and not samples[-sr // 2 :].any()
        at_cut = confidence if path.stem.endswith("04") else 0.95
        return Transcript(
            words=[
                Word("before", 1.3, 1.6, 0.95),
                Word("across", 2.9, 3.1, at_cut),
                Word("after", 4.1, 4.4, 0.95),
            ]
        )

    class BatchBackend(ScriptedBackend):
        def transcribe_many(self, paths, *, language=None):
            batches.append(list(paths))
            return super().transcribe_many(paths, language=language)

    result = detect.check_without_edl(
        calibration.paths["clean"],
        backend=BatchBackend(scripted),
        work_dir=tmp_path,
    )
    assert len(batches) == 1 and len(batches[0]) == 7
    assert result.results[3].listen is expected
    assert all(row.listen is False for i, row in enumerate(result.results) if i != 3)
    findings = [f for f in result.findings if f.check == "join.listen"]
    assert [f.cut for f in findings] == ([4] if expected else [])
    if expected:
        assert findings[0].message == (
            "Cut 4 at 0:09.6: the speech engine is unsure across this cut; listen to it."
        )
    assert result.checks["join.listen"] == {"ran": True, "reason": None}


@pytest.mark.parametrize(
    "words",
    [
        [],
        [Word("short", 2.9, 3.1, 0.2)],
        [Word("a", 1.5, 2, None), Word("b", 2.9, 3.1, None), Word("c", 4, 4.5, None)],
        [Word("a", 1.5, 2, 0.9), Word("b", 2.9, 3.1, None), Word("c", 4, 4.5, 0.9)],
    ],
)
def test_missing_speech_confidence_is_unmeasured(words):
    assert detect._listen(Transcript(words=words))[0] is None


@pytest.mark.parametrize(
    "at_cut,context,expected",
    [
        (0.5, 0.95, False),
        (0.2, 0.49, False),
        (0.2, 0.5, True),
    ],
)
def test_confidence_drop_requires_both_limits(at_cut, context, expected):
    transcript = Transcript(
        words=[
            Word("a", 1.1, 2, context),
            Word("b", 2.6, 2.8, at_cut),
            Word("c", 4, 4.5, context),
            Word("silence", 0.1, 0.2, 0.01),
        ]
    )
    assert detect._listen(transcript)[0] is expected


def test_listen_window_at_start_keeps_join_at_three_seconds(tmp_path):
    path = tmp_path / "edge.wav"
    sound = np.full(4 * 8000, 0.25, dtype=np.float32)
    detect._listen_wav(path, sound, 8000, 0.2)
    with wave.open(str(path), "rb") as stream:
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
    assert len(samples) == 44000
    assert not samples[:22400].any()
    assert samples[22400:40000].min() > 0
    assert not samples[40000:].any()


def test_boundary_frames_thresholds_and_structure_wiring(calibration, monkeypatch, tmp_path):
    path = calibration.paths["clean"]
    cut = calibration.cuts["clean"][0]
    original_detect = detect.detect_cuts
    monkeypatch.setattr(detect, "detect_cuts", lambda *a, **kw: original_detect(*a, **kw)[:1])
    original_frames = video.grey_frames
    original_blocks = video.iter_grey_frames
    decode_widths = []

    def blocks(path, *, w):
        decode_widths.append(w)
        yield from original_blocks(path, w=w)

    def frames(path, *, w):
        decode_widths.append(w)
        pictures = original_frames(path, w=w)
        if w == 64:
            pictures[cut.frame : cut.frame + 2] = 0
        return pictures

    monkeypatch.setattr(video, "grey_frames", frames)
    monkeypatch.setattr(video, "iter_grey_frames", blocks)
    original_structure = detect.structure.check_structure
    calls = []

    def structure(render, **kwargs):
        calls.append(kwargs)
        return original_structure(render, **kwargs)

    monkeypatch.setattr(detect.structure, "check_structure", structure)
    thresholds = replace(Thresholds(), click_fail_db=100, loudness_target_lufs=-20)
    result = detect.check_without_edl(path, backend=None, thresholds=thresholds, work_dir=tmp_path)
    row = result.results[0]
    assert row.duplicate and row.black
    messages = [f.message for f in result.findings if f.check == "join.frames"]
    assert len(messages) == 2
    assert any("a frame is repeated at the cut" in message for message in messages)
    assert any("a black frame shows at the cut" in message for message in messages)
    assert calls == [{"cuts_s": [row.join_s], "thresholds": thresholds}]
    assert decode_widths == [160, 64]
    sr, sound = audio.decode(path)
    assert row.pause_s == joins.pause_len_at(sound, sr, row.join_s)
    difference = np.r_[0.0, abs(np.diff(sound.astype(np.float64)))]
    join, search = detect._place_join(sound, sr, difference, cut.time_s)
    assert row.join_s == join
    assert row.click_db == search["click_db"]
    assert not [f for f in result.findings if f.check == "join.click" and f.cut is not None]
    summary = next(f for f in result.findings if f.check == "detect.cuts")
    assert summary.message.startswith("1 clear cut found in the picture.")


def _low_motion_shots(count: int, cuts: list[int], shape=(36, 64), step=0.015) -> np.ndarray:
    """Shots of a still textured picture with sensor-like noise, cut at ``cuts``.

    The noise is scaled so neighbouring frames differ by about ``step``, the
    motion of a locked-off talking head at 64 px.
    """
    rng = np.random.default_rng(77)
    starts = [0, *cuts]
    frames = np.empty((count, *shape), dtype=np.float32)
    for shot, first in enumerate(starts):
        last = starts[shot + 1] if shot + 1 < len(starts) else count
        base = rng.uniform(40, 215, shape)
        # Mean |a - b| of two independent normals is 1.128 sigma; normalise by contrast.
        sigma = step * base.std() / 1.128
        frames[first:last] = base + rng.normal(0, sigma, (last - first, *shape))
    return frames


def test_low_motion_step_at_a_real_cut_is_not_a_repeated_frame():
    frames = _low_motion_shots(120, [60])
    step = joins.frame_step_at(frames, 60, radius=30)
    # The old fixed limit (0.02) would call the step out of the cut frame a repeat.
    assert step["distances"]["right"] < 0.02
    assert step["distances"]["right"] == pytest.approx(0.015, abs=0.003)
    assert step["baselines"]["right"] == pytest.approx(0.015, abs=0.003)
    assert step["duplicate"] is False
    assert step["rule"] == {
        "identical_below": joins.REPEAT_IDENTICAL,
        "ratio_to_local_motion": joins.REPEAT_RATIO,
        "maximum": joins.REPEAT_MAX,
    }


def test_exactly_repeated_frame_at_a_cut_is_flagged():
    frames = _low_motion_shots(120, [60])
    frames[61] = frames[60]
    step = joins.frame_step_at(frames, 60, radius=30)
    assert step["distances"]["right"] == pytest.approx(0, abs=1e-6)
    assert step["duplicate"] is True
    # A codec-encoded repeat is small but not zero: far below the local motion.
    encoded = _low_motion_shots(120, [60], step=0.03)
    encoded[61] = encoded[60] + np.random.default_rng(3).normal(0, 0.15, encoded[60].shape)
    near = joins.frame_step_at(encoded, 60, radius=30)
    assert joins.REPEAT_IDENTICAL < near["distances"]["right"] < 0.25 * near["baselines"]["right"]
    assert near["duplicate"] is True


def test_detected_cuts_in_low_motion_footage_flag_only_the_real_repeat(
    calibration, monkeypatch, tmp_path
):
    path = calibration.paths["clean"]
    cuts = [cut.frame for cut in calibration.cuts["clean"]]
    original = video.grey_frames
    repeat = {"at": None}

    def frames(path, *, w):
        pictures = original(path, w=w)
        if w != 64:
            return pictures
        low = _low_motion_shots(len(pictures), cuts, shape=pictures.shape[1:])
        if repeat["at"] is not None:
            low[repeat["at"] + 1] = low[repeat["at"]]
        return low

    monkeypatch.setattr(video, "grey_frames", frames)
    result = detect.check_without_edl(path, backend=None, work_dir=tmp_path)
    assert not [f for f in result.findings if "repeated" in f.message]
    assert not any(row.duplicate for row in result.results)
    repeat["at"] = cuts[2]
    result = detect.check_without_edl(path, backend=None, work_dir=tmp_path)
    flagged = [f for f in result.findings if "repeated" in f.message]
    assert [f.cut for f in flagged] == [3]
    assert flagged[0].evidence["baselines"]["right"] > 0.01


@pytest.mark.parametrize("fixture", ["clean", "perclip", "clean_crop"])
def test_streaming_detection_exactly_matches_full_float_array(synth_media, monkeypatch, fixture):
    path = synth_media.path(fixture)
    full = video.grey_frames(path, w=160).astype(np.float32)
    fps = float(ffmpeg.probe(path, count_frames=False).fps)
    expected = detect.detect_cuts(path, frames=full)
    expected_evidence = detect._picture_evidence(full, fps)
    original = video.iter_grey_frames
    monkeypatch.setattr(
        video, "iter_grey_frames", lambda path, *, w: original(path, w=w, block=16)
    )
    assert list(detect._stream_picture_evidence(path, fps)) == expected_evidence
    assert detect.detect_cuts(path) == expected
    assert len(expected) == 7
    assert any(cut.frame % 16 == 0 for cut in expected)


@pytest.mark.parametrize("block", [1, 2, 16, 64])
@pytest.mark.parametrize("fps", [2, 29.97, 60])
def test_streaming_evidence_edges_short_inputs_and_bounded_overlap(monkeypatch, block, fps):
    rng = np.random.default_rng(32)
    original = detect._picture_evidence
    sizes = []

    def measured(frames, rate):
        sizes.append(len(frames))
        return original(frames, rate)

    monkeypatch.setattr(detect, "_picture_evidence", measured)
    for count in (0, 1, 2, 3, 4, 17, 31, 32, 33, 48, 65, 80, 160):
        frames = rng.integers(0, 256, (count, 6, 8), dtype=np.uint8)

        def blocks(*args, frames=frames, **kwargs):
            for first in range(0, len(frames), block):
                yield first, frames[first : first + block]

        monkeypatch.setattr(video, "iter_grey_frames", blocks)
        expected = original(frames.astype(np.float32), fps)
        assert list(detect._stream_picture_evidence(Path("unused"), fps)) == expected
    radius = detect._reference_radius(fps)
    assert max(sizes) <= block + max(2, radius + 1) + max(1, radius)


def test_check_without_edl_never_decodes_full_160px_frames(synth_media, monkeypatch, tmp_path):
    path = synth_media.path("clean")
    original = video.grey_frames
    streamed = video.iter_grey_frames
    calls = []

    def bounded(path, *, w, **kwargs):
        assert w != 160
        return original(path, w=w, **kwargs)

    def blocks(path, *, w):
        calls.append(w)
        yield from streamed(path, w=w, block=16)

    monkeypatch.setattr(video, "grey_frames", bounded)
    monkeypatch.setattr(video, "iter_grey_frames", blocks)
    result = detect.check_without_edl(path, backend=None, work_dir=tmp_path)
    assert len(result.cuts) == 7
    assert calls == [160]
