"""Take loudness matching over retained speech and its cut receipt."""

from __future__ import annotations

import copy
import json
import wave
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import synth
from test_cut import make_project, time_segment, write_edit, write_json

from yap_studio import audio, ffmpeg
from yap_studio.project.status import ProjectContext, cut_inputs, status
from yap_studio.stages import cut as cut_stage


@pytest.fixture(scope="module")
def speech_takes(tmp_path_factory: pytest.TempPathFactory) -> tuple:
    directory = tmp_path_factory.mktemp("take-speech")
    main, main_spans = synth.speech_source(directory / "main.mov", [
        "Bright planets circle slowly over quiet hills on calm winter evenings",
        "Seven green boxes wait beside the open door near our garden",
    ])
    pickup, pickup_spans = synth.speech_source(directory / "pickup.mov", [
        "Small birds gather slowly near quiet fields on warm summer mornings",
    ])
    quiet = directory / "quiet.mov"
    ffmpeg.run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(pickup),
        "-af", "volume=-8dB", "-c:v", "copy", "-c:a", "pcm_s16le", str(quiet),
    ], timeout_s=60)
    return main, main_spans, quiet, pickup_spans


def speech_project(tmp_path: Path, speech_takes: tuple) -> Path:
    """Keep two main sentences around a quieter pickup sentence."""
    main, main_spans, pickup, pickup_spans = speech_takes
    root = make_project(tmp_path / "project", [main, pickup], word_rows=[
        [(word.text, word.start, word.end) for word in spans]
        for spans in (main_spans, pickup_spans)
    ])
    ranges = []
    for take, spans, sentence in [("t1", main_spans, 0), ("t2", pickup_spans, 0),
                                   ("t1", main_spans, 1)]:
        words = [word for word in spans if word.sentence == sentence]
        ranges.append(time_segment(f"s{len(ranges) + 1}", take,
                                   words[0].start - 0.1, words[-1].end + 0.1))
    write_edit(root, ranges)
    return root


@pytest.fixture
def no_asr(monkeypatch: pytest.MonkeyPatch) -> None:
    real_verify = cut_stage.verify
    monkeypatch.setattr(cut_stage, "verify",
                        lambda **kwargs: real_verify(**{**kwargs, "asr": "none"}))


def plain_voice(root: Path) -> np.ndarray:
    """Return the original cut splice without any take gain."""
    plan = cut_stage.resolve_cut(root)
    parts = [cut_stage._audio_piece(piece, plan) for piece in plan.pieces]
    channels = max(part.shape[1] for part in parts)
    parts = [np.repeat(part, channels, axis=1) if part.shape[1] == 1 else part for part in parts]
    voice, _ = audio.splice(np.concatenate(parts),
                            [(piece.sample_start, piece.sample_end) for piece in plan.pieces],
                            cut_stage.FADE_SAMPLES)
    return np.clip(np.rint(voice * 32768), -32768, 32767).astype("<i2")


def test_cut_matches_take_speech_loudness(tmp_path: Path, speech_takes: tuple, no_asr) -> None:
    root = speech_project(tmp_path, speech_takes)
    result = cut_stage.cut(root)
    assert result["status"] == "verified", result
    report = json.loads((root / "cut/verify.json").read_bytes())
    assert not [row for row in report["findings"] if row["severity"] == "fail"]
    assert not [row for row in report["findings"] if row["check"] == "loudness.parts"]
    record = json.loads((root / "cut/receipt.json").read_bytes())["take_levels"]
    assert record["reference"] == "t1"
    assert list(record["takes"]) == ["t1", "t2"]
    assert record["takes"]["t1"]["gain_db"] == 0.0
    assert record["takes"]["t2"]["gain_db"] == pytest.approx(8, abs=0.6)
    assert all(take["limited"] is None for take in record["takes"].values())
    assert result["take_levels"] == record
    assert not [warning for warning in result["warnings"] if "loudness" in warning.lower()]
    assert cut_stage.cut(root)["take_levels"] == record


def test_pause_noise_does_not_set_take_gain(tmp_path: Path, speech_takes: tuple, no_asr) -> None:
    main, spans, _, _ = speech_takes
    _, original = audio.decode(main, sr=48_000)
    first_end = max(word.end for word in spans if word.sentence == 0)
    second_start = min(word.start for word in spans if word.sentence == 1)
    insert = round((first_end + second_start) / 2 * 48_000)
    sound = np.concatenate([original[:insert], np.zeros(5 * 48_000), original[insert:]])
    spans = [replace(word, start=word.start + 5, end=word.end + 5)
             if word.sentence == 1 else word for word in spans]
    pause_start, pause_end = round(first_end * 48_000), round((second_start + 5) * 48_000)
    noise = synth._pink_noise(pause_end - pause_start, np.random.default_rng(17))
    noise *= 0.95 / np.max(np.abs(noise))
    noisy = sound.copy()
    noisy[pause_start:pause_end] += noise
    clean_wav, noisy_wav = tmp_path / "clean.wav", tmp_path / "noisy.wav"
    synth._write_wav(clean_wav, sound, 48_000)
    synth._write_wav(noisy_wav, noisy, 48_000)
    clean, pickup = tmp_path / "clean.mov", tmp_path / "noisy.mov"
    ffmpeg.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        f"testsrc2=size=160x90:rate=30:duration={len(sound) / 48_000}",
        "-i", str(clean_wav), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "pcm_s16le", str(clean),
    ], timeout_s=60)
    ffmpeg.run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(clean), "-i", str(noisy_wav),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "pcm_s16le", str(pickup),
    ], timeout_s=60)
    rows = [(word.text, word.start, word.end) for word in spans]
    root = make_project(tmp_path / "project", [clean, pickup], word_rows=[rows, rows])
    write_edit(root, [time_segment(f"s{index}", f"t{index}", 0.3, spans[-1].end + 0.1)
                      for index in (1, 2)])
    result = cut_stage.cut(root)
    assert result["status"] == "verified", result
    record = result["take_levels"]
    assert abs(record["takes"]["t2"]["gain_db"]) < 1.0
    assert all(row["basis"] == "words" for row in record["takes"].values())
    assert record["rule"]["TAKE_LEVEL_RULE_VERSION"] == 1
    assert record["takes"]["t2"]["peak_dbfs"] == pytest.approx(
        20 * np.log10(np.max(np.abs(noisy))), abs=0.02,
    )


def test_no_retained_words_measures_kept_audio(tmp_path: Path, speech_takes: tuple, no_asr) -> None:
    source = speech_takes[0]
    root = make_project(tmp_path / "project", [source, source], word_rows=[[], []])
    write_edit(root, [time_segment(f"s{index}", f"t{index}", 0.3, 2.7) for index in (1, 2)])
    result = cut_stage.cut(root)
    assert result["status"] == "verified", result
    assert all(row["basis"] == "kept" and row["lufs"] is not None
               for row in result["take_levels"]["takes"].values())


def test_word_audio_clips_and_merges_windows() -> None:
    from yap_studio.stages import take_levels

    sound = np.arange(12, dtype=np.float32).reshape(-1, 1)
    start = Fraction(1)
    spans = [(start + Fraction(first, 48_000), start + Fraction(last, 48_000))
             for first, last in [(9, 30), (5, 7), (-5, 3), (2, 5)]]
    actual = take_levels.word_audio(sound, spans, start, start + Fraction(20, 48_000))
    np.testing.assert_array_equal(actual, np.concatenate([sound[:7], sound[9:]]))


@pytest.mark.parametrize("word_seconds", [0.2, 0.5])
def test_word_measurement_falls_back_to_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, word_seconds: float,
) -> None:
    from yap_studio.stages import take_levels

    kept = np.full((48_000, 1), 0.25, dtype=np.float32)
    speech = np.zeros((round(word_seconds * 48_000), 1), dtype=np.float32)
    calls = []

    def measure(sound, work, name):
        calls.append(sound)
        return (-20.0, -12.04) if sound is kept else (None, None)

    monkeypatch.setattr(take_levels, "measure", measure)
    row = take_levels.measure_take(kept, speech, tmp_path, "t1")
    assert row["basis"] == "kept"
    assert row["lufs"] == -20.0
    assert row["peak_dbfs"] == pytest.approx(-12.0412, abs=0.001)
    assert calls[-1] is kept
    assert len(calls) == (1 if word_seconds < 0.4 else 2)


def test_limit_warning_survives_unchanged_cut(
    tmp_path: Path, speech_takes: tuple, no_asr,
) -> None:
    from yap_studio.stages import take_levels

    main, spans, quiet, pickup_spans = speech_takes
    quieter = tmp_path / "quieter.mov"
    ffmpeg.run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(quiet),
        "-af", "volume=-12dB", "-c:v", "copy", "-c:a", "pcm_s16le", str(quieter),
    ], timeout_s=60)
    root = speech_project(tmp_path, (main, spans, quieter, pickup_spans))
    first = cut_stage.cut(root)
    assert first["status"] == "verified", first
    warnings = take_levels.warnings(first["take_levels"])
    assert warnings and all(warning in first["warnings"] for warning in warnings)
    receipt = json.loads((root / "cut/receipt.json").read_bytes())
    second = cut_stage.cut(root)
    assert second["noop"] is True
    assert second["warnings"] == [*receipt["plan"]["warnings"], *warnings]


def test_headroom_keeps_gain_limit_and_caps_pickup_peak() -> None:
    from yap_studio.stages import take_levels

    record = take_levels.plan({"t1": measured(-20, -3), "t2": measured(-32, 0)}, enabled=True)
    assert all(-12 <= row["gain_db"] <= 12 for row in record["takes"].values())
    pickup = record["takes"]["t2"]
    assert pickup["gain_db"] == 0.0
    assert record["headroom_db"] == -12
    assert all(row["limited"] is None for row in record["takes"].values())
    assert take_levels.warnings(record) == []


def test_peak_above_ceiling_blocks_raise_without_lowering_recording() -> None:
    from yap_studio.stages import take_levels

    record = take_levels.plan({"t1": measured(-20, -6), "t2": measured(-24, -0.5)}, enabled=True)
    assert record["takes"]["t1"]["gain_db"] == -4.0
    assert record["takes"]["t2"]["gain_db"] == 0.0
    assert all(row["limited"] is None for row in record["takes"].values())
    assert take_levels.warnings(record) == []


def test_peak_limit_warns_only_when_it_lowers_gain() -> None:
    from yap_studio.stages import take_levels

    record = take_levels.plan({"t1": measured(-20, -6), "t2": measured(-10, -1),
                               "t3": measured(-30, -5)}, enabled=True)
    assert record["headroom_db"] == -2.0
    assert [row["gain_db"] for row in record["takes"].values()] == [-2.0, -12.0, 4.0]
    assert [row["limited"] for row in record["takes"].values()] == [None, None, "peak"]
    assert record["takes"]["t3"]["peak_dbfs"] + record["takes"]["t3"]["gain_db"] == -1.0
    assert take_levels.warnings(record) == [
        "Take t3 stays 4.0 LU quieter than take t1, because raising it more would clip. "
        "Check that it is the right recording, or record it again.",
    ]
    report = jump_report()
    for finding in [*report["findings"], *report["structure"]["findings"]]:
        finding["evidence"]["right_source"] = "t3"
    explained = take_levels.explain_part_jumps(report, record)
    action = ("Take t3 could only be raised part of the way to match take t1 before it would clip. "
              "Check that it is the right recording, or record it again.")
    assert explained["findings"][0]["suggested_action"] == action
    assert explained["structure"]["findings"][0]["suggested_action"] == action


def test_rounded_gain_stays_under_peak_ceiling() -> None:
    from yap_studio.stages import take_levels

    record = take_levels.plan({"t1": measured(-20, -3), "t2": measured(-28, -5.06)}, enabled=True)
    pickup = record["takes"]["t2"]
    assert pickup["gain_db"] <= 4.0
    assert -5.06 + pickup["gain_db"] <= take_levels.PEAK_CEILING_DBFS + 1e-9


def test_disabled_keeps_samples_and_explains_jump(
    tmp_path: Path, speech_takes: tuple, no_asr,
) -> None:
    root = speech_project(tmp_path, speech_takes)
    project = json.loads((root / "project.json").read_bytes())
    project["audio"] = {"mode": "loudnorm", "integrated_lufs": -14,
                        "true_peak_dbtp": -1.7, "lra_lu": 11, "match_takes": False}
    write_json(root / "project.json", project)
    expected = plain_voice(root)
    result = cut_stage.cut(root)
    assert result["status"] == "verified"
    record = result["take_levels"]
    assert record["match_takes"] is False
    assert all(row["gain_db"] == 0.0 and row["lufs"] is not None
               for row in record["takes"].values())
    with wave.open(str(root / "cut/voice.wav")) as source:
        actual = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    np.testing.assert_array_equal(actual, expected.ravel())
    report = json.loads((root / "cut/verify.json").read_bytes())
    jumps = [row for row in report["findings"] if row["check"] == "loudness.parts"]
    assert len(jumps) == 2
    for jump in jumps:
        assert "match_takes" in jump["message"]
        assert "match_takes" in jump["suggested_action"]
    assert "match_takes" in report["summary"]["text"]
    project["audio"]["match_takes"] = True
    write_json(root / "project.json", project)
    state = next(row for row in status(root)["stages"] if row["name"] == "cut")
    assert state["state"] == "stale"
    assert state["reason"] == "changed input: audio:take_levels"


def test_receipt_keeps_reference_with_sorted_takes(
    tmp_path: Path, speech_takes: tuple, no_asr,
) -> None:
    root = speech_project(tmp_path, speech_takes)
    segments = json.loads((root / "edit.json").read_bytes())["segments"]
    write_edit(root, [segments[1], segments[0], segments[2]])
    result = cut_stage.cut(root)
    assert result["status"] == "verified"
    record = json.loads((root / "cut/receipt.json").read_bytes())["take_levels"]
    assert list(record["takes"]) == ["t1", "t2"]
    assert record["reference"] == "t2"
    assert record == result["take_levels"]
    assert cut_stage.cut(root)["take_levels"] == record


def test_single_take_skips_matching(tmp_path: Path, speech_takes: tuple, no_asr,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    from yap_studio.stages import take_levels

    def refuse(*args, **kwargs):
        raise AssertionError("Single takes must not be measured")

    monkeypatch.setattr(take_levels, "measure", refuse)
    main, spans, _, _ = speech_takes
    root = make_project(tmp_path / "project", [main], word_rows=[
        [(word.text, word.start, word.end) for word in spans],
    ])
    write_edit(root, [time_segment("s1", "t1", spans[0].start - 0.1, spans[10].end + 0.1)])
    result = cut_stage.cut(root)
    assert result["status"] == "verified"
    assert "take_levels" not in result
    assert "take_levels" not in json.loads((root / "cut/receipt.json").read_bytes())
    inputs = cut_inputs(ProjectContext(root))
    assert "audio:take_levels" not in inputs
    names = ["cut.mp4", "voice.wav", "spans.json", "cut-words.json"]
    before = {name: (root / "cut" / name).read_bytes() for name in names}
    project = json.loads((root / "project.json").read_bytes())
    project["audio"] = {"mode": "preserve", "match_takes": False}
    write_json(root / "project.json", project)
    assert cut_inputs(ProjectContext(root)) == inputs
    assert cut_stage.cut(root, force=True)["status"] == "verified"
    assert {name: (root / "cut" / name).read_bytes() for name in names} == before


def measured(lufs: float | None, peak: float | None = -20, seconds: float = 3.1234) -> dict:
    return {"seconds": seconds, "lufs": lufs, "peak_dbfs": peak}


def test_plan_reference_order_and_unmeasured() -> None:
    from yap_studio.stages import take_levels

    record = take_levels.plan({"t3": measured(None, None), "t2": measured(-20),
                               "t1": measured(-22)}, enabled=True)
    assert record["reference"] == "t2"
    assert list(record["takes"]) == ["t3", "t2", "t1"]
    assert record["takes"]["t1"]["gain_db"] == 2
    assert record["takes"]["t3"]["gain_db"] == 0
    assert record["takes"]["t3"]["needed_db"] is None
    silent = take_levels.plan({"t1": measured(None, None), "t2": measured(None, None)},
                              enabled=True)
    assert silent["reference"] is None
    assert all(row["gain_db"] == 0 and row["needed_db"] is None
               for row in silent["takes"].values())


def test_plan_limits_and_warnings() -> None:
    from yap_studio.stages import take_levels

    record = take_levels.plan({"t1": measured(-25), "t2": measured(-45, -30),
                               "t3": measured(-5), "t4": measured(-37)}, enabled=True)
    assert [row["gain_db"] for row in record["takes"].values()] == [0, 12, -12, 12]
    assert [row["limited"] for row in record["takes"].values()] == [None, "gain", "gain", None]
    assert take_levels.warnings(record) == [
        "Take t2 is 20.0 LU quieter than take t1; the cut changed it by the 12 dB limit. "
        "Check that it is the right recording, or record it again.",
        "Take t3 is 20.0 LU louder than take t1; the cut changed it by the 12 dB limit. "
        "Check that it is the right recording, or record it again.",
    ]


def test_plan_headroom_shifts_every_take() -> None:
    from yap_studio.stages import take_levels

    record = take_levels.plan({"t1": measured(-20, -3), "t2": measured(-28, -5),
                               "t3": measured(None, None)}, enabled=True)
    assert record["headroom_db"] == -4
    assert [row["gain_db"] for row in record["takes"].values()] == [-4, 4, -4]
    assert record["takes"]["t1"]["lufs"] - 4 == record["takes"]["t2"]["lufs"] + 4
    assert record["takes"]["t2"]["peak_dbfs"] + 4 == take_levels.PEAK_CEILING_DBFS
    rounded = take_levels.plan({"t1": measured(-20, -3), "t2": measured(-28, -5.1234)},
                               enabled=True)
    assert rounded["headroom_db"] == -3.88


def test_plan_disabled_and_rounding() -> None:
    from yap_studio.stages import take_levels

    measurements = {"t1": measured(-20.1234, -3.1234), "t2": measured(-28.1678, -5)}
    before = copy.deepcopy(measurements)
    disabled = take_levels.plan(measurements, enabled=False)
    assert disabled["headroom_db"] == 0
    assert all(row["gain_db"] == 0 and row["limited"] is None
               for row in disabled["takes"].values())
    assert disabled["takes"]["t1"]["lufs"] == -20.12
    assert disabled["takes"]["t1"]["peak_dbfs"] == -3.12
    assert disabled["takes"]["t1"]["seconds"] == 3.12
    assert disabled["takes"]["t2"]["needed_db"] == 8.04
    assert take_levels.warnings(disabled) == []
    rounded = take_levels.plan({"t1": measured(-20), "t2": measured(-21.26),
                                "t3": measured(-19.99)}, enabled=True)
    assert [row["gain_db"] for row in rounded["takes"].values()] == [0, 1.3, 0]
    assert "-0.0" not in json.dumps([row["gain_db"] for row in rounded["takes"].values()])
    assert measurements == before


def jump_report() -> dict:
    action = "Normalize the loudness of this part to match the rest."
    finding = {"check": "loudness.parts", "severity": "warn",
               "message": f"Cut 1 has a jump. {action}", "suggested_action": action,
               "evidence": {"left_source": "t1", "right_source": "t2",
                            "left_lufs": -19, "right_lufs": -25}}
    return {"findings": [finding], "structure": {"findings": [copy.deepcopy(finding)]}}


@pytest.mark.parametrize("branch", ["off", "limited", "unmeasured", "matched"])
def test_explain_part_jumps(branch: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from yap_studio import report as report_module
    from yap_studio.stages import take_levels

    measurements = {"t1": measured(-20), "t2": measured(-28)}
    if branch == "limited":
        measurements["t2"]["lufs"] = -40
    elif branch == "unmeasured":
        measurements["t2"]["lufs"] = None
    record = take_levels.plan(measurements, enabled=branch != "off")
    original = jump_report()
    original.update({"media": {}, "summary": {"text": "old", "warn": 1}})
    before = copy.deepcopy(original)
    calls = []

    def summarize(report):
        calls.append(report)
        return report["findings"][0]["message"]

    monkeypatch.setattr(report_module, "summarize", summarize)
    result = take_levels.explain_part_jumps(original, record)
    expected = {
        "off": 'The takes keep their recorded levels because "match_takes" is false ',
        "limited": "Take t2 needed more than the 12 dB limit to match take t1.",
        "unmeasured": "Take t2 has too little speech to measure, so it keeps its recorded level.",
        "matched": "The takes are already matched, so the jump is inside take t2: "
                   "the part after the cut is 5.0 LU quieter than the rest of that take.",
    }[branch]
    for row in [result["findings"][0], result["structure"]["findings"][0]]:
        assert row["suggested_action"].startswith(expected)
        assert row["message"] == "Cut 1 has a jump. " + row["suggested_action"]
        assert row["severity"] == "warn"
    assert result["summary"]["text"] == result["findings"][0]["message"]
    assert result["summary"]["warn"] == 1
    assert len(calls) == 1
    assert original == before


def test_explain_left_part_and_unchanged_reports() -> None:
    from yap_studio.stages import take_levels

    record = take_levels.plan({"t1": measured(-20), "t2": measured(-22)}, enabled=True)
    original = jump_report()
    original["findings"][0]["evidence"]["left_lufs"] = -10
    result = take_levels.explain_part_jumps(original, record)
    assert "take t1: the part before the cut is 10.0 LU louder" in result["findings"][0]["message"]
    assert take_levels.explain_part_jumps(original, None) is original
    unrelated = {"findings": [{"check": "picture.black"}]}
    assert take_levels.explain_part_jumps(unrelated, record) == unrelated


@pytest.mark.parametrize(("side", "source"), [
    ("left", None), ("right", None), ("left", "t3"), ("right", "t3"),
])
def test_explain_leaves_unknown_sources_unchanged(side: str, source: str | None) -> None:
    from yap_studio.stages import take_levels

    record = take_levels.plan({"t1": measured(-20), "t2": measured(-22)}, enabled=True)
    original = jump_report()
    for finding in [*original["findings"], *original["structure"]["findings"]]:
        if source is None:
            finding["evidence"].pop(f"{side}_source")
        else:
            finding["evidence"][f"{side}_source"] = source
    before = copy.deepcopy(original)
    assert take_levels.explain_part_jumps(original, record) == before
    assert original == before


@pytest.mark.parametrize("seconds", [0.1, 1.0])
def test_measure_silence_has_no_level(tmp_path: Path, seconds: float) -> None:
    from yap_studio.stages import take_levels

    assert take_levels.measure(np.zeros((round(seconds * 48_000), 2), dtype=np.float32),
                               tmp_path, "silent") == (None, None)


def test_measure_float_channels_and_peak(tmp_path: Path) -> None:
    from yap_studio.stages import take_levels

    sound = np.zeros((48_000, 2), dtype=np.float32)
    sound[:, 1] = 0.25 * np.sin(np.arange(48_000) * 2 * np.pi * 440 / 48_000)
    lufs, peak = take_levels.measure(sound, tmp_path, "stereo")
    assert lufs is not None
    assert peak == pytest.approx(-12.0412, abs=0.001)
    wav = next(tmp_path.glob("*.wav"))
    info = ffmpeg.probe(wav)
    assert (info.audio_codec, info.sample_rate, info.channels) == ("pcm_f32le", 48_000, 2)
