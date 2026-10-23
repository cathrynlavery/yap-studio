"""Short join readings catch a restart that a whole-render reading smooths away."""

from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import synth

from yap_studio import audio
from yap_studio.checks.joins import check_joins
from yap_studio.checks.relisten import relisten_at, relisten_findings
from yap_studio.checks.sync import check_sync
from yap_studio.edl import load_edit_list
from yap_studio.plan import Plan
from yap_studio.transcript import (
    ENGINES,
    IsolatedBackend,
    ScriptedBackend,
    Transcript,
    Word,
    compare_words,
    resolve_second_backend,
    second_engine_unmeasured_reason,
)

PITCHES = {"one": 310, "two": 430, "three": 570, "four": 730, "five": 910}
PLANNED = ["one", "two", "three", "four", "five"]
NEW_CHECKS = ("join.relisten", "join.second_model")


class ToneBackend(ScriptedBackend):
    """Hear words from WAV bursts, optionally smoothing repeats or inserting a word."""

    def __init__(
        self,
        *,
        name: str = "tones-first",
        smooth: bool = False,
        invent: bool = False,
    ) -> None:
        super().__init__(self.hear, name=name)
        self.smooth = smooth
        self.invent = invent
        self.batches: list[list[Path]] = []

    def hear(self, path: Path) -> Transcript:
        sr, samples = (
            synth.read_wav(path) if path.suffix == ".wav" else audio.decode(path, sr=16_000)
        )
        # A 5 ms RMS gate separates bursts from the noise bed and ignores codec ringing.
        hop = round(0.005 * sr)
        count = len(samples) // hop
        blocks = samples[: count * hop].reshape(count, hop)
        active = np.sqrt(np.mean(blocks.astype(np.float64) ** 2, axis=1)) > 0.025
        changes = np.diff(np.r_[False, active, False].astype(int))
        starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        words = []
        for first, last in zip(starts, ends, strict=True):
            if (last - first) * hop / sr < 0.04:
                continue
            burst = samples[first * hop : last * hop]
            spectrum = np.abs(np.fft.rfft(burst * np.hanning(len(burst)), n=8192))
            pitch = np.argmax(spectrum) * sr / 8192
            word = min(PITCHES, key=lambda text: abs(PITCHES[text] - pitch))
            if self.smooth and words and words[-1].text == word:
                continue
            words.append(Word(word, first * hop / sr, last * hop / sr))
        if self.invent and path.stem.endswith("render"):
            # Put a new word in the middle, where window-edge tolerance cannot hide it.
            at = words[2].start - 0.04
            words.insert(2, Word("extra", at, at + 0.03))
        return Transcript(words, backend=self.name, model=self.model, duration_s=len(samples) / sr)

    def transcribe_many(
        self,
        paths: Sequence[Path],
        *,
        language: str | None = None,
    ) -> dict[Path, Transcript]:
        self.batches.append(list(paths))
        return super().transcribe_many(paths, language=language)


@pytest.fixture(scope="module")
def tone_media(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    directory = tmp_path_factory.mktemp("relisten")
    source, words = synth.tone_speech_source(
        directory / "source.mov",
        ["one", "two", "three", "three", "four", "five"],
        PITCHES,
    )
    cases = {}
    # The first restart starts at frame 42; frame 44 keeps its first 67 ms.
    # Frames 40 and 53 sit in the pauses before each restart.
    for name, spans in {
        "clean": [(0, 40 / 30), (53 / 30, 3.5)],
        "doubled": [(0, 44 / 30), (53 / 30, 3.5)],
        "short": [(0, 40 / 30), (53 / 30, 76 / 30), (77 / 30, 3.5)],
    }.items():
        edl = directory / f"{name}.json"
        edl.write_text(
            json.dumps(
                {
                    "source": str(source),
                    "segments": [{"start": start, "end": end} for start, end in spans],
                }
            )
        )
        plan = Plan.build(load_edit_list(edl), Fraction(30), 48_000)
        render = synth.render_clean(source, spans, directory / f"{name}.mp4", fade_s=0.005)
        sync = check_sync(render, plan, {"main": source})
        cases[name] = SimpleNamespace(edl=edl, plan=plan, render=render, sync=sync)
    return SimpleNamespace(source=source, words=words, cases=cases)


def _joins(media, case: str, directory: Path, first, second=None):
    row = media.cases[case]
    return check_joins(
        row.render,
        row.plan,
        {"main": media.source},
        row.sync,
        backend=first,
        second_backend=second,
        work_dir=directory,
    )


def test_doubled_syllable_warns_despite_whole_render_and_audio_passing(tone_media, tmp_path):
    whole = ToneBackend(smooth=True).hear(tone_media.cases["doubled"].render)
    assert [word.text for word in whole.words] == PLANNED
    assert compare_words(PLANNED, [word.text for word in whole.words])["core_exact"]
    first, second = ToneBackend(), ToneBackend(name="tones-second")
    result = _joins(tone_media, "doubled", tmp_path, first, second)
    assert result.cuts[0].words.status == "pass"
    assert not [f for f in result.findings if f.check == "join.words"]
    (warning,) = [f for f in result.findings if f.check == "join.relisten"]
    assert warning.severity == "warn" and warning.cut == 1
    for engine, reading in ((first, "first"), (second, "second")):
        evidence = warning.evidence[reading]
        assert evidence["engine"] == engine.name
        assert evidence["planned"] == PLANNED
        assert evidence["heard"] == ["one", "two", "three", "three", "four", "five"]
        assert evidence["diff"] and not evidence["agreement"]
    assert '"one two three three four five"' in warning.message
    assert '"one two three four five"' in warning.message
    assert result.cuts[0].relisten.first.plan_agreement
    assert result.cuts[0].relisten.doubled_words[0]["word"] == "three"
    assert "source does not repeat" in result.cuts[0].relisten.reason
    heard_words = result.cuts[0].relisten.first.heard_words
    assert [word.text for word in heard_words] == warning.evidence["first"]["heard"]
    assert 0.04 <= heard_words[2].end - heard_words[2].start <= 0.08
    assert result.cuts[0].relisten.second_status == "pass"
    assert len(first.batches) == len(second.batches) == 1
    assert first.batches == second.batches
    assert len(first.batches[0]) == 4


def test_short_inserted_repeat_warns_from_render_vs_plan_audio(tone_media, tmp_path):
    row = tone_media.cases["clean"]
    samples = synth._calibration_audio(row.render)
    join = float(row.plan.cuts[0].out_time)
    at = round(join * synth.RATE)
    duration = 0.05
    time = np.arange(round(duration * synth.RATE)) / synth.RATE
    phase = 2 * np.pi * PITCHES["two"] * time
    burst = 0.18 * np.sin(phase) + 0.06 * np.sin(2 * phase) + 0.025 * np.sin(3 * phase)
    fade = round(0.005 * synth.RATE)
    burst[:fade] *= np.linspace(0, 1, fade)
    burst[-fade:] *= np.linspace(1, 0, fade)
    wav = tmp_path / "inserted.wav"
    synth._write_wav(wav, np.concatenate((samples[:at], burst, samples[at:])), synth.RATE)
    render = tmp_path / "inserted.mp4"
    synth._run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(row.render),
            "-i",
            str(wav),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(render),
        ]
    )
    measured = check_sync(render, row.plan, {"main": tone_media.source})
    result = check_joins(
        render,
        row.plan,
        {"main": tone_media.source},
        measured,
        backend=ToneBackend(),
        work_dir=tmp_path,
    )
    cut = result.cuts[0]
    assert cut.j_start_s - cut.j_end_s == pytest.approx(duration, abs=0.005)
    assert 0.001 < cut.j_start_s - cut.j_end_s < 0.06
    assert cut.words.status == "pass"
    assert not [f for f in result.findings if f.check == "join.words"]
    (warning,) = [f for f in result.findings if f.check == "join.relisten"]
    assert warning.severity == "warn" and warning.cut == 1
    assert "the edit's own audio says" in warning.message
    assert warning.evidence["first"]["plan_heard"] == PLANNED
    assert warning.evidence["first"]["heard"] == ["one", "two", "two", "three", "four", "five"]
    assert not warning.evidence["first"]["plan_agreement"]
    assert warning.evidence["first"]["plan_diff"]


@pytest.mark.parametrize("invent", [False, True])
def test_clean_join_with_two_engines(tone_media, tmp_path, invent):
    first, second = ToneBackend(), ToneBackend(name="tones-second", invent=invent)
    result = _joins(tone_media, "clean", tmp_path, first, second)
    assert not [f for f in result.findings if f.check == "join.relisten"]
    row = result.cuts[0].relisten
    assert row.status == "pass" and row.first.planned == row.first.heard == PLANNED
    assert row.second_status == ("warn" if invent else "pass")
    warnings = [f for f in result.findings if f.check == "join.second_model"]
    if invent:
        (warning,) = warnings
        assert warning.cut == 1 and warning.severity == "warn"
        assert warning.evidence["first"]["heard"] == PLANNED
        assert warning.evidence["second"]["heard"] == [
            "one",
            "two",
            "extra",
            "three",
            "four",
            "five",
        ]
        assert first.name in warning.message and second.name in warning.message
        assert warning.evidence["diff"]
    else:
        assert not warnings
    assert len(first.batches) == len(second.batches) == 1
    assert first.batches == second.batches


def test_short_neighbours_and_both_models_share_one_batch_and_media_decode(
    tone_media,
    tmp_path,
    monkeypatch,
):
    from yap_studio import video

    decoded = {"audio": [], "picture": []}

    def record_decode(fn, kind):
        def wrapped(path, *args, **kwargs):
            decoded[kind].append(path)
            return fn(path, *args, **kwargs)

        return wrapped

    monkeypatch.setattr(audio, "decode", record_decode(audio.decode, "audio"))
    monkeypatch.setattr(video, "grey_frames", record_decode(video.grey_frames, "picture"))
    first, second = ToneBackend(), ToneBackend(name="second")
    result = _joins(tone_media, "short", tmp_path, first, second)
    assert len(result.cuts) == 2
    assert not [f for f in result.findings if f.check in NEW_CHECKS]
    assert all(row.relisten.status == row.relisten.second_status == "pass" for row in result.cuts)
    assert result.cuts[0].relisten.first.heard == PLANNED[:-1]
    assert result.cuts[1].relisten.first.heard == PLANNED[2:]
    assert len(first.batches) == len(second.batches) == 1
    assert len(first.batches[0]) == 8 and first.batches == second.batches
    for paths in decoded.values():
        assert paths == [tone_media.cases["short"].render, tone_media.source]


@pytest.mark.parametrize(
    "engines,installed",
    [(0, ()), (1, ("parakeet-mlx",)), (1, ("parakeet-mlx", "parakeet-onnx")), (2, ENGINES)],
)
def test_verify_engine_coverage_and_receipt(tone_media, monkeypatch, engines, installed):
    module = importlib.import_module("yap_studio.verify")
    backends = importlib.import_module("yap_studio.transcript.backends")
    monkeypatch.setattr(
        backends, "available_engines", lambda: {name: name in installed for name in ENGINES}
    )
    first = ToneBackend(name="parakeet-mlx") if engines else None
    second = ToneBackend(name="faster-whisper") if engines == 2 else None
    monkeypatch.setattr(module, "resolve_backend", lambda *_: first)
    monkeypatch.setattr(module, "resolve_second_backend", lambda backend: second)
    clean = tone_media.cases["clean"]
    result = module.verify(str(clean.render), edl=str(clean.edl), detail="full")
    checks = result["checks"]
    if engines == 0:
        for check in NEW_CHECKS:
            assert checks[check] == {
                "ran": False,
                "status": "skipped",
                "reason": module._NO_ENGINE_REASON,
            }
        assert result["receipt"]["speech"] is None
        lines = result["summary"]["text"].splitlines()
        assert len([line for line in lines if "Re-listening" in line]) == 1
    else:
        assert checks["join.relisten"] == {"ran": True, "status": "pass", "reason": None}
        assert result["receipt"]["speech"]["engine"] == first.name
        if engines == 1:
            assert checks["join.second_model"]["ran"] is False
            assert checks["join.second_model"]["status"] == "skipped"
            reason = checks["join.second_model"]["reason"]
            assert reason.startswith("unmeasured:") and first.name in reason
            assert "pip install 'yap-studio[whisper]'" in reason
            assert reason == second_engine_unmeasured_reason(first.name)
            if "parakeet-onnx" in installed:
                assert "parakeet-onnx is the same model family as parakeet-mlx" in reason
            else:
                assert "only parakeet-mlx is installed" in reason
        else:
            assert checks["join.second_model"] == {"ran": True, "status": "pass", "reason": None}
            assert result["receipt"]["speech_second"] == {
                "engine": second.name,
                "model": second.model,
                "version": "0",
            }
    if engines < 2:
        assert result["receipt"]["speech_second"] is None


@pytest.mark.parametrize(
    "first_name,installed,expected",
    [
        ("parakeet-mlx", ENGINES, "faster-whisper"),
        ("parakeet-onnx", ENGINES, "faster-whisper"),
        ("faster-whisper", ENGINES, "parakeet-mlx"),
        ("faster-whisper", ("parakeet-onnx", "faster-whisper"), "parakeet-onnx"),
        ("parakeet-mlx", ("parakeet-mlx", "faster-whisper"), "faster-whisper"),
        ("parakeet-mlx", ("parakeet-mlx", "parakeet-onnx"), None),
        ("parakeet-onnx", ("parakeet-mlx",), None),
        ("parakeet-mlx", ("parakeet-onnx",), None),
        ("parakeet-mlx", ("parakeet-mlx",), None),
        ("parakeet-mlx", (), None),
        ("scripted", ENGINES, None),
        (None, ENGINES, None),
    ],
)
def test_second_backend_resolver(monkeypatch, first_name, installed, expected):
    module = importlib.import_module("yap_studio.transcript.backends")
    monkeypatch.setattr(
        module, "available_engines", lambda: {name: name in installed for name in ENGINES}
    )
    first = (
        ScriptedBackend(lambda _: Transcript([]), name=first_name, model="custom")
        if first_name is not None
        else None
    )
    second = resolve_second_backend(first)
    assert (second.name if second else None) == expected
    if second is not None:
        assert isinstance(second, IsolatedBackend)
        assert second.name == second.inner.name and second.model == second.inner.model
        assert (
            second.model
            == {
                "parakeet-mlx": "mlx-community/parakeet-tdt-0.6b-v3",
                "parakeet-onnx": "nemo-parakeet-tdt-0.6b-v3",
                "faster-whisper": "small.en",
            }[expected]
        )


@pytest.mark.parametrize(
    "first_name,installed,expected",
    [
        (
            "parakeet-mlx", ("parakeet-mlx", "parakeet-onnx"),
            "unmeasured: no speech engine from another model family is installed "
            "(parakeet-onnx is the same model family as parakeet-mlx); install one, "
            "for example pip install 'yap-studio[whisper]'",
        ),
        (
            "parakeet-onnx", ("parakeet-mlx", "parakeet-onnx"),
            "unmeasured: no speech engine from another model family is installed "
            "(parakeet-mlx is the same model family as parakeet-onnx); install one, "
            "for example pip install 'yap-studio[whisper]'",
        ),
        (
            "parakeet-mlx", ("parakeet-mlx",),
            "unmeasured: only parakeet-mlx is installed; install a speech engine "
            "from another model family, for example pip install 'yap-studio[whisper]'",
        ),
        (
            "faster-whisper", ("faster-whisper",),
            "unmeasured: only faster-whisper is installed; install a speech engine "
            "from another model family, for example pip install 'yap-studio[asr]'",
        ),
    ],
)
def test_second_engine_unmeasured_reason(monkeypatch, first_name, installed, expected):
    module = importlib.import_module("yap_studio.transcript.backends")
    monkeypatch.setattr(
        module, "available_engines", lambda: {name: name in installed for name in ENGINES}
    )
    assert second_engine_unmeasured_reason(first_name) == expected


def _transcript(words: list[tuple[str, float, float]]) -> Transcript:
    return Transcript([Word(text, start + 1, end + 1) for text, start, end in words])


def _phrase(text: str, start: float, step: float = 0.2) -> list[tuple[str, float, float]]:
    return [(word, start + i * step, start + i * step + 0.1) for i, word in enumerate(text.split())]


def _at_cut(
    render,
    plan,
    left=(),
    right=(),
    *,
    second=None,
    words_failed=False,
    left_length_s=3,
    right_length_s=3,
    j_end_s=1.5,
    j_start_s=1.5,
    window_start_s=0,
):
    transcripts = [_transcript(words) for words in (render, plan, left, right)]
    return relisten_at(
        transcripts,
        engine="first",
        left_length_s=left_length_s,
        right_length_s=right_length_s,
        j_end_s=j_end_s,
        j_start_s=j_start_s,
        window_start_s=window_start_s,
        second_engine="second" if second is not None else None,
        second_transcripts=(
            [_transcript(second), *transcripts[1:]] if second is not None else None
        ),
        words_failed=words_failed,
    )


def test_second_model_one_word_disagreement_at_cut_warns():
    render = [("yes", 1.4, 1.6)]
    result = _at_cut(render, render, second=[("no", 1.4, 1.6)])
    assert result.second_status == "warn"
    assert result.status == "pass"
    assert result.second_comparison["first"] == ["yes"]
    assert result.second_comparison["second"] == ["no"]
    assert result.second_comparison["diff"] and not result.second_comparison["agreement"]
    (warning,) = relisten_findings(result, 1, 1.5)
    assert warning.check == "join.second_model" and warning.severity == "warn"
    assert warning.evidence["diff"] == result.second_comparison["diff"]


@pytest.mark.parametrize(
    "start,end,j_end_s,j_start_s,window_start_s",
    [(0.9, 2.1, 1.5, 1.5, 0), (0.9, 1.6, 1.5, 1.5, 0), (0.9, 1.7, 11.5, 11.8, 10)],
    ids=["both_boundaries", "start_boundary", "inserted_join_offset_clock"],
)
def test_second_model_outer_boundary_word_touching_join_warns(
    start, end, j_end_s, j_start_s, window_start_s,
):
    render = [("yes", start, end)]
    result = _at_cut(
        render,
        render,
        second=[("no", start, end)],
        j_end_s=j_end_s,
        j_start_s=j_start_s,
        window_start_s=window_start_s,
    )
    assert result.second_status == "warn"
    assert result.status == "pass"
    (warning,) = relisten_findings(result, 1, j_end_s)
    assert warning.check == "join.second_model" and warning.severity == "warn"
    assert warning.evidence["diff"] == ["replace exp['yes'] got['no']"]
    assert warning.evidence["second_comparison"]["agreement"] is False


@pytest.mark.parametrize(
    "first,second",
    [
        (
            [("non", -1.16, -0.04), ("techies", -1.16, -0.04),
             ("do", 0.0, 0.1), ("you", 0.12, 0.22), ("remember", 0.24, 0.36)],
            [("techies", -1.16, -0.04), ("do", 0.0, 0.1), ("you", 0.12, 0.22),
             ("remember", 0.24, 0.36), ("when", 0.38, 0.86)],
        ),
        (
            [("skill", -0.64, -0.24), ("is", -0.2, -0.1), ("a", 0.0, 0.05),
             ("skill", 0.1, 0.25)],
            [("is", -0.2, -0.1), ("a", 0.0, 0.05), ("skill", 0.1, 0.25)],
        ),
        (
            [("claude", -0.52, -0.12), ("video", 0.0, 0.2)],
            [("quad", -0.54, -0.32), ("video", 0.0, 0.2), ("if", 0.3, 0.6)],
        ),
        (
            [("a", -0.53, -0.45), ("skill", -0.2, 0.1), ("make", 0.27, 0.75)],
            [("skill", -0.2, 0.1), ("make", 0.27, 0.75), ("a", 0.43, 1.45)],
        ),
        (
            [("for", -0.59, -0.27), ("you", -0.27, 0.21), ("you", 0.21, 0.37),
             ("don't", 0.37, 0.69)],
            [("you", -0.57, -0.23), ("you", 0.37, 0.55)],
        ),
        (
            [("to", -0.51, -0.43), ("code", -0.3, -0.2), ("the", -0.1, 0.0),
             ("easiest", 0.1, 0.2), ("way", 0.3, 0.4)],
            [("code", -0.3, -0.2), ("the", -0.1, 0.0), ("easiest", 0.1, 0.2),
             ("way", 0.3, 0.4), ("to", 0.47, 0.59)],
        ),
    ],
    ids=["non_techies", "left_overlap", "both_outer_edges", "opposite_outer_edges",
         "planned_you_you", "to_code"],
)
def test_second_model_outer_boundary_disagreements_are_evidence_only(first, second):
    render = [(text, 1.5 + start, 1.5 + end) for text, start, end in first]
    other = [(text, 1.5 + start, 1.5 + end) for text, start, end in second]
    # Controls retain the first reading's planned repetition at the join.
    right = [(text, start - 0.5, end - 0.5) for text, start, end in render]
    result = _at_cut(render, render, render, right, second=other)
    assert result.second_status == "pass"
    assert result.second_comparison["first"] == [word[0] for word in first]
    assert result.second_comparison["second"] == [word[0] for word in second]
    assert result.second_comparison["diff"] and result.second_comparison["agreement"]
    assert not relisten_findings(result, 1, 1.5)


def test_render_vs_plan_one_word_disagreement_at_cut_warns():
    result = _at_cut([("yes", 1.4, 1.6)], [("no", 1.4, 1.6)])
    assert result.status == "warn"
    assert result.first.plan_diff and not result.first.plan_agreement
    (warning,) = relisten_findings(result, 1, 1.5)
    assert warning.check == "join.relisten" and warning.severity == "warn"
    assert warning.evidence["first"]["plan_diff"] == result.first.plan_diff
    assert warning.evidence["first"]["plan_agreement"] is False


@pytest.mark.parametrize(
    "start,end,j_end_s,j_start_s,window_start_s",
    [(-0.1, 3.1, 1.5, 1.5, 0), (-0.1, 1.6, 1.5, 1.5, 0), (-0.1, 1.7, 11.5, 11.8, 10)],
    ids=["both_boundaries", "start_boundary", "inserted_join_offset_clock"],
)
def test_render_vs_plan_outer_boundary_word_touching_join_warns(
    start, end, j_end_s, j_start_s, window_start_s,
):
    result = _at_cut(
        [("yes", start, end)],
        [("no", start, end)],
        j_end_s=j_end_s,
        j_start_s=j_start_s,
        window_start_s=window_start_s,
    )
    assert result.status == "warn"
    (warning,) = relisten_findings(result, 1, j_end_s)
    assert warning.check == "join.relisten" and warning.severity == "warn"
    assert warning.evidence["first"]["plan_diff"] == ["replace exp['no'] got['yes']"]
    assert warning.evidence["first"]["plan_agreement"] is False


@pytest.mark.parametrize("start,end", [(0.85, 1.1), (1.9, 2.2)], ids=["left", "right"])
def test_render_vs_plan_outer_boundary_disagreement_is_evidence_only(start, end):
    result = _at_cut(
        [("yes", start, end)],
        [("no", start, end)],
        left_length_s=0.5,
        right_length_s=0.5,
    )
    assert result.status == "pass"
    assert result.first.heard == ["yes"] and result.first.plan_heard == ["no"]
    assert result.first.plan_diff and result.first.plan_agreement
    assert not relisten_findings(result, 1, 1.5)


@pytest.mark.parametrize("check", NEW_CHECKS)
def test_merged_outer_boundary_replacement_is_evidence_only(check):
    render = [("yes", 0.85, 1.1), ("yes", 1.9, 2.2)]
    other = [("no", 0.85, 1.1), ("no", 1.9, 2.2)]
    result = _at_cut(
        render,
        other if check == "join.relisten" else render,
        second=other if check == "join.second_model" else None,
        left_length_s=0.5,
        right_length_s=0.5,
    )
    if check == "join.relisten":
        assert result.status == "pass"
        assert result.first.plan_diff == ["replace exp['no', 'no'] got['yes', 'yes']"]
        assert result.first.plan_agreement
    else:
        assert result.second_status == "pass"
        assert result.second_comparison["diff"] == ["replace exp['yes', 'yes'] got['no', 'no']"]
        assert result.second_comparison["agreement"]
    assert not relisten_findings(result, 1, 1.5)


@pytest.mark.parametrize("start,end", [(1.02, 1.08), (1.92, 1.98)], ids=["left", "right"])
def test_fully_contained_words_near_outer_edges_still_disagree(start, end):
    render = [("yes", start, end)]
    other = [("no", start, end)]
    result = _at_cut(
        render,
        other,
        second=other,
        left_length_s=0.5,
        right_length_s=0.5,
    )
    assert result.status == result.second_status == "warn"
    assert not result.first.plan_agreement and not result.second_comparison["agreement"]
    assert {finding.check for finding in relisten_findings(result, 1, 1.5)} == set(NEW_CHECKS)


def test_fully_contained_a_vs_our_still_disagrees():
    render = [("a", 1.91, 1.99)]
    other = [("our", 1.87, 1.95)]
    result = _at_cut(render, other, second=other, left_length_s=0.5, right_length_s=0.5)
    assert result.status == result.second_status == "warn"
    assert not result.first.plan_agreement and not result.second_comparison["agreement"]
    assert {finding.check for finding in relisten_findings(result, 1, 1.5)} == set(NEW_CHECKS)


def test_span_comparisons_normalise_words_and_drop_optional_fillers():
    render = [("Gonna", 1.2, 1.4), ("um", 1.4, 1.5), ("GO.", 1.5, 1.7)]
    other = [("going to", 1.2, 1.4), ("go", 1.5, 1.7)]
    result = _at_cut(render, other, second=other)
    assert result.status == result.second_status == "pass"
    assert result.first.plan_agreement and result.second_comparison["agreement"]
    assert not result.first.plan_diff and not result.second_comparison["diff"]
    assert not relisten_findings(result, 1, 1.5)


def test_distant_repetition_does_not_warn_at_cut():
    render = [("go", 0.4, 0.5), ("go", 2.5, 2.6)]
    result = _at_cut(
        render,
        render,
        [("go", 0.4, 0.5)],
        [("home", 1.0, 1.1)],
    )
    assert result.status == "pass"
    assert result.first.plan_agreement and not result.doubled_words
    assert not relisten_findings(result, 1, 1.5)


def test_earlier_planned_repetition_does_not_exempt_stutter_at_cut():
    left = [("go", 0.15, 0.25), ("go", 0.35, 0.45), ("go", 1.3, 1.5)]
    right = [("home", 1.0, 1.2)]
    render = [*left, ("go", 1.5, 1.65), ("home", 1.65, 1.85)]
    result = _at_cut(render, render, left, right)
    assert result.status == "warn"
    assert result.first.plan_agreement
    (pair,) = result.doubled_words
    assert pair["word"] == "go" and pair["planned_pair"] is None
    assert pair["first"]["start"] == pytest.approx(1.3)
    assert pair["second"]["start"] == pytest.approx(1.5)
    (warning,) = relisten_findings(result, 1, 1.5)
    assert warning.check == "join.relisten" and warning.evidence["doubled_words"] == [pair]


def test_planned_repetition_at_cut_is_exempt_and_records_the_matched_pair():
    left = [("for", 1.15, 1.25), ("YOU", 1.35, 1.5)]
    right = [("you.", 1.0, 1.15), ("don't", 1.2, 1.4)]
    render = [*left, ("you", 1.5, 1.65), ("don't", 1.7, 1.9)]
    result = _at_cut(render, render, left, right)
    assert result.status == "pass"
    (pair,) = result.doubled_words
    assert pair["word"] == "you"
    assert pair["planned_pair"]["first"] == {
        "text": "you", "start": pytest.approx(1.35), "end": pytest.approx(1.5), "conf": None,
    }
    assert pair["planned_pair"]["second"] == {
        "text": "you", "start": pytest.approx(1.5), "end": pytest.approx(1.65), "conf": None,
    }
    assert not relisten_findings(result, 1, 1.5)


@pytest.mark.parametrize("planned_repeat", [False, True])
def test_doubled_pair_uses_each_landing_and_maps_controls_to_render_time(planned_repeat):
    left = [("go", 1.3, 1.5)]
    right = [("go" if planned_repeat else "home", 1.0, 1.2)]
    render = [("go", 1.3, 1.5), ("go", 2.5, 2.7)]
    result = _at_cut(
        render,
        render,
        left,
        right,
        j_end_s=11.5,
        j_start_s=12.5,
        window_start_s=10,
    )
    assert result.status == ("pass" if planned_repeat else "warn")
    (pair,) = result.doubled_words
    assert pair["first"]["end"] == pytest.approx(11.5)
    assert pair["second"]["start"] == pytest.approx(12.5)
    assert (pair["planned_pair"] is not None) == planned_repeat
    assert [word.start for word in result.first.planned_words] == pytest.approx([11.3, 12.5])
    assert [word.end for word in result.first.planned_words] == pytest.approx([11.5, 12.7])
    assert bool(relisten_findings(result, 1, 11.5)) == (not planned_repeat)


@pytest.mark.parametrize(
    "heard,left,right,planned,start",
    [
        (
            "for non techies do you remember",
            "for non tuckies if",
            "you remember",
            "for non tuckies if you remember",
            0.7,
        ),
        (
            "of my dog skill a skill can range",
            "of my dog skill",
            "can range",
            "of my dog skill can range",
            0.4,
        ),
        ("for you you don't", "for you", "you don't", "for you you don't", 1.0),
    ],
)
def test_clean_readings_ignore_control_context_and_coarse_timestamps(
    heard,
    left,
    right,
    planned,
    start,
):
    render = _phrase(heard, start)
    result = _at_cut(
        render,
        render,
        _phrase(left, start),
        _phrase(right, 1.0),
    )
    assert result.first.planned == planned.split()
    assert result.first.heard == result.first.plan_heard == heard.split()
    assert result.first.plan_agreement
    assert result.status == "pass"
    assert all(pair["planned_pair"] is not None for pair in result.doubled_words)
    assert not relisten_findings(result, 1, 1.5)
    if planned != heard:
        assert result.first.diff and not result.first.agreement


@pytest.mark.parametrize("words_failed", [False, True])
def test_render_vs_plan_warning_is_suppressed_only_when_join_words_failed(words_failed):
    plan = _phrase("alpha beta gamma delta omega", 0.7)
    render = [("changed" if text == "gamma" else text, start, end) for text, start, end in plan]
    result = _at_cut(render, plan, words_failed=words_failed)
    assert result.first.plan_diff and not result.first.plan_agreement
    assert result.status == ("pass" if words_failed else "warn")
    findings = relisten_findings(result, 1, 1.5)
    if words_failed:
        assert not findings
    else:
        (warning,) = findings
        assert warning.check == "join.relisten" and warning.severity == "warn"
        assert "the edit's own audio says" in warning.message


@pytest.mark.parametrize(
    "first_end,second_start,warns",
    [
        (1.45, 1.55, True),
        (0.95, 1.05, False),
        (2.05, 2.1, False),
        (1.8, 1.8, True),
        (1.2, 1.2, True),
    ],
)
def test_doubled_word_must_touch_the_cut_and_still_warns_after_an_audio_failure(
    first_end,
    second_start,
    warns,
):
    render = [
        ("alpha", 0.3, 0.4),
        ("GO", first_end - 0.1, first_end),
        ("go.", second_start, second_start + 0.1),
        ("omega", 2.5, 2.6),
    ]
    result = _at_cut(
        render,
        render,
        [("alpha", 0.3, 0.4), ("go", 1.1, 1.2)],
        [("omega", 1.0, 1.1)],
        words_failed=True,
    )
    assert result.first.plan_agreement
    assert bool(result.doubled_words) == warns
    assert result.status == ("warn" if warns else "pass")
    assert bool(relisten_findings(result, 1, 1.5)) == warns


@pytest.mark.parametrize("where,warns", [("away", False), ("cut", True), ("empty", True)])
def test_second_engine_compares_only_half_a_second_at_the_cut(where, warns):
    render = [
        ("before", 0.4, 0.5),
        ("this", 1.02, 1.1),
        ("is", 1.2, 1.3),
        ("the", 1.4, 1.5),
        ("right", 1.6, 1.7),
        ("cut", 1.85, 1.95),
        ("after", 2.5, 2.6),
    ]
    changed_word = "before" if where == "away" else "the"
    second = (
        []
        if where == "empty"
        else [
            ("changed" if text == changed_word else text, start, end) for text, start, end in render
        ]
    )
    result = _at_cut(render, render, second=second)
    assert result.status == "pass"
    assert result.second_status == ("warn" if warns else "pass")
    assert result.second_comparison["span"] == {"start_s": 1.0, "end_s": 2.0}
    assert result.second_comparison["first"] == ["this", "is", "the", "right", "cut"]
    assert result.first.heard == [word[0] for word in render]
    assert result.second.heard == [word[0] for word in second]
    findings = relisten_findings(result, 1, 1.5)
    if warns:
        (warning,) = findings
        assert warning.check == "join.second_model" and warning.severity == "warn"
        assert "before" not in warning.message and "after" not in warning.message
        assert warning.evidence["first"]["heard"] == result.first.heard
        assert warning.evidence["second"]["heard"] == result.second.heard
    else:
        assert not findings


def test_second_engine_skips_when_only_the_wider_span_has_speech():
    render = [("before", 0.4, 0.5), ("after", 2.5, 2.6)]
    result = _at_cut(render, render, second=render)
    assert result.second_status == "skipped" and result.second_reason == "no speech near the cut"
    assert result.first.heard == result.second.heard == ["before", "after"]
    assert not relisten_findings(result, 1, 1.5)


def test_midpoints_short_segments_and_inserted_gap_use_the_correct_clocks():
    # Only a word's midpoint decides whether the planned edit kept enough of it.
    left = _transcript(
        [
            ("neighbour", 0.5, 0.6),
            ("before", 0.8, 1.0),
            ("kept", 1.3, 1.6),
            ("fragment", 1.45, 1.75),
            ("discarded", 1.5, 1.7),
        ]
    )
    right = _transcript(
        [
            ("discarded", 0.5, 0.7),
            ("fragment", 0.8, 1.1),
            ("kept", 0.9, 1.2),
            ("after", 1.4, 1.6),
            ("neighbour", 1.6, 1.8),
        ]
    )
    render = _transcript(
        [
            ("neighbour", 0.5, 0.6),
            ("before", 0.8, 1.0),
            ("kept", 1.3, 1.5),
            ("inserted", 1.6, 1.7),
            ("kept", 1.8, 2.0),
            ("after", 2.2, 2.4),
            ("neighbour", 2.4, 2.6),
        ]
    )
    # Window starts at render 10 s; left sound ends at 11.5 s, right starts at 11.8 s.
    result = relisten_at(
        [
            render,
            _transcript(
                [
                    ("before", 0.8, 1.0),
                    ("kept", 1.3, 1.5),
                    ("kept", 1.8, 2.0),
                    ("after", 2.2, 2.4),
                ]
            ),
            left,
            right,
        ],
        engine="first",
        left_length_s=0.8,
        right_length_s=0.6,
        j_end_s=11.5,
        j_start_s=11.8,
        window_start_s=10,
    )
    assert result.first.planned == ["before", "kept", "kept", "after"]
    assert result.first.heard == ["before", "kept", "inserted", "kept", "after"]
    assert result.span == pytest.approx({"start_s": 10.7, "end_s": 12.4})
    assert result.status == "warn"
    assert result.first.plan_heard == ["before", "kept", "kept", "after"]
    assert result.first.heard_words[2].start == pytest.approx(11.6)
    assert result.first.heard_words[2].end == pytest.approx(11.7)


def test_normalisation_and_optional_fillers_agree_without_mutating_transcripts():
    left = _transcript([("Gonna", 0.8, 1.0), ("go", 1.1, 1.3)])
    right = _transcript([("home.", 1.2, 1.4)])
    render = _transcript(
        [("going to", 0.8, 1.0), ("um", 1.0, 1.1), ("GO", 1.1, 1.3), ("home", 1.7, 1.9)]
    )
    before = render.to_dict()
    result = relisten_at(
        [render, render, left, right],
        engine="first",
        left_length_s=3,
        right_length_s=3,
        j_end_s=1.5,
        j_start_s=1.5,
        window_start_s=0,
        second_engine="second",
        second_transcripts=[render, render, left, right],
    )
    assert result.status == result.second_status == "pass"
    assert not relisten_findings(result, 1, 1.5)
    assert render.to_dict() == before


def test_empty_readings_are_skipped_and_control_only_differences_are_evidence():
    empty = [Transcript([]) for _ in range(4)]
    result = relisten_at(
        empty,
        engine="first",
        left_length_s=2,
        right_length_s=2,
        j_end_s=1.5,
        j_start_s=1.5,
        window_start_s=0,
        second_engine="second",
        second_transcripts=empty,
    )
    assert result.status == result.second_status == "skipped"
    assert result.reason == result.second_reason == "no speech near the cut"
    assert not relisten_findings(result, 1, 1.5)
    # Speech heard only in the controls does not produce a warning.
    planned = _transcript([("alpha beta gamma delta", 0.2, 1.2)])
    missing = [Transcript([]), Transcript([]), planned, Transcript([])]
    result = relisten_at(
        missing,
        engine="first",
        left_length_s=2,
        right_length_s=2,
        j_end_s=1.5,
        j_start_s=1.5,
        window_start_s=0,
        second_engine="second",
        second_transcripts=missing,
    )
    assert result.status == "pass" and result.second_status == "skipped"
    assert result.first.diff and not result.first.agreement
    assert not relisten_findings(result, 1, 1.5)


def test_second_model_compares_heard_words_even_when_its_controls_disagree():
    heard = _transcript(
        [("alpha", 0.3, 0.4), ("beta", 0.8, 0.9), ("gamma", 1.8, 1.9), ("delta", 2.1, 2.2)]
    )
    left = _transcript([("alpha", 0.3, 0.4), ("beta", 0.8, 0.9)])
    right = _transcript([("gamma", 1.3, 1.4), ("delta", 1.6, 1.7)])
    wrong_left = _transcript([("alpha", 0.3, 0.4), ("wrong", 0.8, 0.9)])
    result = relisten_at(
        [heard, heard, left, right],
        engine="first",
        left_length_s=2,
        right_length_s=2,
        j_end_s=1.5,
        j_start_s=1.5,
        window_start_s=0,
        second_engine="second",
        second_transcripts=[heard, heard, wrong_left, right],
    )
    assert result.status == result.second_status == "pass"
    assert not result.second.agreement and result.second.diff
    assert not relisten_findings(result, 1, 1.5)


def test_verify_skips_empty_readings(tone_media, monkeypatch):
    module = importlib.import_module("yap_studio.verify")
    first = ScriptedBackend(lambda _: Transcript([]), name="first")
    second = ScriptedBackend(lambda _: Transcript([]), name="second")
    monkeypatch.setattr(module, "resolve_backend", lambda *_: first)
    monkeypatch.setattr(module, "resolve_second_backend", lambda _: second)
    row = tone_media.cases["clean"]
    result = module.verify(str(row.render), edl=str(row.edl))
    for check in NEW_CHECKS:
        assert result["checks"][check] == {
            "ran": False,
            "status": "skipped",
            "reason": "no speech near the cut",
        }


def test_verify_skips_missing_source_and_no_cuts(tone_media, monkeypatch, tmp_path):
    module = importlib.import_module("yap_studio.verify")
    monkeypatch.setattr(module, "resolve_backend", lambda *_: ToneBackend())
    monkeypatch.setattr(module, "resolve_second_backend", lambda _: ToneBackend(name="second"))
    row = tone_media.cases["clean"]
    missing = json.loads(row.edl.read_text())
    missing["source"] = str(tmp_path / "missing.mov")
    edl = tmp_path / "missing.json"
    edl.write_text(json.dumps(missing))
    result = module.verify(str(row.render), edl=str(edl))
    for check in NEW_CHECKS:
        assert result["checks"][check] == {
            "ran": False,
            "status": "skipped",
            "reason": "source not found",
        }
    continuous = tmp_path / "continuous.json"
    continuous.write_text(
        json.dumps({"source": str(tone_media.source), "segments": [{"start": 0, "end": 3.5}]})
    )
    result = module.verify(str(tone_media.source), edl=str(continuous))
    for check in NEW_CHECKS:
        assert result["checks"][check] == {
            "ran": False,
            "status": "skipped",
            "reason": "cut coverage: not exposed",
        }


def test_detect_mode_has_no_plan_for_new_checks(tone_media, monkeypatch):
    module = importlib.import_module("yap_studio.verify")
    monkeypatch.setattr(module, "resolve_backend", lambda *_: None)
    monkeypatch.setattr(module, "resolve_second_backend", lambda _: None)
    result = module.verify(str(tone_media.cases["clean"].render), asr="none")
    for check in NEW_CHECKS:
        assert result["checks"][check] == {
            "ran": False,
            "status": "skipped",
            "reason": "needs the source and edit list",
        }
