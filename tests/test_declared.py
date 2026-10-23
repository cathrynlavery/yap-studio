"""Joins declared on the output clock, checked by the words heard across them."""

from __future__ import annotations

import json
import wave
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import synth

from yap_studio import audio
from yap_studio.checks import declared
from yap_studio.edl import JoinAssertion
from yap_studio.plan import snap_frame
from yap_studio.transcript import ScriptedBackend, Transcript, Word, resolve_backend

SR = 8000
JOIN = JoinAssertion(
    index=1,
    out_frame=274,
    before="going kind of viral",
    after="Now if you don't",
    absent=("mansplainers",),
    once=("kind of viral",),
)
JOIN_S = 274 / 30


def _transcript(text: str, start: float = 1.5) -> Transcript:
    return Transcript(
        [
            Word(word, start + 0.3 * index, start + 0.3 * index + 0.25)
            for index, word in enumerate(text.split())
        ]
    )


def _backend(windows: dict[str, str], whole: str, calls: list[list[Path]]) -> ScriptedBackend:
    class Recording(ScriptedBackend):
        def transcribe_many(self, paths, *, language=None):
            calls.append(list(paths))
            return super().transcribe_many(paths, language=language)

    def script(path: Path) -> Transcript:
        if path.stem == "declared-render":
            return _transcript(whole, start=1.0 + JOIN_S - 4)
        lead = path.stem.rsplit("-", 1)[1]
        words = windows[lead].split()
        # Place the words so "now" starts 50 ms after the join (1 s of WAV padding first).
        first_after = words.index("now") if "now" in words else 0
        return _transcript(windows[lead], start=1.0 + float(lead) + 0.05 - 0.3 * first_after)

    return Recording(script)


def _placed(words: list[tuple[str, float]]) -> ScriptedBackend:
    """Words at fixed offsets from the join, in every window."""

    def script(path: Path) -> Transcript:
        lead = float(path.stem.rsplit("-", 1)[1])
        return Transcript(
            [Word(text, 1.0 + lead + at, 1.0 + lead + at + 0.25) for text, at in words]
        )

    return ScriptedBackend(script)


@pytest.fixture
def render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "render.mp4"
    path.write_bytes(b"not decoded")
    sound = np.random.default_rng(5).normal(0, 0.1, 20 * SR).astype(np.float32)
    monkeypatch.setattr(audio, "decode", lambda *args, **kwargs: (SR, sound))
    return path


GOOD = "people keep going kind of viral now if you don't know"
WHOLE = "this is going kinda viral um now if you don't know what I mean"


def _check(render: Path, tmp_path: Path, backend, joins=(JOIN,)):
    return declared.check_declared_joins(
        render, list(joins), backend=backend, fps=Fraction(30), work_dir=tmp_path / "work"
    )


def test_declared_join_passes_when_words_run_across_it_in_both_windows(render, tmp_path):
    calls: list[list[Path]] = []
    backend = _backend({"5": GOOD, "9": "so " + GOOD}, WHOLE, calls)
    result = _check(render, tmp_path, backend)
    assert len(calls) == 1
    assert [path.name for path in calls[0]] == [
        "declared-join-01-lead-5.wav",
        "declared-join-01-lead-9.wav",
        "declared-render.wav",
    ]
    row = result.joins[0]
    assert row.status == "pass"
    assert row.expected == ["going", "kind", "of", "viral", "now", "if", "you", "don't"]
    assert [window.contiguous for window in row.windows] == [True, True]
    assert row.absent[0].count == 0
    assert row.once[0].count == 1
    assert [finding.severity for finding in result.findings] == ["info"]
    finding = result.findings[0]
    assert finding.check == "join.words"
    assert finding.cut is None
    assert finding.time_s == pytest.approx(JOIN_S)
    assert finding.evidence["source"] == "declared"
    assert finding.evidence["declared_join"] == 1
    assert "is heard across the join, as declared" in finding.message
    json.dumps(result.to_dict(), allow_nan=False)


def test_windows_hold_the_lead_in_and_tail_on_the_render_clock(render, tmp_path):
    _check(render, tmp_path, _backend({"5": GOOD, "9": GOOD}, WHOLE, []))
    for lead in (5, 9):
        with wave.open(str(tmp_path / "work" / f"declared-join-01-lead-{lead}.wav")) as wav:
            assert wav.getframerate() == SR
            # One second of padding before, half a second after.
            assert wav.getnframes() == round((1 + lead + declared.AFTER_S + 0.5) * SR)


def test_declared_join_fails_when_a_window_does_not_hear_the_words_in_a_row(render, tmp_path):
    heard = "going kind of viral mansplainers now if you don't"
    result = _check(render, tmp_path, _backend({"5": GOOD, "9": heard}, WHOLE, []))
    assert result.joins[0].status == "fail"
    failure = next(f for f in result.findings if f.severity == "fail")
    assert "is not heard across the join" in failure.message
    assert heard in failure.message
    assert failure.suggested_action
    assert [w["contiguous"] for w in failure.evidence["windows"]] == [True, False]
    assert failure.evidence["windows"][1]["heard"] == heard


def test_declared_join_fails_when_the_words_sit_away_from_the_join(render, tmp_path):
    # "good morning" is heard 3 s before the declared cut; the cut itself plays silence.
    join = JoinAssertion(3, 274, "good", "morning")
    result = _check(render, tmp_path, _placed([("good", -3.0), ("morning", -2.7)]), [join])
    row = result.joins[0]
    assert row.status == "fail"
    assert [window.in_order for window in row.windows] == [True, True]
    assert [window.contiguous for window in row.windows] == [False, False]
    assert row.windows[0].heard_at_s[0] == pytest.approx(JOIN_S - 3.0)
    failure = next(f for f in result.findings if f.severity == "fail")
    assert "is not heard across the join" in failure.message
    assert "away from the join" in failure.message


@pytest.mark.parametrize(
    "before,after,words,status",
    [
        ("good morning", "", [("good", -0.8), ("morning", -0.5)], "pass"),
        ("good morning", "", [("good", -2.8), ("morning", -2.5)], "fail"),
        ("", "good morning", [("good", 0.1), ("morning", 0.4)], "pass"),
        ("", "good morning", [("good", 2.0), ("morning", 2.3)], "fail"),
        # Both sides heard in a row, but across a 3.75 s silence at the cut.
        ("good", "morning", [("good", -2.0), ("morning", 1.5)], "fail"),
        # Word timings may overlap the join slightly either way.
        ("good", "morning", [("good", -0.6), ("morning", -0.25)], "pass"),
        ("good", "morning", [("good", 0.0), ("morning", 0.3)], "pass"),
    ],
)
def test_declared_words_must_sit_next_to_the_join(render, tmp_path, before, after, words, status):
    join = JoinAssertion(4, 274, before, after)
    result = _check(render, tmp_path, _placed(words), [join])
    assert result.joins[0].status == status


def test_absent_and_once_phrases_are_checked_on_the_whole_render(render, tmp_path):
    whole = "going kind of viral mansplainers again kind of viral now if you don't"
    result = _check(render, tmp_path, _backend({"5": GOOD, "9": GOOD}, whole, []))
    messages = [f.message for f in result.findings if f.severity == "fail"]
    assert len(messages) == 2
    assert any('"mansplainers" should be cut but is heard at' in text for text in messages)
    assert any('"kind of viral" is heard 2 times' in text for text in messages)
    row = result.joins[0]
    assert row.status == "fail"
    assert row.absent[0].count == 1
    assert row.absent[0].times_s[0] == pytest.approx(JOIN_S - 4 + 4 * 0.3)
    missing = _check(render, tmp_path, _backend({"5": GOOD, "9": GOOD}, "nothing here", []))
    assert any('"kind of viral" is not heard' in f.message for f in missing.findings)


def test_joins_without_absent_or_once_skip_the_whole_render(render, tmp_path):
    calls: list[list[Path]] = []
    plain = JoinAssertion(2, 150, "going kind of viral", "now if you don't")
    result = _check(render, tmp_path, _backend({"5": GOOD, "9": GOOD}, WHOLE, calls), [plain])
    assert [path.name for path in calls[0]] == [
        "declared-join-02-lead-5.wav",
        "declared-join-02-lead-9.wav",
    ]
    assert result.joins[0].status == "pass"
    assert declared.check_declared_joins(
        render, [], backend=ScriptedBackend(lambda _: pytest.fail("no joins")),
        fps=30, work_dir=tmp_path,
    ).findings == []


@pytest.fixture(scope="module")
def declared_speech(tmp_path_factory):
    sentences = [
        "Bright planets circle slowly over quiet hills on calm winter evenings.",
        "An extraordinary discovery brings people together near the open garden.",
        "Seven green boxes wait beside the open door near our garden.",
    ]
    directory = tmp_path_factory.mktemp("declared-speech")
    source, words = synth.speech_source(directory / "speech.mov", sentences)
    ends = [max(word.end for word in words if word.sentence == i) for i in range(3)]
    starts = [min(word.start for word in words if word.sentence == i) for i in range(3)]
    spans = [(0.0, ends[0] + 0.15), (starts[2] - 0.15, ends[2] + 0.4)]
    render = synth.render_clean(source, spans, directory / "render.mp4", fade_s=0.005)
    out_frame = snap_frame(Fraction(str(spans[0][1])), Fraction(30))
    return SimpleNamespace(source=source, render=render, out_frame=out_frame, dir=directory)


@pytest.mark.asr
def test_real_speech_declared_join(declared_speech, tmp_path: Path) -> None:
    backend = resolve_backend("auto")
    if backend is None:
        pytest.skip("no speech engine installed")
    join = JoinAssertion(
        1,
        declared_speech.out_frame,
        "calm winter evenings",
        # Synthetic speech makes "boxes" ambiguous, so stop before it.
        "Seven green",
        absent=("extraordinary discovery",),
        once=("quiet hills",),
    )
    result = declared.check_declared_joins(
        declared_speech.render, [join], backend=backend, fps=30, work_dir=tmp_path / "cut"
    )
    assert result.joins[0].status == "pass", result.to_dict()
    uncut = declared.check_declared_joins(
        declared_speech.source, [join], backend=backend, fps=30, work_dir=tmp_path / "uncut"
    )
    row = uncut.joins[0]
    assert row.status == "fail", uncut.to_dict()
    assert not all(window.contiguous for window in row.windows)
    assert row.absent[0].count == 1
