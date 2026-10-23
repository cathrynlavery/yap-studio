"""Caption words checked against cached speech on the final's clock."""

from __future__ import annotations

import importlib
import json
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest
import synth

from yap_studio import cache
from yap_studio.checks import caption_words as stage
from yap_studio.checks.caption_layout import CaptionWord, write_layout_record
from yap_studio.cli import main
from yap_studio.faces import detect
from yap_studio.transcript import ScriptedBackend, Transcript, Word
from yap_studio.verify import _NO_AUDIO_REASON, _NO_ENGINE_REASON, _verify, verify


def test_heard_word_missing_from_captions_warns_at_its_time() -> None:
    words = {"last": (CaptionWord("today", 94, 94.3),),
             "first": (CaptionWord("answer", 93.2, 93.5),)}
    heard = Transcript([Word("answer", 93.2, 93.5), Word("like", 93.7, 93.9),
                        Word("today", 94, 94.3)])
    finding, = stage.check_caption_words(words, heard)
    assert finding.check == "captions.words" and finding.severity == "warn"
    assert finding.message == 'Heard "like" at 1:33.7 with no caption.'
    assert finding.time_s == 93.7
    assert finding.evidence == {"heard": ["like"], "captioned": [], "start_s": 93.7, "end_s": 93.9}
    assert "undo the removal" in finding.suggested_action


@pytest.mark.parametrize(("masked", "spoken"), [("F***,", "fuck"), ("s***-show", "shit-show")])
def test_masked_caption_and_overlapping_heard_words_are_skipped(masked, spoken) -> None:
    words = {"one": (CaptionWord("hello", 0, .3), CaptionWord(masked, .4, .8),
                     CaptionWord("world", .9, 1.2))}
    heard = Transcript([Word("hello", 0, .3), Word(spoken, .35, .85),
                        Word("world", .9, 1.2)])
    assert stage.check_caption_words(words, heard) == []
    assert stage.check_caption_words(words, Transcript([Word("hello", 0, .3),
                                                       Word("world", .9, 1.2)])) == []


def test_heard_neighbours_with_midpoints_outside_the_mask_are_kept() -> None:
    words = {"one": (CaptionWord("hello", .1, .4), CaptionWord("F***", .4, .8),
                     CaptionWord("world", .8, 1.1))}
    heard = Transcript([Word("hello", .1, .42), Word("fuck", .38, .82), Word("world", .78, 1.1)])
    assert stage.check_caption_words(words, heard) == []


@pytest.mark.parametrize(("text", "spoken"), [
    ("holy s***", [Word("holy", .4, .5), Word("ship", .6, .8)]),
    ("holy s***", [Word("holy", .4, .5)]),
    ("cafe\u0301 s***", [Word("café", .4, .5), Word("ship", .6, .8)]),
    ("s*** hits", [Word("shit", .4, .5), Word("hits", .6, .8)]),
    ("holy s*** today", [Word("holy", .4, .5), Word("ship", .51, .55),
                          Word("today", .56, .8)]),
])
def test_masks_inside_merged_words_keep_the_unmasked_speech(text, spoken) -> None:
    words = {"one": (CaptionWord("before", .1, .42), CaptionWord(text, .4, .8),
                     CaptionWord("after", .8, 1.1))}
    heard = Transcript([Word("before", .1, .42), *spoken, Word("after", .78, 1.1)])
    assert stage.check_caption_words(words, heard) == []


def test_unmasked_words_inside_a_merged_mask_are_still_checked() -> None:
    finding, = stage.check_caption_words(
        {"one": (CaptionWord("holy s***", .4, .8),)},
        Transcript([Word("ship", .6, .8)]),
    )
    assert finding.evidence["captioned"] == ["holy"]


def test_non_ascii_substitution_is_info() -> None:
    finding, = stage.check_caption_words(
        {"one": (CaptionWord("猫", 0.3, 0.7),)}, Transcript([Word("犬", 0.3, 0.7)]),
    )
    assert finding.severity == "info" and finding.time_s == 0.3
    assert finding.evidence == {"heard": ["犬"], "captioned": ["猫"],
                                "start_s": 0.3, "end_s": 0.7}


def test_non_ascii_heard_word_without_caption_warns_at_its_time() -> None:
    finding, = stage.check_caption_words({}, Transcript([Word("猫", 93.7, 93.9)]))
    assert finding.severity == "warn" and finding.time_s == 93.7
    assert finding.message == 'Heard "猫" at 1:33.7 with no caption.'
    assert finding.evidence == {"heard": ["猫"], "captioned": [], "start_s": 93.7, "end_s": 93.9}


def test_canonically_equivalent_accents_match() -> None:
    assert stage.check_caption_words(
        {"one": (CaptionWord("café", 0.3, 0.7),)}, Transcript([Word("cafe\u0301", 0.3, 0.7)]),
    ) == []


def test_caption_word_not_heard_warns_at_its_time() -> None:
    finding, = stage.check_caption_words({"one": (CaptionWord("word", 23.8, 24),)}, Transcript([]))
    assert finding.severity == "warn" and finding.time_s == 23.8
    assert finding.message == 'Caption "word" at 0:23.8 is not heard in the audio.'
    assert finding.evidence == {"heard": [], "captioned": ["word"], "start_s": 23.8, "end_s": 24}
    assert "fix the cut or the caption text" in finding.suggested_action


@pytest.mark.parametrize("captioned,heard", [
    ("3", "three"), ("R's", "Rs"), ("R’s", "Rs"),
    ("non techies", "nontechies"), ("s t", "st"), ("nontechies", "non techies"),
    ("um hello", "hello uh"), ("uh", "um"),
])
def test_normalized_words_match(captioned: str, heard: str) -> None:
    assert stage.check_caption_words(
        {"one": (CaptionWord(captioned, 0.3, 0.7),)}, Transcript([Word(heard, 0.3, 0.7)]),
    ) == []


@pytest.mark.parametrize("heard,severity", [("answer", "info"), ("answer like", "warn")])
def test_substitution_severity_follows_token_count(heard: str, severity: str) -> None:
    finding, = stage.check_caption_words(
        {"one": (CaptionWord("response", 93.2, 93.5),)}, Transcript([Word(heard, 93.3, 93.9)]),
    )
    assert finding.severity == severity and finding.time_s == 93.2
    assert finding.message == f'Heard "{heard}" at 1:33.2 where the caption shows "response".'
    assert finding.evidence == {"heard": heard.split(), "captioned": ["response"],
                                "start_s": 93.2, "end_s": 93.5}


@pytest.fixture
def matched_speech() -> Transcript:
    return Transcript([Word("Amber", 0, .2), Word("lantern", .4, .6), Word("glows", .8, 1)])


@pytest.mark.parametrize("offset", [5, -5])
def test_matching_words_shifted_warn_once_at_the_caption_start(matched_speech, offset) -> None:
    words = {"one": tuple(CaptionWord(word.text, word.start + max(offset, 0),
                                      word.end + max(offset, 0)) for word in matched_speech.words)}
    heard = Transcript([Word(word.text, word.start + max(-offset, 0), word.end + max(-offset, 0))
                        for word in matched_speech.words])
    finding, = stage.check_caption_words(words, heard)
    assert finding.check == "captions.words" and finding.severity == "warn"
    assert finding.time_s == max(offset, 0)
    assert finding.message == (
        f'Captions "amber lantern glows" at 0:0{max(offset, 0)}.0 start 5.0 s '
        f'{"after" if offset > 0 else "before"} the matching speech '
        f'at 0:0{max(-offset, 0)}.0 (median word offset).'
    )
    assert finding.evidence == {
        "heard": ["amber", "lantern", "glows"], "captioned": ["amber", "lantern", "glows"],
        "start_s": max(offset, 0), "end_s": max(offset, 0) + 1,
        "heard_start_s": max(-offset, 0), "heard_end_s": max(-offset, 0) + 1,
        "offset_s": offset, "matched_words": 3,
    }
    assert "Align the caption word starts" in finding.suggested_action


@pytest.mark.parametrize("offsets", [(0, 0, 0), (.08, -.05, .03), (.5, .5, .5), (0, .7, .7)])
def test_aligned_words_jitter_and_short_offsets_do_not_warn(matched_speech, offsets) -> None:
    words = {"one": tuple(CaptionWord(word.text, word.start + offset, word.end + offset + .3)
                          for word, offset in zip(matched_speech.words, offsets, strict=True))}
    assert stage.check_caption_words(words, matched_speech) == []


def test_matching_words_with_growing_drift_warn_at_the_affected_run() -> None:
    text = "Amber lantern glows beside quiet rivers carrying silver leaves".split()
    offsets = [0, .1, .2, .3, .4, .5, .6, .7, .8]
    heard = Transcript([Word(word, i * 10, i * 10 + .3) for i, word in enumerate(text)])
    words = {"one": tuple(CaptionWord(word.text, word.start + offset, word.end + offset)
                          for word, offset in zip(heard.words, offsets, strict=True))}
    finding, = stage.check_caption_words(words, heard)
    assert finding.time_s == 60.6
    assert 'at 1:00.6' in finding.message
    assert finding.evidence["offset_s"] == pytest.approx(.7)
    assert finding.evidence["matched_words"] == 3


@pytest.mark.parametrize("text", ["the", "the answer", "go go go", "amber lantern glows " * 2])
def test_short_or_repeated_sequences_are_not_timing_anchors(text) -> None:
    heard = Transcript([Word(word, i, i + .3) for i, word in enumerate(text.split())])
    words = {"one": tuple(CaptionWord(word.text, word.start + 5, word.end + 5)
                          for word in heard.words)}
    assert stage.check_caption_words(words, heard) == []


def test_repeated_phrase_does_not_anchor_to_the_wrong_occurrence(matched_speech) -> None:
    words = {"one": tuple(CaptionWord(word.text, word.start + 5, word.end + 5)
                          for word in matched_speech.words)}
    heard = Transcript([*matched_speech.words,
                        *(Word(word.text, word.start + 5, word.end + 5)
                          for word in matched_speech.words)])
    findings = stage.check_caption_words(words, heard)
    assert all("offset_s" not in finding.evidence for finding in findings)


@pytest.mark.parametrize("occurrence", [False, True])
def test_display_fixes_do_not_make_a_timing_run(occurrence) -> None:
    from yap_studio.captions import CaptionWord as SourceWord
    from yap_studio.captions import build_phrases

    text = "Amber lantern glows above clouds"
    heard = Transcript([Word(word, i * .8, i * .8 + .3)
                        for i, word in enumerate(text.split())])
    source = ([SourceWord("w0", "placeholder", 0, 3.5)] if occurrence else
              [SourceWord(f"w{i}", word.text, word.start, word.end)
               for i, word in enumerate(heard.words)])
    phrases = build_phrases(
        source, {"max_words_per_phrase": 5, "max_characters_per_phrase": 80},
        word_text_fixes={"w0": text} if occurrence else {},
        word_fixes={} if occurrence else {text: text},
    )
    words = {str(phrase.index): tuple(CaptionWord(word.text, word.start, word.end)
                                     for word in phrase.words) for phrase in phrases}
    assert [word.text for phrase in phrases for word in phrase.words] == [text]
    assert stage.check_caption_words(words, heard) == []


@pytest.fixture
def audio_path(tmp_path: Path) -> Path:
    target = tmp_path / "voice.wav"
    sound = 0.1 * np.sin(np.arange(48_000) * 2 * np.pi * 220 / 48_000)
    synth._write_wav(target, sound, 48_000)
    return target


def test_heard_transcript_reuses_decoded_audio_cache(audio_path: Path, tmp_path: Path) -> None:
    calls = []
    expected = Transcript([Word("hello", 0.4, 0.7)])

    def script(target: Path) -> Transcript:
        calls.append(target)
        with wave.open(str(target), "rb") as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16_000)
            assert wav.getnframes() == 16_000
        return expected

    backend = ScriptedBackend(script)
    assert stage.heard_transcript(audio_path, backend) == expected
    other = tmp_path / "same-voice.wav"
    shutil.copyfile(audio_path, other)
    assert stage.heard_transcript(other, backend) == expected
    assert len(calls) == 1
    saved, = cache.cache_dir("speech", "heard").glob("*.json")
    assert json.loads(saved.read_text()) == expected.to_dict()


@pytest.mark.parametrize("changed", ["name", "model", "version", "audio"])
def test_heard_cache_depends_on_audio_and_engine(
    audio_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str,
) -> None:
    calls = []
    backend = ScriptedBackend(lambda target: calls.append(target) or Transcript([]))
    stage.heard_transcript(audio_path, backend)
    if changed == "version":
        monkeypatch.setattr(backend, "version", lambda: "1")
    elif changed == "audio":
        synth._write_wav(audio_path, np.zeros(48_000), 48_000)
    else:
        setattr(backend, changed, "other")
    stage.heard_transcript(audio_path, backend)
    assert len(calls) == 2


@pytest.mark.parametrize("broken", ["{", "[]", '{"schema":"unknown"}',
                                    '{"schema":"yap-studio/transcript@1"}',
                                    '{"schema":"yap-studio/transcript@1","words":[{}]}',
                                    '{"schema":"yap-studio/transcript@1","words":'
                                    '[{"text":"hello","start":NaN,"end":1}]}'])
def test_unreadable_heard_cache_transcribes_again(audio_path: Path, broken: str) -> None:
    calls = []
    expected = Transcript([Word("hello", 0.4, 0.7)])
    backend = ScriptedBackend(lambda target: calls.append(target) or expected)
    stage.heard_transcript(audio_path, backend)
    target, = cache.cache_dir("speech", "heard").glob("*.json")
    target.write_text(broken)
    assert stage.heard_transcript(audio_path, backend) == expected
    assert len(calls) == 2


@pytest.mark.parametrize("failure", ["directory", "write"])
def test_unwritable_heard_cache_still_returns_transcript(
    audio_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    def denied(*args, **kwargs):
        raise PermissionError("cache unavailable")

    if failure == "directory":
        monkeypatch.setattr(stage.cache, "cache_dir", denied)
    else:
        monkeypatch.setattr(stage.paths, "atomic_write_text", denied)
    expected = Transcript([Word("hello", 0.4, 0.7)])
    assert stage.heard_transcript(audio_path, ScriptedBackend(lambda _: expected)) == expected


@pytest.mark.parametrize("voice_stem", [False, True])
def test_verify_heard_word_missing_from_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, voice_stem: bool,
) -> None:
    module = importlib.import_module("yap_studio.verify")
    resolve_backend = module.resolve_backend
    media = synth.calib_source(tmp_path / "final.mov", seconds=2, size="320x180")
    document = synth.calibration_layout()
    document["captions"] = [document["captions"][0]]
    document["captions"][0].update(
        text="hello", start_frame=12, end_frame=24,
        words=[{"text": "hello", "start_s": 0.4, "end_s": 0.7}],
    )
    document["elements"] = []
    write_layout_record(media, document)
    monkeypatch.setattr(detect, "unavailable_reason", lambda: "not measured in this test")
    heard = Transcript([Word("hello", 0.4, 0.7), Word("like", 1.2, 1.4)])
    monkeypatch.setattr(module, "resolve_backend", lambda *_: ScriptedBackend(lambda _: heard))
    decoded = []
    original = stage.audio_tools.decode

    def decode(target, **kwargs):
        decoded.append(target)
        return original(target, **kwargs)

    monkeypatch.setattr(stage.audio_tools, "decode", decode)
    stem = tmp_path / "voice.wav" if voice_stem else None
    if stem is not None:
        synth._write_wav(stem, np.zeros(2 * 48_000), 48_000)
    result = _verify(str(media), click_audio=stem, loudness="note")
    finding, = [item for item in result["findings"] if item["check"] == "captions.words"]
    assert finding["severity"] == "warn" and finding["time_s"] == 1.2
    assert finding["message"] == 'Heard "like" at 0:01.2 with no caption.'
    assert result["checks"]["captions.words"]["ran"] is True
    assert result["checks"]["captions.words"]["reason"] is None
    assert decoded[-1] == (stem or media)

    monkeypatch.setattr(module, "resolve_backend", resolve_backend)
    result = verify(str(media), asr="none", loudness="note")
    assert result["checks"]["captions.words"]["ran"] is False
    assert result["checks"]["captions.words"]["reason"] == _NO_ENGINE_REASON
    assert not any(item["check"] == "captions.words" for item in result["findings"])


def test_verify_shifted_matching_words_reach_json_and_agent_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, matched_speech, capsys,
) -> None:
    module = importlib.import_module("yap_studio.verify")
    media = synth.calib_source(tmp_path / "final.mov", seconds=7, size="320x180")
    document = synth.calibration_layout()
    document["captions"] = [document["captions"][0]]
    document["captions"][0].update(
        text="Amber lantern glows", start_frame=150, end_frame=190,
        words=[{"text": word.text, "start_s": word.start + 5, "end_s": word.end + 5}
               for word in matched_speech.words],
    )
    document["elements"] = []
    write_layout_record(media, document)
    monkeypatch.setattr(detect, "unavailable_reason", lambda: "not measured in this test")
    monkeypatch.setattr(module, "resolve_backend", lambda *_: ScriptedBackend(
        lambda _: matched_speech))
    assert main(["verify", str(media), "--json", "--loudness", "note"]) == 0
    result = json.loads(capsys.readouterr().out)
    finding, = [item for item in result["findings"] if item["check"] == "captions.words"]
    assert result["checks"]["captions.words"]["ran"] is True
    assert result["checks"]["captions.words"]["status"] == "warn"
    assert finding["severity"] == "warn" and finding["time_s"] == 5
    assert finding["evidence"]["offset_s"] == 5
    assert main(["verify", str(media), "--loudness", "note"]) == 0
    summary = capsys.readouterr().out
    assert 'Captions "amber lantern glows" at 0:05.0' in summary
    assert "5.0 s after" in summary
    assert "Align the caption word starts" in summary


@pytest.mark.parametrize("state,reason", [
    ("missing", "no layout record next to this render"),
    ("corrupt", "corrupt layout record"),
    ("unmeasured", "caption layout was not measured: no geometry"),
    ("empty", "no captions in this render"),
    ("silent", _NO_AUDIO_REASON),
])
def test_verify_caption_words_skipped_with_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str, reason: str,
) -> None:
    module = importlib.import_module("yap_studio.verify")
    media = synth.calib_source(tmp_path / "source.mov", seconds=1, size="320x180")
    if state == "silent":
        target = tmp_path / "silent.mov"
        synth._run(["ffmpeg", "-v", "error", "-y", "-i", str(media),
                    "-map", "0:v:0", "-c:v", "copy", "-an", str(target)])
        media = target
    document = synth.calibration_layout()
    document["elements"] = []
    if state == "unmeasured":
        document.update(measured=False, reason="no geometry")
    elif state == "empty":
        document["captions"] = []
    if state == "corrupt":
        media.with_suffix(".layout.json").write_text("{")
    elif state != "missing":
        write_layout_record(media, document)
    monkeypatch.setattr(detect, "unavailable_reason", lambda: "not measured in this test")
    monkeypatch.setattr(module, "resolve_backend", lambda *_: ScriptedBackend(
        lambda _: pytest.fail("skipped caption check must not transcribe")))
    result = verify(str(media), loudness="note")
    coverage = result["checks"]["captions.words"]
    assert coverage["ran"] is False
    assert reason in coverage["reason"]
    assert not any(item["check"] == "captions.words" for item in result["findings"])
