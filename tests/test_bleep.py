"""Synthetic word masking and sample-level bleep coverage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import synth

from yap_studio.captions import CaptionWord
from yap_studio.project.model import AUDIO_DEFAULTS
from yap_studio.sfx import bleep
from yap_studio.sfx import made as made_sfx
from yap_studio.stages.assemble_audio import decode, render_audio
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline
from yap_studio.style import StyleError, resolve_style_data


def words(*tokens: str) -> list[CaptionWord]:
    return [CaptionWord(str(i), text, 1 + i * .3, 1.2 + i * .3)
            for i, text in enumerate(tokens)]


@pytest.mark.parametrize("token", [
    "FUCKING", "shit!", "(bitch)", "fuckin’", "fuckin'", "bull-shit",
    "motherfuckers", "goddamned", "pissing", "wankers",
])
def test_detection_forms(token: str) -> None:
    found, = bleep.find(words(token), bleep.listed({}))
    assert found.indices == (0,) and found.ids == ("0",)
    assert found.masked == (bleep.mask_word(token),)


def test_compounds_and_lists_use_each_token_once() -> None:
    found = bleep.find(words("bull", "shit", "mother", "fucker", "fuck"), bleep.listed({}))
    assert [item.indices for item in found] == [(0, 1), (2, 3), (4,)]
    assert found[0].masked == ("b***", "s***")
    custom = bleep.listed({"bleep_words_add": ["crap"], "bleep_words_remove": ["fuck"]})
    assert len(bleep.find(words("crap", "fuck"), custom)) == 1
    assert not bleep.find(words("class", "assess", "Scunthorpe", "shitake", "ass", "Dick",
                                "cock", "damn", "hell", "crap"), bleep.listed({}))
    assert bleep.builtin_words() == tuple(sorted(set(bleep.builtin_words())))


@pytest.mark.parametrize(("token", "expected"), [
    ("shit-show", "s***-show"), ("Fuck-you!", "F***-you!"),
    ("pre-shit-post", "pre-s***-post"), ("shit-fuck", "s***-f***"),
    ("shit--show", "s***--show"),
    ("god-damn-it", "g**-d***-it"), ("pre-god-damn-post!", "pre-g**-d***-post!"),
    ("shıt-show", "s***-show"), ("ＳＨＩＴ-show", "Ｓ***-show"),
])
def test_hyphen_parts_are_masked_and_the_whole_timed_token_is_checked(token, expected) -> None:
    source = words(token, "today")
    vocabulary = bleep.listed({})
    found, = bleep.find(source, vocabulary)
    assert found.indices == (0,) and found.masked == (expected,)
    assert bleep.is_listed(token, vocabulary)
    assert bleep.mask_text(f"A {token} today", vocabulary) == f"A {expected} today"
    assert bleep.masked(expected)
    assert not bleep.masked(f"{expected} today")
    original = np.full((3 * bleep.RATE, 1), .2, dtype=np.float32)
    voice, tone, _ = bleep.render([found], original, 3, len(original))
    options = dict(music={}, original_voice=original, tone_stem=tone,
                   phrases={"phrases": [{"text": expected, "start": 1, "end": 1.2}]},
                   title_layers=[], timeline=ResolvedTimeline((), (), (), 90, 0))
    assert bleep.check(source, voice=voice, **options)["pass"]
    assert not bleep.check(source, voice=original, **options)["pass"]
    options["phrases"]["phrases"][0]["text"] = token
    assert not bleep.check(source, voice=voice, **options)["pass"]
    assert not bleep.is_listed("shitake-show", vocabulary)


def test_unicode_and_hyphen_compounds_stay_masked() -> None:
    vocabulary = bleep.listed({})
    assert bleep.normal("(ＳＨıＴ!)") == "shit"
    for text in ("shıt hits", "god-damn-it hits", "pre-god-damn-post"):
        censored = bleep.mask_text(text, vocabulary)
        assert censored != text
        assert bleep.mask_text(censored, vocabulary) == censored
        assert not bleep.find_display(words(censored), vocabulary)
    assert not bleep.is_listed("god--damn-it", vocabulary)


@pytest.mark.parametrize(("tokens", "fixes", "indices", "masked"), [
    (("ship",), {"ship": "shit"}, (0,), ("s***",)),
    (("shit",), {"shit": "ship"}, (), ()),
    (("good", "dam"), {"good dam": "goddamn"}, (0, 1), ("g******",)),
    (("ship",), {"ship": "oh god damn"}, (0,), ("g**", "d***")),
])
def test_detection_uses_corrected_display_words(tokens, fixes, indices, masked) -> None:
    source = words(*tokens)
    found = bleep.find_display(source, bleep.listed({}), word_fixes=fixes)
    assert len(found) == bool(indices)
    if found:
        assert found[0].indices == indices
        assert found[0].ids == tuple(source[index].id for index in indices)
        assert (found[0].start_s, found[0].end_s) == (source[0].start, source[-1].end)
        assert found[0].masked == masked


@pytest.mark.parametrize(("raw", "corrected", "count"), [("ship", "shit", 1),
                                                        ("shit", "ship", 0)])
def test_check_uses_the_same_word_fixes_as_detection(raw, corrected, count) -> None:
    source = words(raw)
    fixes = {raw: corrected}
    original = np.full((3 * bleep.RATE, 1), .2, dtype=np.float32)
    occurrences = bleep.find_display(source, bleep.listed({}), word_fixes=fixes)
    voice, tone, _ = bleep.render(occurrences, original, 3, len(original))
    options = dict(music={}, original_voice=original, tone_stem=tone,
                   phrases={"phrases": [{"text": "s***" if count else "ship",
                                          "start": 1, "end": 1.2}]},
                   title_layers=[], timeline=ResolvedTimeline((), (), (), 90, 0),
                   word_fixes=fixes)
    check = bleep.check(source, voice=voice, **options)
    assert check["pass"] and check["words"] == count
    if count:
        assert not bleep.check(source, voice=original, **options)["pass"]


def test_corrected_detection_keeps_reused_ids_on_their_output_times() -> None:
    source = [CaptionWord("same", "ship", 1, 1.2), CaptionWord("other", "it", 1.3, 1.5),
              CaptionWord("same", "ship", 2, 2.2)]
    found, = bleep.find_display(source, bleep.listed({}), word_fixes={"ship it": "shit"})
    assert found.indices == (0, 1) and found.ids == ("same", "other")
    assert (found.start_s, found.end_s) == (1, 1.5)


@pytest.mark.parametrize(("text", "expected"), [
    ("Fuck,", "F***,"), ("fuckin'", "f*****'"), ("(BULL-shit!)", "(B***-****!)"),
    ("fuckin’", "f*****’"), ("a1-b2", "a*-**"),
])
def test_mask_word(text: str, expected: str) -> None:
    assert bleep.mask_word(text) == expected
    assert bleep.masked(expected)
    assert not bleep.masked(text)


def test_free_text_preserves_whitespace_punctuation_and_safe_text() -> None:
    text = "First,\tFUCK!\n(bull  shit), mother\nfucker."
    assert bleep.mask_text(text, bleep.listed({})) == "First,\tF***!\n(b***  s***), m*****\nf*****."
    safe = "  Café! class\tshitake\nword's---end  "
    assert bleep.mask_text(safe, bleep.listed({})) == safe
    assert not bleep.masked("first") and not bleep.masked("f*** word")
    custom = bleep.listed({"bleep_words_add": ["can't"]})
    assert bleep.normal("CAN‘T") == "can't"
    assert bleep.mask_text("CAN‘T", custom) == "C**‘*"


@pytest.mark.parametrize("text", ["s*** hits the fan", "s*** show", "s*** hit", "f*** uck",
                                  "f*****' hits", "C**‘* hits", "B***-**** hits",
                                  "s***-show hits"])
def test_masks_are_atomic_barriers_to_compounds(text: str) -> None:
    vocabulary = bleep.listed({"bleep_words_add": ["s", "f", "c"]})
    assert bleep.mask_text(text, vocabulary) == text
    assert not bleep.find(words(*text.split()), vocabulary)
    assert not bleep.find_display(words(text), vocabulary)


@pytest.mark.parametrize(("text", "expected"), [("show*shit", "show*s***"),
                                               ("**shit**", "**s*****"),
                                               ("s***shit", "s***s***")])
def test_literal_asterisks_do_not_hide_unmasked_words(text: str, expected: str) -> None:
    assert bleep.mask_text(text, bleep.listed({})) == expected


def test_masked_sentence_passes_the_same_tokenizer_in_the_final_check() -> None:
    source = words("shit", "hits", "the", "fan")
    vocabulary = bleep.listed({})
    text = bleep.mask_text("shit hits the fan", vocabulary)
    assert text == "s*** hits the fan"
    original = np.full((3 * bleep.RATE, 1), .2, dtype=np.float32)
    voice, tone, _ = bleep.render(bleep.find(source, vocabulary), original, 3, len(original))
    result = bleep.check(
        source, music={}, voice=voice, original_voice=original, tone_stem=tone,
        phrases={"phrases": [{"text": text, "start": 1, "end": 2.1}]},
        title_layers=[{"id": "title", "kind": "title", "start_frame": 0,
                       "display_text": [text]}],
        timeline=ResolvedTimeline((), (), (), 90, 0),
    )
    assert result["pass"] and result["words"] == 1


def test_nested_slots_preserve_paths_enums_and_mask_defaults() -> None:
    slots = {"rows": [{"text": "Fuck", "image": "fuck.png", "mode": "shit",
                        "nested": ["bull shit", {"quote": "FUCK!"}]}],
             "images": ["shit.png"], "video": "fuck.mov"}
    rules = {"rows": {"type": "row_list", "items": {"text": {"type": "string"},
              "image": {"type": "image"}, "mode": {"type": "enum"}}},
             "images": {"type": "image_list"}, "video": {"type": "video"},
             "label": {"type": "string", "default": "Fuck"}}
    result, display = bleep.mask_slots(slots, rules, bleep.listed({}))
    assert result["rows"][0] == {"text": "F***", "image": "fuck.png", "mode": "shit",
                                  "nested": ["b*** s***", {"quote": "F***!"}]}
    assert result["images"] == ["shit.png"] and result["video"] == "fuck.mov"
    assert result["label"] == "F***"
    assert display == ["F***", "b*** s***", "F***!", "F***"]
    assert slots["rows"][0]["text"] == "Fuck"


def test_slots_keep_omitted_defaults_out_of_the_card_inputs() -> None:
    slots = {"quote": "Fuck", "name": "Speaker"}
    rules = {"quote": {"type": "string"}, "name": {"type": "string"},
             "role": {"type": "string", "default": ""},
             "label": {"type": "string", "default": "Read more"},
             "mode": {"type": "enum", "default": "plain"},
             "count": {"type": "number", "default": 3}}
    result, display = bleep.mask_slots(slots, rules, bleep.listed({}))
    assert result == {"quote": "F***", "name": "Speaker"}
    assert display == ["F***", "Speaker", "", "Read more"]
    rules["label"]["default"] = "Fuck"
    result, display = bleep.mask_slots(slots, rules, bleep.listed({}))
    assert result == {"quote": "F***", "name": "Speaker", "label": "F***"}
    assert display == ["F***", "Speaker", "", "F***"]


@pytest.mark.parametrize("at", [0, 1, 3])
def test_zero_length_words_have_padded_spans_and_pass_the_final_check(at: float) -> None:
    tokens = [CaptionWord("point", "fuck", at, at)]
    original = np.full((3 * bleep.RATE, 1), .2, dtype=np.float32)
    occurrences = bleep.find(tokens, bleep.listed({}))
    span, = bleep.spans(occurrences, original, 3)
    assert span.start <= max(0, round((at - bleep.PAD_S) * bleep.RATE))
    assert span.end >= min(len(original), round((at + bleep.PAD_S) * bleep.RATE))
    assert 0 <= span.start < span.end <= len(original)
    voice, tone, record = bleep.render(occurrences, original, 3, len(original))
    assert not voice[span.start:span.end].any()
    assert tone[span.start:span.end].any()
    assert record["spans"][0]["tone_rms_dbfs"] is not None
    options = {"music": {}, "voice": voice, "original_voice": original,
               "phrases": {"phrases": [{"text": "f***", "start": at, "end": at}]},
               "title_layers": [], "timeline": ResolvedTimeline((), (), (), 90, 0)}
    check = bleep.check(tokens, tone_stem=tone, **options)
    assert check["pass"] and check["words"] == 1
    assert not bleep.check(tokens, tone_stem=np.zeros_like(tone), **options)["pass"]


def test_quiet_threshold_and_word_union_levels() -> None:
    voice = np.full((3 * 48000, 1), .001, dtype=np.float32)
    voice[48000:57600] = .2
    occurrence, = bleep.find(words("fuck"), bleep.listed({}))
    span, = bleep.spans([occurrence], voice, 3)
    assert span.start == round(.94 * 48000) and span.end == round(1.26 * 48000)
    silent = np.zeros_like(voice)
    muted, stem, record = bleep.render([occurrence], silent, 3, len(silent))
    assert not muted.any() and not stem.any() and record["spans"][0]["tone_rms_dbfs"] is None
    voice[:] = .2
    voice[48000:57600] = 0
    _, stem, record = bleep.render([occurrence], voice, 3, len(voice))
    assert stem.any() and record["spans"][0]["tone_rms_dbfs"] is not None


def test_split_detection_crosses_a_gap_and_does_not_use_punctuation_as_a_word() -> None:
    tokens = [CaptionWord("left", "mother", .1, .3), CaptionWord("right", "fucker", 1, 1.3)]
    found, = bleep.find(tokens, bleep.listed({}))
    assert found.ids == ("left", "right") and (found.start_s, found.end_s) == (.1, 1.3)
    found, = bleep.find(words("...", "fuck"), bleep.listed({}))
    assert found.indices == (1,)


def test_spans_pad_extend_merge_and_clip() -> None:
    voice = np.zeros((3 * bleep.RATE, 1), dtype=np.float32)
    voice[round(.9 * bleep.RATE):round(1.28 * bleep.RATE)] = .2
    occurrence, = bleep.find(words("fuck"), bleep.listed({}))
    span, = bleep.spans([occurrence], voice, 3)
    assert .895 <= span.start / bleep.RATE <= .9
    assert 1.28 <= span.end / bleep.RATE <= 1.285
    loud = np.full_like(voice, .2)
    span, = bleep.spans([occurrence], loud, 3)
    assert span.start == round((1 - bleep.PAD_S - bleep.EXTEND_MAX_S) * bleep.RATE)
    assert span.end == round((1.2 + bleep.PAD_S + bleep.EXTEND_MAX_S) * bleep.RATE)
    edge = [CaptionWord("a", "fuck", .01, .1), CaptionWord("b", "shit", .15, .3)]
    span, = bleep.spans(bleep.find(edge, bleep.listed({})), loud, .32)
    assert span.start == 0 and span.end == round(.32 * bleep.RATE)
    assert len(span.occurrences) == 2 and span.masked == ("f***", "s***")


@pytest.mark.parametrize("detect", [bleep.find, bleep.find_display])
@pytest.mark.parametrize(("previous_end", "next_start", "start", "end"), [
    (1, 1.2, .94, 1.26), (.88, 1.32, .88, 1.32),
])
def test_span_extension_stops_at_neighbour_timestamps(detect, previous_end, next_start,
                                                     start, end) -> None:
    source = [CaptionWord("before", "before", .2, previous_end),
              CaptionWord("listed", "shit", 1, 1.2),
              CaptionWord("after", "after", next_start, 1.8)]
    voice = np.full((2 * bleep.RATE, 1), .2, dtype=np.float32)
    span, = bleep.spans(detect(source, bleep.listed({})), voice, 2)
    assert (span.start, span.end) == (round(start * bleep.RATE), round(end * bleep.RATE))


def test_span_extension_covers_speech_into_a_quiet_gap() -> None:
    source = [CaptionWord("before", "before", .2, .6),
              CaptionWord("listed", "shit", 1, 1.2),
              CaptionWord("after", "after", 1.6, 1.8)]
    original = np.zeros((2 * bleep.RATE, 1), dtype=np.float32)
    original[round(.9 * bleep.RATE):round(1.28 * bleep.RATE)] = .2
    found = bleep.find_display(source, bleep.listed({}))
    span, = bleep.spans(found, original, 2)
    assert .895 <= span.start / bleep.RATE <= .9
    assert 1.28 <= span.end / bleep.RATE <= 1.285
    muted, _, _ = bleep.render(found, original, 2, len(original))
    assert not muted[round(.9 * bleep.RATE):round(1.28 * bleep.RATE)].any()


@pytest.mark.parametrize("slots", [{}, {"clip": None}, {"clip": ""}])
def test_file_card_check_skips_an_absent_fallback_path(slots) -> None:
    audio = np.zeros((bleep.RATE, 1), dtype=np.float32)
    timeline = ResolvedTimeline((ResolvedElement("agent-file", "graphic", "agent", 0, 30,
                                                "front", {"template": "file_drop", "slots": slots}),
                                ), (), (), 30, 0)
    check = bleep.check([], music={}, voice=audio, original_voice=audio, tone_stem=None,
                        phrases={}, title_layers=[], timeline=timeline)
    assert check["pass"]


def test_render_keeps_continuous_neighbours_outside_the_pad(tmp_path: Path) -> None:
    original = np.full((2 * bleep.RATE, 1), .2, dtype=np.float32)
    synth.float_audio(tmp_path / "cut/voice.wav", original)
    source = [CaptionWord("before", "before", .2, 1), CaptionWord("listed", "shit", 1, 1.2),
              CaptionWord("after", "after", 1.2, 1.8)]
    result = render_audio(tmp_path, tmp_path / "out.m4a", work_dir=tmp_path / "mix",
                          timeline=ResolvedTimeline((), (), (), 60, 0),
                          policy={"mode": "preserve"}, music={},
                          bleeps=bleep.find_display(source, bleep.listed({})))
    span, = result.bleep["spans"]
    assert (span["start_s"], span["end_s"]) == (.94, 1.26)
    voice = np.fromfile(result.bleep_voice, dtype="<f4")
    np.testing.assert_array_equal(voice[:round(.9 * bleep.RATE)],
                                  original[:round(.9 * bleep.RATE), 0])
    np.testing.assert_array_equal(voice[round(1.3 * bleep.RATE):],
                                  original[round(1.3 * bleep.RATE):, 0])


def test_tone_frequency_fades_and_bytes() -> None:
    tone = bleep.tone(48000)
    assert tone.dtype == np.float32 and tone.tobytes() == bleep.tone(48000).tobytes()
    assert tone[0] == tone[-1] == 0 and np.max(np.abs(tone)) == 1
    assert np.argmax(np.abs(np.fft.rfft(tone))) == 1000
    assert np.max(np.abs(tone[:24])) < .03
    assert np.max(np.abs(tone[-24:])) < .03
    assert len(bleep.tone(0)) == 0 and np.isfinite(bleep.tone(3)).all()


def audio_project(root: Path, *, stereo: bool = True) -> tuple[np.ndarray, ResolvedTimeline]:
    time = np.arange(3 * 48000) / 48000
    voice = (.25 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    envelope = np.zeros(len(time), dtype=np.float32)
    envelope[4800:24000] = 1
    envelope[round(.95 * 48000):round(1.28 * 48000)] = 1
    envelope[round(2 * 48000):round(2.4 * 48000)] = 1
    voice *= envelope
    pcm = np.column_stack((voice, voice)) if stereo else voice[:, None]
    synth.float_audio(root / "cut/voice.wav", pcm)
    return pcm, ResolvedTimeline((), (), (), 90, 15)


@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize(("token", "masked"), [("fuck", "f***"), ("shit-show", "s***-show")])
def test_render_mutes_coverage_and_matches_tone_level(tmp_path: Path, stereo: bool,
                                                    token: str, masked: str) -> None:
    voice, timeline = audio_project(tmp_path, stereo=stereo)
    occurrences = bleep.find(words(token), bleep.listed({}))
    work = tmp_path / "work"
    result = render_audio(tmp_path, tmp_path / "out.m4a", work_dir=work, timeline=timeline,
                          policy={"mode": "preserve"}, music={}, bleeps=occurrences)
    mix = decode(work / "mix.wav", [], channels=2)
    tone = np.fromfile(work / "bleep.f32le", dtype="<f4").reshape(-1, 2)
    muted = np.fromfile(result.bleep_voice, dtype="<f4").reshape(-1, voice.shape[1])
    span, = result.bleep["spans"]
    start, end = (round(span[key] * 48000) for key in ("start_s", "end_s"))
    assert not (mix[start:end] - tone[start:end]).any()
    assert not muted[round(.94 * 48000):round(1.28 * 48000)].any()
    assert np.max(np.abs(voice[48000:57600])) > .2
    np.testing.assert_array_equal(mix[4800:24000, 0], voice[4800:24000, 0])
    word_rms = np.sqrt(np.mean(voice[48000:57600].astype(np.float64) ** 2))
    tone_rms = np.sqrt(np.mean(tone[start:end].astype(np.float64) ** 2))
    assert abs(20 * np.log10(tone_rms / word_rms)) < .5
    assert np.max(np.abs(tone)) <= np.max(np.abs(voice))
    assert not mix[3 * 48000:].any() and not tone[3 * 48000:].any()
    assert result.words_stem is not None
    clean = decode(result.words_stem, [], channels=2)
    np.testing.assert_array_equal(clean[:len(voice), 0], voice[:, 0])
    assert "fuck" not in json.dumps(result.to_dict()["bleep"])
    assert result.bleep["count"] == 1
    assert result.bleep["spans"][0]["masked"] == [masked]


@pytest.mark.parametrize("mode", ["preserve", "loudnorm"])
def test_music_key_and_clean_stems(tmp_path: Path, mode: str) -> None:
    _, timeline = audio_project(tmp_path)
    music = resolve_style_data({"schema_version": 1})["music"]
    music["dry_open_s"] = 0
    time = np.arange(3 * 48000 + 15 * 1600) / 48000
    bed = np.column_stack([.03 * np.sin(2 * np.pi * 880 * time)] * 2).astype(np.float32)
    bed[3 * 48000:] = 0
    track = synth.float_audio(tmp_path / "bed.wav", bed)
    options = {"timeline": timeline, "policy": AUDIO_DEFAULTS if mode == "loudnorm"
               else {"mode": "preserve"}, "music": music, "bed": track}
    result = render_audio(tmp_path, tmp_path / "on.m4a", work_dir=tmp_path / "on",
                          bleeps=bleep.find(words("fuck"), bleep.listed({})), **options)
    render_audio(tmp_path, tmp_path / "off.m4a", work_dir=tmp_path / "off",
                 **{**options, "policy": {"mode": "preserve"}})
    reference = tmp_path / "off/mix.wav"
    if mode == "loudnorm":
        from yap_studio import ffmpeg

        command = next(cmd for cmd in result.commands if cmd[-1].endswith(
            f"normalised-{result.normalization['iterations'] - 1}.wav"))
        expected = tmp_path / "expected.wav"
        ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(reference), "-af",
                    command[command.index("-af") + 1], "-c:a", "pcm_f32le", str(expected)],
                   timeout_s=60)
        reference = expected
    np.testing.assert_allclose(decode(result.words_stem, [], channels=2),
                               decode(reference, [], channels=2), rtol=0, atol=.001)
    assert result.click_stem is not None
    assert np.max(np.abs(decode(result.click_stem, [], channels=2)[48000:57600])) > .1


def test_words_stem_keeps_inserts_and_leaves_out_both_effects(tmp_path: Path) -> None:
    audio_project(tmp_path)
    timeline = ResolvedTimeline((ResolvedElement("agent-chapter", "chapter", "agent", 30, 60,
                                                 "front", {}),), (), (), 90, 15)
    time = np.arange(48000) / 48000
    clip = synth.float_audio(tmp_path / "clip.wav", .05 * np.sin(2 * np.pi * 770 * time))
    options = {"timeline": timeline, "policy": {"mode": "preserve"}, "music": {},
               "clips": [{"id": "insert", "path": clip.name, "start_frame": 15, "frames": 15,
                          "gain_db": -6}]}
    result = render_audio(tmp_path, tmp_path / "on.m4a", work_dir=tmp_path / "on",
                          bleeps=bleep.find(words("fuck"), bleep.listed({})),
                          sfx={**made_sfx.listing(), "ready": True}, **options)
    render_audio(tmp_path, tmp_path / "off.m4a", work_dir=tmp_path / "off", **options)
    assert result.sfx["cues"] and result.words_stem is not None
    np.testing.assert_array_equal(decode(result.words_stem, [], channels=2),
                                  decode(tmp_path / "off/mix.wav", [], channels=2))


def test_defaults_and_empty_path_are_identical(tmp_path: Path) -> None:
    music = resolve_style_data({"schema_version": 1})["music"]
    assert music["bleep_swears"] is False
    assert music["bleep_words_add"] == music["bleep_words_remove"] == []
    assert not bleep.enabled({}) and bleep.enabled({"music": {"bleep_swears": True}})
    _, timeline = audio_project(tmp_path)
    for name, kwargs in [("baseline", {}), ("none", {"bleeps": None}),
                         ("empty", {"bleeps": bleep.find(words("safe"), bleep.listed({}))})]:
        work = tmp_path / name
        result = render_audio(tmp_path, tmp_path / f"{name}.m4a", work_dir=work,
                              timeline=timeline, policy={"mode": "preserve"}, music=music,
                              **kwargs)
        assert result.bleep is None and "bleep" not in result.to_dict()
        assert not list(work.glob("bleep*"))
        assert (tmp_path / f"{name}.m4a").read_bytes() == (tmp_path / "baseline.m4a").read_bytes()


def test_pilot_applies_bleeps_without_verification_stems(tmp_path: Path) -> None:
    _, timeline = audio_project(tmp_path)
    work = tmp_path / "work"
    result = render_audio(tmp_path, tmp_path / "pilot.m4a", work_dir=work, timeline=timeline,
                          policy={"mode": "preserve"}, music={}, frames=(0, 59),
                          bleeps=bleep.find(words("fuck"), bleep.listed({})))
    assert result.samples == 60 * 1600 and result.bleep["count"] == 1
    assert result.words_stem is result.click_stem is None
    assert not list(work.glob("*stem*"))
    voice = np.fromfile(result.bleep_voice, dtype="<f4").reshape(-1, 2)
    assert not voice[round(.94 * 48000):round(1.28 * 48000)].any()


def test_voice_treatment_and_end_hold(tmp_path: Path) -> None:
    voice, timeline = audio_project(tmp_path, stereo=False)
    voice[-round(.2 * 48000):] = .2
    synth.float_audio(tmp_path / "cut/voice.wav", voice)
    work = tmp_path / "work"
    occurrence = CaptionWord("last", "shit", 2.85, 3)
    result = render_audio(tmp_path, tmp_path / "out.m4a", work_dir=work, timeline=timeline,
                          policy=AUDIO_DEFAULTS, music={}, voice={"preset": "clean"},
                          bleeps=bleep.find([occurrence], bleep.listed({})))
    assert result.bleep["spans"][0]["end_s"] == 3
    stem = np.fromfile(work / "bleep.f32le", dtype="<f4").reshape(-1, 2)
    assert not stem[3 * 48000:].any()
    clean = decode(result.words_stem, [], channels=2)
    assert clean.shape == (3 * 48000 + 15 * 1600, 2) and not clean[3 * 48000:].any()
    assert result.normalization and result.voice["filters"]


@pytest.mark.parametrize(("field", "value"), [
    ("bleep_swears", "true"), ("bleep_words_add", ["LOUD"]),
    ("bleep_words_add", ["x"]), ("bleep_words_remove", ["x"]),
    ("bleep_words_add", ["two words"]), ("bleep_words_add", ["x", "x"]),
    ("bleep_words_remove", ["a" * 41]), ("bleep_words_remove", ["a"] * 101),
    ("bleep_words_remove", ["2word"]), ("bleep_words_remove", "word"),
])
def test_schema_refuses_invalid_bleep_choices(field: str, value: object) -> None:
    with pytest.raises(StyleError, match=field):
        resolve_style_data({"schema_version": 1, "music": {field: value}})


@pytest.mark.parametrize("kind", ["insert", "overlay", "title"])
def test_check_covers_drawn_letters_and_missing_records(kind: str) -> None:
    audio = np.zeros((bleep.RATE, 1), dtype=np.float32)
    payload = {"lines": ["Fuck"]} if kind == "title" else {"letters": {"text": "Fuck"}}
    element = ResolvedElement("letters", kind, "agent", 0, 30, "front", payload)
    options = dict(music={}, voice=audio, original_voice=audio, tone_stem=None, phrases={},
                   timeline=ResolvedTimeline((element,), (), (), 30, 0))
    drawn = {"id": element.id, "kind": kind, "start_frame": 0, "display_text": ["F***"]}
    assert bleep.check([], title_layers=[drawn], **options)["pass"]
    exposed = {**drawn, "display_text": ["FUCK"]}
    assert not bleep.check([], title_layers=[exposed], **options)["pass"]
    result = bleep.check([], title_layers=[], **options)
    assert not result["pass"] and result["unmasked"][0]["where"] == f"{kind} letters"
    assert result["unmasked"][0]["masked"] == "F***"


@pytest.mark.parametrize("miss", [None, "voice", "tone", "caption", "card", "record"])
def test_check_catches_audio_and_display_misses(miss: str | None) -> None:
    original = np.full((3 * 48000, 1), .2, dtype=np.float32)
    voice = np.zeros_like(original)
    tone = np.ones((3 * 48000, 2), dtype=np.float32)
    phrases = {"phrases": [{"text": "f***", "start": 1, "end": 1.2}]}
    layers = [{"id": "agent-card", "kind": "graphic", "start_frame": 30,
               "display_text": ["f***"]}]
    timeline = ResolvedTimeline((), (), (), 90, 0)
    if miss == "voice":
        voice[round(.95 * 48000)] = .1
    if miss == "tone":
        tone[:] = 0
    if miss == "caption":
        phrases["phrases"][0]["text"] = "fuck"
    if miss == "card":
        layers[0]["display_text"] = ["fuck"]
    if miss == "record":
        timeline = ResolvedTimeline((ResolvedElement("agent-title", "title", "agent", 0, 30,
                                    "front", {"lines": ["fuck"]}),), (), (), 90, 0)
    check = bleep.check(words("fuck"), music={}, voice=voice, original_voice=original,
                        tone_stem=tone, phrases=phrases, title_layers=layers, timeline=timeline)
    assert check["id"] == "bleep.words" and check["words"] == 1
    assert check["pass"] is (miss is None)
    assert "fuck" not in json.dumps(check)
    if miss in {"voice", "tone"}:
        assert check["audible"][0]["time_s"] == 1
    if miss in {"caption", "card", "record"}:
        assert check["unmasked"]
    if miss is not None:
        from yap_studio.project.receipts import failed_assert_lines

        message, = failed_assert_lines([check])
        assert message.startswith("Bleep check failed: a swear word at ")
        assert "fuck" not in message and check["fix"]


def test_check_requires_tone_over_both_tokens_and_detects_split_caption() -> None:
    tokens = words("mother", "fucker")
    original = np.full((3 * 48000, 1), .2, dtype=np.float32)
    tone = np.ones_like(original)
    tone[round(tokens[1].start * 48000):round(tokens[1].end * 48000)] = 0
    phrases = {"phrases": [{"text": token.text, "start": token.start, "end": token.end}
                           for token in tokens]}
    check = bleep.check(tokens, music={}, voice=np.zeros_like(original), original_voice=original,
                        tone_stem=tone, phrases=phrases, title_layers=[],
                        timeline=ResolvedTimeline((), (), (), 90, 0))
    assert not check["pass"] and len(check["audible"]) == len(check["unmasked"]) == 1
    assert check["unmasked"][0]["masked"] == ["m*****", "f*****"]
