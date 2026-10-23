"""Caption CLI integration with real media and a deterministic alpha renderer."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import synth
from test_captions_render import _validate_variables
from test_taste_render import _body

from yap_studio import cli, ffmpeg, paths, receipt, text_rules, transcript
from yap_studio.brand_words import set_words
from yap_studio.captions import CaptionWord, build_phrases
from yap_studio.errors import InputError, ProcessingError, RendererBlockedError, YapError
from yap_studio.render import hyperframes
from yap_studio.sfx import bleep
from yap_studio.stages import captions
from yap_studio.stages.assemble_audio import decode
from yap_studio.style import resolve
from yap_studio.taste import add_proposals, approve, read_taste_record, write_taste_record
from yap_studio.taste import render as taste_render
from yap_studio.transcript import Transcript, Word


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    """A short vertical source with audio at a different source frame rate."""
    return synth.calib_source(tmp_path / "clip.mov", seconds=1.25, fps="24/1", size="108x192")


@pytest.fixture
def words(tmp_path: Path) -> Path:
    result = tmp_path / "words.json"
    Transcript(
        [
            Word("First", 0.1, 0.25),
            Word("words.", 0.25, 0.45),
            Word("Next", 0.7, 0.85),
            Word("phrase.", 0.85, 1.1),
        ]
    ).save(result)
    return result


@pytest.fixture(autouse=True)
def fake_layers(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Encode actual alpha layers so assembly and receipts exercise real ffmpeg."""
    calls: list[dict] = []

    def render(
        composition_dir: Path,
        out: Path,
        *,
        frames: int,
        variables: Mapping[str, object],
        fps: int = 30,
        install: hyperframes.RendererInstall | None = None,
        workers: int | str = hyperframes.RENDER_WORKERS,
        browser_gpu: bool = True,
    ) -> hyperframes.LayerResult:
        source = (composition_dir / "index.html").read_text()
        _validate_variables(source, variables)
        parser = captions._RootParser()
        parser.feed(source)
        assert parser.root is not None
        width = int(re.search(r'data-width="(\d+)"', parser.root)[1])
        height = int(re.search(r'data-height="(\d+)"', parser.root)[1])
        calls.append({"variables": dict(variables), "width": width, "height": height,
                      "workers": workers, "browser_gpu": browser_gpu})
        if "phrases" in variables:
            phrases = json.loads(str(variables["phrases"]))
            windows = "+".join(f"gte(T,{p['start']})*lt(T,{p['end']})" for p in phrases) or "0"
        else:
            windows = "1"
        with paths.atomic_output(out) as temporary:
            ffmpeg.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    f"color=c=black:s={width}x{height}:r={fps},format=rgba",
                    "-map",
                    "0:v:0",
                    "-vf",
                    f"geq=r=230:g=90:b=40:a='if({windows},255,0)',"
                    "scale=out_color_matrix=bt709:out_range=tv,format=yuva444p10le,"
                    "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
                    "-frames:v",
                    str(frames),
                    "-an",
                    "-c:v",
                    "prores_ks",
                    "-profile:v",
                    "4444",
                    "-pix_fmt",
                    "yuva444p10le",
                    "-colorspace",
                    "bt709",
                    "-color_primaries",
                    "bt709",
                    "-color_trc",
                    "bt709",
                    "-color_range",
                    "tv",
                    "-threads",
                    "1",
                    str(temporary),
                ],
                timeout_s=120 + frames,
            )
        return hyperframes.LayerResult(out, frames, width, height, frames / fps)

    monkeypatch.setattr(hyperframes, "render_layer", render)
    return calls


def _args(clip: Path, words: Path, out: Path) -> list[str]:
    return [
        "captions",
        str(clip),
        "--words",
        str(words),
        "--out",
        str(out),
        "--style",
        "preset:electric",
        "--srt",
        "--json",
    ]


@pytest.mark.parametrize("on", [False, True])
def test_caption_clip_bleeps_saved_style_audio_and_all_display_text(
    clip: Path, words: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    fake_layers: list[dict], on: bool,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_KEEP_WORK", "1")
    Transcript([Word("shit-show", .1, .3), Word("fuck-you", .4, .6),
                Word("today", .8, 1.0)]).save(words)
    style_path = tmp_path / "style.json"
    style_data = {"schema_version": 1, "name": "Clip bleep", "music": {"bleep_swears": on},
                  "hook": {"style": "quiet_serif"}}
    style_path.write_text(json.dumps(style_data))
    style = resolve(str(style_path))
    out = tmp_path / "captioned.mp4"
    captions.caption_clip(clip, style=style, out=out, words=words,
                          title="Shit-show|Fuck-you", srt=True, force=False)
    variables = fake_layers[0]["variables"]
    assert (variables["plain"], variables["key"]) == (
        ("S***-show", "F***-you") if on else ("Shit-show", "Fuck-you"))
    shown = [word["text"] for phrase in json.loads(fake_layers[1]["variables"]["phrases"])
             for word in phrase["words"]]
    assert shown[:2] == (["s***-show", "f***-you"] if on else ["shit-show", "fuck-you"])
    subtitles = out.with_suffix(".srt").read_text()
    assert ("s***-show" if on else "shit-show") in subtitles
    record = json.loads(out.with_suffix(".receipt.json").read_text())
    work, = tmp_path.glob(".yap-captions-*")
    if on:
        treatment = record["bleep"]
        assert treatment["count"] == 2 and record["asserts"][0]["pass"]
        assert "shit" not in json.dumps(treatment) and "fuck" not in json.dumps(treatment)
        original = decode(work / "conformed.mp4", [], channels=2)
        muted = np.fromfile(work / "bleep-voice.f32le", dtype="<f4").reshape(-1, 2)
        tone = np.fromfile(work / "bleep.f32le", dtype="<f4").reshape(-1, 2)
        mixed = np.fromfile(work / "bleep-mix.f32le", dtype="<f4").reshape(-1, 2)
        for span in treatment["spans"]:
            first, last = (round(span[key] * bleep.RATE) for key in ("start_s", "end_s"))
            assert original[first:last].any() and not muted[first:last].any()
            np.testing.assert_array_equal(mixed[first:last], tone[first:last])
        delivered = decode(out, [], channels=2)[round(.12 * bleep.RATE):round(.28 * bleep.RATE), 0]
        peak = np.argmax(np.abs(np.fft.rfft(delivered))) * bleep.RATE / len(delivered)
        assert abs(peak - bleep.TONE_HZ) < 10
    else:
        assert "bleep" not in record and "asserts" not in record
        assert not list(work.glob("bleep*"))
        before = [dict(call["variables"]) for call in fake_layers]
        style_data["music"].pop("bleep_swears")
        style_path.write_text(json.dumps(style_data))
        baseline = tmp_path / "baseline.mp4"
        captions.caption_clip(clip, style=resolve(str(style_path)), out=baseline, words=words,
                              title="Shit-show|Fuck-you", srt=True, force=False)
        assert out.read_bytes() == baseline.read_bytes()
        assert out.with_suffix(".srt").read_bytes() == baseline.with_suffix(".srt").read_bytes()
        assert before == [dict(call["variables"]) for call in fake_layers[2:]]


@pytest.mark.parametrize("unmuted", [False, True])
def test_caption_clip_detects_custom_words_before_number_formatting(
    clip: Path, words: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    fake_layers: list[dict], unmuted: bool,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_KEEP_WORK", "1")
    Transcript([Word("twenty", .1, .3), Word("five", .4, .6)]).save(words)
    style = resolve("preset:clean_minimal")
    style.data["music"].update(bleep_swears=True, bleep_words_add=["twenty"])
    out = tmp_path / "number.mp4"
    if unmuted:
        render = bleep.render

        def leak(occurrences, voice, cut_s, samples):
            _, tone, record = render(occurrences, voice, cut_s, samples)
            return voice.copy(), tone, record

        monkeypatch.setattr(bleep, "render", leak)
        with pytest.raises(YapError, match="Bleep check failed"):
            captions.caption_clip(clip, style=style, out=out, words=words,
                                  title=None, srt=True, force=False)
        assert not out.exists() and not out.with_suffix(".receipt.json").exists()
        return
    captions.caption_clip(clip, style=style, out=out, words=words,
                          title=None, srt=True, force=False)
    shown = [word["text"] for phrase in json.loads(fake_layers[0]["variables"]["phrases"])
             for word in phrase["words"]]
    assert shown == ["25"] and "25" in out.with_suffix(".srt").read_text()
    record = json.loads(out.with_suffix(".receipt.json").read_text())
    assert record["bleep"]["count"] == 1
    assert record["asserts"][0]["pass"] and record["asserts"][0]["words"] == 1
    assert record["bleep"]["spans"][0]["masked"] == ["t*****"]
    work, = tmp_path.glob(".yap-captions-*")
    original = decode(work / "conformed.mp4", [], channels=2)
    muted = np.fromfile(work / "bleep-voice.f32le", dtype="<f4").reshape(-1, 2)
    tone = np.fromfile(work / "bleep.f32le", dtype="<f4").reshape(-1, 2)
    first, last = round(.04 * bleep.RATE), round(.36 * bleep.RATE)
    assert original[first:last].any() and not muted[first:last].any()
    assert tone[round(.1 * bleep.RATE):round(.3 * bleep.RATE)].any()


def test_caption_clip_accepts_a_mask_beside_an_ordinary_word(
    clip: Path, words: Path, tmp_path: Path, fake_layers: list[dict],
) -> None:
    Transcript([Word("shit", .1, .2), Word("hits", .25, .4), Word("the", .5, .65),
                Word("fan", .8, 1.0)]).save(words)
    style = resolve("preset:clean_minimal")
    style.data["music"]["bleep_swears"] = True
    out = tmp_path / "sentence.mp4"
    captions.caption_clip(clip, style=style, out=out, words=words,
                          title=None, srt=True, force=False)
    shown = [word["text"] for phrase in json.loads(fake_layers[0]["variables"]["phrases"])
             for word in phrase["words"]]
    assert shown == ["s***", "hits", "the", "fan"]
    assert "s*** hits" in out.with_suffix(".srt").read_text()
    record = json.loads(out.with_suffix(".receipt.json").read_text())
    assert record["asserts"][0]["pass"] and record["bleep"]["count"] == 1


def test_caption_clip_bleeps_kinetic_title_after_matching_original_words(
    clip: Path, words: Path, tmp_path: Path, fake_layers: list[dict],
) -> None:
    Transcript([Word("shit", .1, .3)]).save(words)
    style = resolve("preset:clean_minimal")
    style.data["music"]["bleep_swears"] = True
    style.data["hook"]["style"] = "kinetic"
    captions.caption_clip(clip, style=style, out=tmp_path / "kinetic.mp4", words=words,
                          title="Shit", srt=False, force=False)
    assert fake_layers[0]["variables"]["plain"] == "S***"
    assert json.loads(fake_layers[0]["variables"]["word_times"]) == [.1]


def test_caption_clip_refuses_a_failed_bleep_check_before_publication(
    clip: Path, words: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    Transcript([Word("shit-show", .1, .3)]).save(words)
    style = resolve("preset:electric")
    style.data["music"]["bleep_swears"] = True
    render = bleep.render

    def unmuted(occurrences, voice, cut_s, samples):
        _, tone, record = render(occurrences, voice, cut_s, samples)
        return voice.copy(), tone, record

    monkeypatch.setattr(bleep, "render", unmuted)
    out = tmp_path / "refused.mp4"
    with pytest.raises(YapError, match="Bleep check failed") as caught:
        captions.caption_clip(clip, style=style, out=out, words=words,
                              title=None, srt=True, force=False)
    assert "caption export" in caught.value.suggested_action
    assert not any(target.exists() for target in (out, out.with_suffix(".srt"),
                                                  out.with_suffix(".receipt.json")))
    assert not list(tmp_path.glob(".yap-captions-*"))


def test_caption_clip_masks_a_title_without_changing_unlisted_audio(
    clip: Path, words: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    fake_layers: list[dict],
) -> None:
    monkeypatch.setenv("YAP_STUDIO_KEEP_WORK", "1")
    Transcript([Word("today", .1, .3)]).save(words)
    style = resolve("preset:clean_minimal")
    style.data["music"]["bleep_swears"] = True
    out = tmp_path / "title-only.mp4"
    captions.caption_clip(clip, style=style, out=out, words=words,
                          title="god|damn", srt=False, force=False)
    assert (fake_layers[0]["variables"]["plain"], fake_layers[0]["variables"]["key"]) == \
        ("g**", "d***")
    record = json.loads(out.with_suffix(".receipt.json").read_text())
    assert record["bleep"]["count"] == 0 and record["bleep"]["spans"] == []
    assert record["asserts"][0]["pass"]
    work, = tmp_path.glob(".yap-captions-*")
    assert not list(work.glob("bleep*"))
    np.testing.assert_array_equal(decode(out, [], channels=2),
                                  decode(work / "conformed.mp4", [], channels=2))


def test_caption_clip_bleep_accepts_a_clip_without_an_audio_stream(
    clip: Path, words: Path, tmp_path: Path,
) -> None:
    silent = tmp_path / "silent.mov"
    ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(clip), "-map", "0:v:0", "-c:v", "copy",
                "-an", str(silent)], timeout_s=30)
    Transcript([Word("shit", .1, .3)]).save(words)
    style = resolve("preset:clean_minimal")
    style.data["music"]["bleep_swears"] = True
    out = tmp_path / "silent-captioned.mp4"
    captions.caption_clip(silent, style=style, out=out, words=words,
                          title=None, srt=True, force=False)
    assert not ffmpeg.probe(out).has_audio
    assert "s***" in out.with_suffix(".srt").read_text()
    record = json.loads(out.with_suffix(".receipt.json").read_text())
    assert record["asserts"][0]["pass"]
    assert record["bleep"]["count"] == 1
    assert record["bleep"]["spans"][0]["tone_rms_dbfs"] is None


def test_caption_clip_applies_title_rules_and_reports_brand_spelling(
    clip: Path, words: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    fake_layers: list[dict],
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    set_words(["SparrowLight"])
    Transcript([Word("Sparrowlite", 0.1, 0.3), Word("twenty", 0.4, 0.6),
                Word("years", 0.7, 0.9)]).save(words)
    out = tmp_path / "captioned.mp4"
    result = captions.caption_clip(clip, style=resolve("preset:electric"), out=out,
                                   words=words, title="sparrowlight|five dollars",
                                   srt=False, force=False)
    assert (fake_layers[0]["variables"]["plain"], fake_layers[0]["variables"]["key"]) == \
        ("SparrowLight", "$5")
    rendered_words = json.loads(fake_layers[1]["variables"]["phrases"])[0]["words"]
    assert [word["text"] for word in rendered_words][-2:] == ["20", "years"]
    assert result["caption_spelling"]["near"][0]["text"] == "Sparrowlite"
    saved = json.loads(Path(result["receipt"]).read_text())
    assert saved["caption_spelling"] == result["caption_spelling"]


@pytest.mark.parametrize("title,spoken,expected,shown", [
    ("First|words.", [("First", .1, .25), ("words.", .25, .45)],
     [.1, 8 / 30], ("First", "words.")),
    ("brightloom|twenty five", [("brightloom", .13, .3), ("twenty", .4, .55),
                                ("five", .6, .8)],
     [4 / 30, .4], ("BrightLoom", "25")),
])
def test_kinetic_clip_title_follows_spoken_frames_and_display_rules(
    clip: Path, words: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    fake_layers: list[dict], title, spoken, expected, shown,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    set_words(["BrightLoom"])
    Transcript([Word(*word) for word in spoken]).save(words)
    style = resolve("preset:clean_minimal")
    style.data["hook"]["style"] = "kinetic"
    result = captions.caption_clip(clip, style=style, out=tmp_path / "timed.mp4", words=words,
                                   title=title, srt=False, force=False)
    variables = fake_layers[0]["variables"]
    assert json.loads(variables["word_times"]) == expected
    assert (variables["plain"], variables["key"]) == shown
    assert not result.get("warnings")
    assert ffmpeg.probe(Path(result["out"])).video_frames == result["frames"]


def test_kinetic_clip_title_rules_use_vocabulary(
    clip: Path, words: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    fake_layers: list[dict],
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    set_words(["SandPiper"])

    def aliases(vocabulary):
        assert tuple(vocabulary) == ("SandPiper",)
        return {"sand piper": "SandPiper"}

    monkeypatch.setattr(text_rules, "vocabulary_fixes", aliases)
    Transcript([Word("sand", 0.1, 0.2), Word("piper", 0.4, 0.5)]).save(words)
    style = resolve("preset:clean_minimal")
    style.data["hook"].update(style="kinetic", duration_s=2)
    result = captions.caption_clip(clip, style=style, out=tmp_path / "timed.mp4", words=words,
                                   title="SandPiper", srt=False, force=False)
    assert json.loads(fake_layers[0]["variables"]["word_times"]) == [0.1]
    assert not result.get("warnings")


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("title,duration,unmatched", [
    ("Unspoken|title", 1.5, "Unspoken"), ("First|Next", .7, "Next"),
])
def test_kinetic_clip_title_fallback_warns_in_result_and_receipt(
    clip: Path, words: Path, tmp_path: Path, fake_layers: list[dict],
    capsys: pytest.CaptureFixture[str], as_json, title, duration, unmatched,
) -> None:
    style = tmp_path / "kinetic.json"
    style.write_text(json.dumps({"schema_version": 1, "hook": {
        "style": "kinetic", "duration_s": duration,
    }}))
    out = tmp_path / "fallback.mp4"
    args = ["captions", str(clip), "--words", str(words), "--out", str(out),
            "--style", str(style), "--title", title]
    assert cli.main(args + (["--json"] if as_json else [])) == 0
    output = capsys.readouterr().out
    assert json.loads(fake_layers[0]["variables"]["word_times"]) == []
    record = json.loads(out.with_suffix(".receipt.json").read_text())
    warnings = record["warnings"]
    assert any("even spacing" in warning and unmatched in warning for warning in warnings)
    if as_json:
        assert json.loads(output)["warnings"] == warnings
    else:
        assert all(f"warning: {warning}" in output for warning in warnings)


def test_non_kinetic_clip_title_keeps_its_plan(
    clip: Path, words: Path, tmp_path: Path, fake_layers: list[dict],
) -> None:
    style = resolve("preset:clean_minimal")
    expected = captions.title_layer_plan("Arbitrary", "title", style)
    result = captions.caption_clip(clip, style=style, out=tmp_path / "ordinary.mp4", words=words,
                                   title="Arbitrary|title", srt=False, force=False)
    assert fake_layers[0]["variables"] == expected.variables
    assert (fake_layers[0]["width"], fake_layers[0]["height"]) == (expected.width, expected.height)
    assert not result.get("warnings")


@pytest.mark.parametrize("template", ["captions", "title"])
@pytest.mark.parametrize(
    "problem",
    ["strings", "missing_value", "missing_label", "numeric_value", "numeric_label", "unknown"],
)
def test_fake_renderer_rejects_invalid_enums(tmp_path: Path, template: str, problem: str) -> None:
    template_dir = Path(__file__).parents[1] / "src" / "yap_studio" / "templates" / template
    declarations = json.loads((template_dir / "template.json").read_text())["variables"]
    variables = {declaration["id"]: declaration["default"] for declaration in declarations}
    enum = next(declaration for declaration in declarations if declaration["type"] == "enum")
    if problem == "strings":
        enum["options"] = [option["value"] for option in enum["options"]]
    elif problem.startswith("missing_"):
        del enum["options"][0][problem.removeprefix("missing_")]
    elif problem.startswith("numeric_"):
        enum["options"][0][problem.removeprefix("numeric_")] = 1
    else:
        variables[enum["id"]] = "unknown"
    source = (template_dir / "template.html").read_text()
    source = re.sub(
        r"data-composition-variables='[^']*'",
        lambda match: "data-composition-variables='" + json.dumps(declarations) + "'",
        source,
        count=1,
    )
    (tmp_path / "index.html").write_text(source)
    with pytest.raises(ProcessingError, match="Variable validation failed") as caught:
        hyperframes.render_layer(tmp_path, tmp_path / "layer.mov", frames=1, variables=variables)
    assert caught.value.returncode == 1
    assert not (tmp_path / "layer.mov").exists()


def test_composite_copies_encoded_aac_packets(
    clip: Path,
    words: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    composite = captions._composite
    verified = []

    def audio_packets(media: Path) -> bytes:
        return ffmpeg.run(
            [
                "ffmpeg",
                "-nostdin",
                "-i",
                str(media),
                "-map",
                "0:a:0",
                "-vn",
                "-c:a",
                "copy",
                "-f",
                "adts",
                "pipe:1",
            ],
            timeout_s=60,
            binary=True,
        ).stdout

    def inspect(base: Path, layers: list, out: Path, frames: int, timeout_s: float) -> None:
        before = audio_packets(base)
        composite(base, layers, out, frames, timeout_s)
        assert before and audio_packets(out) == before
        verified.append(True)

    monkeypatch.setattr(captions, "_composite", inspect)
    assert cli.main(_args(clip, words, tmp_path / "captioned.mp4")) == 0
    assert verified == [True]


def test_caption_output_srt_and_receipt(
    clip: Path,
    words: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fake_layers: list[dict],
) -> None:
    out = tmp_path / "captioned.mp4"
    assert cli.main(_args(clip, words, out) + ["--title", "Small|steps"]) == 0
    summary = json.loads(capsys.readouterr().out)
    facts = ffmpeg.probe(out)
    assert (facts.width, facts.height, facts.fps) == (1080, 1920, Fraction(30))
    assert facts.video_frames == summary["frames"] == 38
    assert (facts.color_space, facts.color_primaries, facts.color_transfer) == ("bt709",) * 3
    assert facts.color_range == "tv"
    assert (facts.audio_codec, facts.sample_rate, facts.channels) == ("aac", 48000, 2)
    style = resolve("preset:electric")
    phrases = build_phrases(
        [
            CaptionWord(f"w{i}", word.text, word.start, word.end)
            for i, word in enumerate(Transcript.load(words).words)
        ],
        style.data["captions"],
    )
    assert summary["phrases"] == len(phrases) == 2
    assert out.with_suffix(".srt").read_text() == (
        "1\n00:00:00,100 --> 00:00:00,640\nFirst words.\n\n"
        "2\n00:00:00,700 --> 00:00:01,267\nNext phrase.\n\n"
    )
    record = json.loads(out.with_suffix(".receipt.json").read_text())
    assert record["schema"] == "yap-studio/receipt@1"
    assert record["stage"] == "captions"
    assert record["style"] == {"name": style.name, "sha256": style.sha256()}
    assert record["renderer"] == {"name": "hyperframes", "version": hyperframes.HYPERFRAMES_VERSION,
                                  "workers": "auto", "capture": "screenshot", "browser_gpu": False}
    assert [item["path"] for item in record["inputs"]] == [str(clip), str(words)]
    assert {item["path"] for item in record["outputs"]} == {out.name, out.with_suffix(".srt").name}
    for item in record["outputs"]:
        target = tmp_path / item["path"]
        assert item["sha256"] == receipt.sha256_file(target)
        assert item["bytes"] == target.stat().st_size
    assert "plain" in fake_layers[0]["variables"]
    assert "phrases" in fake_layers[1]["variables"]
    assert [layer["workers"] for layer in fake_layers] == [1, "auto"]
    assert [layer["browser_gpu"] for layer in fake_layers] == [True, False]
    assert not list(tmp_path.glob(".yap-captions-*"))
    assert not list(tmp_path.glob(".yap_tmp_*"))


def test_standalone_giant_uses_resolved_size_ground_and_phrase_limits(
    clip: Path, words: Path, tmp_path: Path, fake_layers: list[dict],
) -> None:
    style = tmp_path / "giant.json"
    style.write_text(json.dumps({"schema_version": 1, "brand": {"accent_color": "#ABCDEF"},
                                 "captions": {"highlight_style": "giant", "size_px": 83,
                                              "max_words_per_phrase": 8}}))
    text = "One two three four five six seven eight"
    Transcript([Word(word, 0.1 + i * 0.1, 0.2 + i * 0.1)
                for i, word in enumerate(text.split())]).save(words)
    out = tmp_path / "giant.mp4"
    assert cli.main(["captions", str(clip), "--words", str(words), "--out", str(out),
                     "--style", str(style), "--json"]) == 0
    variables = fake_layers[0]["variables"]
    assert variables["size_px"] == 124
    assert variables["background_mode"] == "box"
    assert variables["background_color"] == "#ABCDEF"
    assert all(len(slot["words"]) <= 3 for slot in json.loads(variables["phrases"]))


def test_bottom_caption_clamp_warning_survives_cli_receipt(
    clip: Path, words: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    style = tmp_path / "style.json"
    style.write_text(json.dumps({
        "schema_version": 1, "name": "Bottom caption",
        "captions": {
            "position": "bottom", "top_px": 1650, "size_px": 104,
            "shadow_blur_px": 5, "shadow_x_px": 3, "shadow_y_px": 4, "entry_rise_px": 12,
        },
    }))
    out = tmp_path / "captioned.mp4"
    assert cli.main([
        "captions", str(clip), "--words", str(words), "--out", str(out),
        "--style", str(style), "--json",
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    warnings = summary["warnings"]
    assert len(warnings) == 1 and "clamped" in warnings[0]
    assert "1650" in warnings[0] and "1740" in warnings[0]
    record = json.loads(out.with_suffix(".receipt.json").read_text())
    assert record["warnings"] == warnings


@pytest.mark.parametrize("case", ["moving_face", "unmeasured", "fixed", "default", "preset"])
def test_standalone_caption_placement_mode(
    clip: Path, words: Path, tmp_path: Path, fake_layers: list[dict],
    monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    from yap_studio.faces import detect
    from yap_studio.faces.detect import find_gaps
    from yap_studio.faces.model import Box, FaceBox, FacesData, VideoIdentity
    from yap_studio.stages.captions import caption_band_geometry

    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    source = tmp_path / "style.json"
    data = {"schema_version": 1, "captions": {}}
    if case not in {"default", "preset"}:
        data["captions"]["placement"] = "fixed" if case == "fixed" else "avoid_face"
    source.write_text(json.dumps(data))
    spec = "preset:electric" if case == "preset" else str(source)
    style = resolve(spec)
    assert style.data["captions"]["placement"] == ("fixed" if case == "fixed" else "avoid_face")
    assert style.caption_placement_explicit is (case not in {"default", "preset"})
    geometry = caption_band_geometry(style.data["captions"], safe_bottom_px=1740)
    measured = []

    def detect_conformed(video: Path, *, every: int) -> FacesData:
        assert case not in {"fixed", "default", "preset"}, "Fixed clips must not measure faces"
        info = ffmpeg.probe(video)
        assert info.fps == 30 and (info.width, info.height) == (1080, 1920)
        measured.append(video.name)
        frames = tuple((FaceBox(Box(400, geometry.y + 40, 200, 180), 0.95, "detected"),)
                       if frame < 20 else () for frame in range(info.video_frames))
        return FacesData(
            "unmeasured" if case == "unmeasured" else "measured",
            "unavailable" if case == "unmeasured" else None,
            VideoIdentity(video.name, "a" * 64, info.video_frames, Fraction(30), 1080, 1920),
            {}, {}, () if case == "unmeasured" else frames,
            () if case == "unmeasured" else tuple(find_gaps(frames)),
        )

    monkeypatch.setattr(detect, "detect_faces", detect_conformed)
    out = tmp_path / "captioned.mp4"
    assert cli.main(["captions", str(clip), "--words", str(words), "--out", str(out),
                     "--style", spec, "--json"]) == 0
    slots = json.loads(fake_layers[0]["variables"]["phrases"])
    record = json.loads(out.with_suffix(".receipt.json").read_text())
    if case == "moving_face":
        assert slots[0]["offset_px"] < 0 and slots[1]["offset_px"] == 0
        assert record["caption_placement"]["kept_phrases"] == 1
    else:
        assert all("offset_px" not in slot for slot in slots)
    if case == "unmeasured":
        assert record["caption_placement"]["faces"] == {
            "status": "unmeasured", "reason": "unavailable",
        }
    if case in {"fixed", "default", "preset"}:
        assert "caption_placement" not in record
    assert measured == ([] if case in {"fixed", "default", "preset"} else ["conformed.mp4"])


def test_repeat_requires_force_and_current_receipt(
    clip: Path,
    words: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "owned.mp4"
    args = _args(clip, words, out)
    assert cli.main(args) == 0
    capsys.readouterr()
    before = out.read_bytes()
    assert cli.main(args) == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert out.name in error["message"] and "--force" in error["suggested_action"]
    assert out.read_bytes() == before
    assert cli.main(args + ["--force"]) == 0
    capsys.readouterr()
    out.write_bytes(b"external replacement")
    assert cli.main(args + ["--force"]) == 3
    assert out.read_bytes() == b"external replacement"


def test_interrupted_publish_can_retry_with_force(
    clip: Path, words: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "owned.mp4"
    args = _args(clip, words, out)
    assert cli.main(args) == 0
    old_records = json.loads(out.with_suffix(".receipt.json").read_text())["outputs"]
    original_srt = out.with_suffix(".srt").read_bytes()
    Transcript([Word("Changed", 0.1, 0.4)]).save(words)
    replace_output = paths.os.replace

    def interrupt(source: Path, target: Path) -> None:
        if Path(target) == out.with_suffix(".srt"):
            transition = json.loads(out.with_suffix(".receipt.json").read_text())
            assert transition["status"] == "publishing"
            assert all(record in transition["outputs"] for record in old_records)
            assert receipt.sha256_file(out) != old_records[0]["sha256"]
            assert any(
                record["path"] == out.name and record["sha256"] == receipt.sha256_file(out)
                for record in transition["outputs"]
            )
            assert len(transition["outputs"]) == 4
            assert any(
                record["path"] == out.with_suffix(".srt").name
                and record["sha256"] == receipt.sha256_file(source)
                for record in transition["outputs"]
            )
            raise OSError("subtitle publication interrupted")
        replace_output(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(paths.os, "replace", interrupt)
        with pytest.raises(OSError, match="publication interrupted"):
            captions.caption_clip(
                clip, style=resolve("preset:electric"), out=out, words=words,
                title=None, srt=True, force=True,
            )
    assert out.with_suffix(".srt").read_bytes() == original_srt
    assert cli.main(args + ["--force"]) == 0
    final = json.loads(out.with_suffix(".receipt.json").read_text())
    assert final["status"] == "verified" and len(final["outputs"]) == 2
    for record in final["outputs"]:
        assert record["sha256"] == receipt.sha256_file(tmp_path / record["path"])
    out.write_bytes(b"external replacement")
    assert cli.main(args + ["--force"]) == 3
    assert out.read_bytes() == b"external replacement"


def test_concurrent_unforced_publish_preserves_first_outputs_and_receipt(
    clip: Path, words: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "shared.mp4"
    targets = [out, out.with_suffix(".srt"), out.with_suffix(".receipt.json")]
    first_waiting = threading.Event()
    second_ready = threading.Event()
    second_publishing = threading.Event()
    release_first = threading.Event()
    first_published = threading.Event()
    local = threading.local()
    originals: list[bytes] = []
    build_receipt = receipt.build_receipt
    write_receipt = receipt.write_receipt

    def staged_receipt(*args: object, **kwargs: object) -> dict:
        result = build_receipt(*args, **kwargs)  # type: ignore[arg-type]
        if local.role == "second":
            second_ready.set()
        return result

    def interleave_publish(target: Path, record: dict) -> Path:
        if record["status"] == "publishing":
            if local.role == "first":
                first_waiting.set()
                assert release_first.wait(timeout=120)
            else:
                second_publishing.set()
                assert first_published.wait(timeout=120)
        result = write_receipt(target, record)
        if local.role == "first" and record["status"] == "verified":
            originals.extend(target.read_bytes() for target in targets)
            first_published.set()
        return result

    def run(role: str) -> dict:
        local.role = role
        return captions.caption_clip(
            clip, style=resolve("preset:electric"), out=out, words=words,
            title=None, srt=True, force=False,
        )

    monkeypatch.setattr(receipt, "build_receipt", staged_receipt)
    monkeypatch.setattr(receipt, "write_receipt", interleave_publish)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run, "first")
        try:
            assert first_waiting.wait(timeout=120)
            assert not any(target.exists() for target in targets)
            second = pool.submit(run, "second")
            assert second_ready.wait(timeout=120)
            # Give the competing publisher a chance to reach the locked transaction.
            second_publishing.wait(timeout=0.2)
            release_first.set()
            first.result(timeout=120)
            with pytest.raises(captions.CaptionRefusal):
                second.result(timeout=120)
        finally:
            release_first.set()
            first_published.set()
    assert len(originals) == len(targets)
    assert [target.read_bytes() for target in targets] == originals
    assert json.loads(targets[-1].read_text())["status"] == "verified"


def test_subtitle_staging_failure_preserves_outputs_and_receipt(
    clip: Path, words: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "owned.mp4"
    args = _args(clip, words, out)
    assert cli.main(args) == 0
    targets = [out, out.with_suffix(".srt"), out.with_suffix(".receipt.json")]
    originals = [target.read_bytes() for target in targets]
    Transcript([Word("Changed", 0.1, 0.4)]).save(words)
    write_text = Path.write_text

    def fail_subtitles(target: Path, text: str, **kwargs: object) -> int:
        if target.suffix == ".srt":
            raise OSError("subtitle staging interrupted")
        return write_text(target, text, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", fail_subtitles)
    with pytest.raises(OSError, match="subtitle staging interrupted"):
        captions.caption_clip(
            clip, style=resolve("preset:electric"), out=out, words=words,
            title=None, srt=True, force=True,
        )
    assert [target.read_bytes() for target in targets] == originals
    assert not list(tmp_path.glob(".yap-captions-*"))


def test_trailing_phrases_share_clipped_slots_srt_and_count(
    clip: Path, words: Path, tmp_path: Path,
    capsys: pytest.CaptureFixture[str], fake_layers: list[dict],
) -> None:
    Transcript([
        Word("Inside.", 0.7, 2.0), Word("Outside.", 3.0, 3.3),
    ]).save(words)
    out = tmp_path / "clipped.mp4"
    assert cli.main(_args(clip, words, out)) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["phrases"] == 1
    slots = json.loads(fake_layers[0]["variables"]["phrases"])
    assert len(slots) == 1
    assert slots[0]["start"] == 0.7 and slots[0]["end"] == summary["frames"] / 30
    assert out.with_suffix(".srt").read_text() == (
        "1\n00:00:00,700 --> 00:00:01,267\nInside.\n\n"
    )


def test_force_refuses_unowned_media(
    clip: Path,
    words: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "existing.mp4"
    out.write_bytes(clip.read_bytes())
    before = out.read_bytes()
    assert cli.main(_args(clip, words, out) + ["--force"]) == 3
    assert "new --out" in json.loads(capsys.readouterr().err)["error"]["suggested_action"]
    assert out.read_bytes() == before


def test_changed_srt_is_refused_before_media_replacement(
    clip: Path,
    words: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "owned.mp4"
    args = _args(clip, words, out)
    assert cli.main(args) == 0
    original_media = out.read_bytes()
    original_receipt = out.with_suffix(".receipt.json").read_bytes()
    out.with_suffix(".srt").write_text("external subtitles")
    assert cli.main(args + ["--force"]) == 3
    assert out.read_bytes() == original_media
    assert out.with_suffix(".receipt.json").read_bytes() == original_receipt
    assert out.with_suffix(".srt").read_text() == "external subtitles"


def test_words_support_a_silent_vertical_clip(
    clip: Path,
    words: Path,
    tmp_path: Path,
) -> None:
    silent = tmp_path / "silent.mp4"
    ffmpeg.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(clip),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            str(silent),
        ],
        timeout_s=60,
    )
    out = tmp_path / "captioned.mp4"
    assert cli.main(_args(silent, words, out)) == 0
    assert not ffmpeg.probe(out).has_audio


@pytest.mark.parametrize("kind", ["horizontal", "square", "hdr"])
def test_project_flow_refusal(
    tmp_path: Path,
    words: Path,
    kind: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = synth.calib_source(
        tmp_path / "source.mov",
        seconds=0.25,
        size={"horizontal": "192x108", "square": "108x108", "hdr": "108x192"}[kind],
    )
    if kind == "hdr":
        tagged = tmp_path / "hdr.mp4"
        ffmpeg.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-c:v",
                "libx264",
                "-vf",
                "setparams=color_trc=arib-std-b67",
                "-color_trc",
                "arib-std-b67",
                str(tagged),
            ],
            timeout_s=60,
        )
        source = tagged
    out = tmp_path / "out.mp4"
    assert cli.main(_args(source, words, out)) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert "project flow" in error["message"] and "yap-studio init" in error["message"]
    assert not out.exists()


def test_title_disabled_by_style(
    clip: Path,
    words: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    style_file = tmp_path / "plain.json"
    style_file.write_text(json.dumps({"schema_version": 1, "hook": {"style": "none"}}))
    args = _args(clip, words, tmp_path / "out.mp4")
    args[args.index("preset:electric")] = str(style_file)
    assert cli.main(args + ["--title", "Opening|words"]) == 2
    assert "hook.style is none" in json.loads(capsys.readouterr().err)["error"]["message"]


def test_missing_backend_and_renderer_fix(
    clip: Path,
    tmp_path: Path,
    words: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(transcript, "resolve_backend", lambda name: None)
    assert cli.main(["captions", str(clip), "--out", str(tmp_path / "out.mp4"), "--json"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "speech_engine_not_found"

    def blocked(*args: object, **kwargs: object) -> None:
        raise RendererBlockedError()

    monkeypatch.setattr(hyperframes, "render_layer", blocked)
    args = _args(clip, words, tmp_path / "out.mp4")
    args.remove("--json")
    assert cli.main(args) == 2
    assert "outside the agent sandbox" in capsys.readouterr().err
    assert not (tmp_path / "out.mp4").exists()


def test_transcribes_when_words_omitted(
    clip: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = []

    class Backend:
        def transcribe_many(self, sources: list[Path]) -> dict[Path, Transcript]:
            seen.extend(sources)
            return {sources[0]: Transcript([Word("Speech.", 0.1, 0.7)])}

    monkeypatch.setattr(transcript, "resolve_backend", lambda name: Backend())
    assert cli.main(["captions", str(clip), "--out", str(tmp_path / "out.mp4")]) == 0
    assert seen == [clip.resolve()]


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        "{}",
        '{"schema":"yap-studio/transcript@1","words":[{"text":"word","start":0.5,"end":0.2}]}',
    ],
)
def test_invalid_words_exit_two(
    clip: Path,
    words: Path,
    tmp_path: Path,
    payload: str,
) -> None:
    words.write_text(payload)
    assert cli.main(_args(clip, words, tmp_path / "out.mp4")) == 2


def test_project_captions_cli_default_directory_json_force_and_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    fake_layers: list[dict],
) -> None:
    from test_stage_captions import make_project

    project = make_project(tmp_path)
    monkeypatch.chdir(project)
    assert cli.main(["captions", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["out"] == "captions/captions.mov" and result["frames"] == 90
    assert cli.main(["captions", str(project)]) == 0
    assert "Captions unchanged" in capsys.readouterr().out and len(fake_layers) == 1
    assert cli.main(["captions", "--force", "--json"]) == 0
    assert not json.loads(capsys.readouterr().out)["noop"] and len(fake_layers) == 2
    assert cli.main(["status", "--json"]) == 0
    stages = {item["name"]: item for item in json.loads(capsys.readouterr().out)["stages"]}
    assert stages["captions"]["state"] == "done" and stages["graphics"]["state"] == "missing"


@pytest.mark.parametrize("options", [["--style", "preset:electric"], ["--out", "other.mp4"],
                                     ["--words", "words.json"], ["--title", "Title"],
                                     ["--title", ""]])
def test_project_mode_refuses_standalone_overrides(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], options: list[str],
) -> None:
    from test_stage_captions import make_project

    project = make_project(tmp_path)
    assert cli.main(["captions", str(project), "--json", *options]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert "project words, timeline and style" in error["message"]
    assert not (project / "captions").exists()


def test_standalone_clip_still_requires_output(clip: Path) -> None:
    assert cli.main(["captions", str(clip)]) == 2


def test_project_captions_cli_timeline_and_missing_input_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from test_stage_captions import make_project, set_elements

    project = make_project(tmp_path)
    set_elements(project, {"id": "agent-gap", "kind": "caption_gap",
                           "start": {"time": 2.9}, "hold_s": 1})
    assert cli.main(["captions", str(project), "--json"]) == 3
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "timeline_refused"
    (project / "cut/receipt.json").unlink()
    assert cli.main(["captions", str(project), "--json"]) == 3
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "stale_cut"


def test_captions_keeps_minus_sixteen_master_with_zero_lag(tmp_path: Path, words: Path) -> None:
    """Standalone captioning must retain an approved master's loudness and timing."""
    import numpy as np

    from yap_studio.stages.assemble_audio import decode, measure

    source = synth.loudness_master(tmp_path / "master.mov")
    out = tmp_path / "captioned.mp4"
    assert cli.main(_args(source, words, out)) == 0
    assert measure(source, []).integrated_lufs == pytest.approx(-16, abs=0.1)
    assert measure(out, []).integrated_lufs == pytest.approx(-16, abs=0.1)
    original = decode(source, [], channels=1)[:, 0]
    decoded = decode(out, [], channels=1)[:, 0]
    scores = [np.dot(original[100:10000], decoded[100 + lag:10000 + lag])
              for lag in range(-20, 21)]
    assert np.argmax(scores) - 20 == 0


@pytest.mark.parametrize("approved,json_output", [(True, False), (False, False), (True, True)])
def test_captions_command_applies_and_records_taste(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    approved: bool, json_output: bool,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    identities = []
    if approved:
        for body in (_body(), _body("#654321", field="hook.text_color")):
            identity = add_proposals(body)[0].id
            approve(identity)
            identities.append(identity)
    captured = []

    def capture_style(clip: Path, *, style, out: Path, title: str | None, **kwargs) -> dict:
        captured.append(style)
        assert title == "Opening|title"
        out.write_bytes(b"finished render")
        return {
            "out": str(out), "srt": None, "receipt": str(out.with_suffix(".receipt.json")),
            "frames": 30, "phrases": 1, "style": {"name": style.name, "sha256": style.sha256()},
        }

    monkeypatch.setattr(captions, "caption_clip", capture_style)
    out = tmp_path / "out.mp4"
    arguments = ["captions", str(tmp_path / "clip.mp4"), "--out", str(out),
                 "--style", "preset:clean_minimal", "--title", "Opening|title"]
    if json_output:
        arguments.append("--json")
    assert cli.main(arguments) == 0
    output = capsys.readouterr().out
    assert len(captured) == 1
    style = captured[0]
    record = read_taste_record(out)
    assert record["style_sha256"] == style.sha256()
    assert record["revision"] == (2 if approved else 0)
    assert {item["id"] for item in record["applied"]} == set(identities)
    if approved:
        assert style.data["captions"]["text_color"] == "#123456"
        assert style.data["hook"]["text_color"] == "#654321"
    else:
        assert style.data == resolve("preset:clean_minimal").data
    if json_output:
        pin = json.loads(output)["taste"]
        assert pin == {key: value for key, value in record.items()
                       if key not in {"schema", "media_sha256"}}
    else:
        assert ("Taste: revision 2, 2 rules applied\n" if approved else
                "Taste: none approved\n") in output


def test_delayed_taste_record_refuses_second_caption_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    applications = []
    resolve_style = taste_render.render_style

    def capture_application(spec):
        application = resolve_style(spec)
        applications.append(application)
        return application

    def publish(clip: Path, *, style, out: Path, **kwargs) -> dict:
        assert kwargs["force"] is True
        with captions._publish_lock(out, 140):
            out.write_bytes(style.sha256().encode())
            out.with_suffix(".receipt.json").write_text(json.dumps({
                "style": {"sha256": style.sha256()},
                "outputs": [{"path": out.name, "sha256": receipt.sha256_file(out)}],
            }))
        return {"out": str(out), "frames": 30}

    monkeypatch.setattr(taste_render, "render_style", capture_application)
    monkeypatch.setattr(captions, "caption_clip", publish)
    out = tmp_path / "out.mp4"
    arguments = ["captions", str(tmp_path / "clip.mp4"), "--out", str(out), "--force", "--json"]
    for color in ("#123456", "#654321"):
        approve(add_proposals(_body(color))[0].id)
        assert cli.main(arguments) == 0
        capsys.readouterr()
    assert applications[0].style.sha256() != applications[1].style.sha256()
    target = out.with_suffix(".taste.json")
    second_record = target.read_bytes()
    with pytest.raises(InputError, match="render changed while its taste was being recorded"):
        with captions._publish_lock(out, 140):
            write_taste_record(out, applications[0])
    assert target.read_bytes() == second_record
    assert read_taste_record(out) == {
        "schema": taste_render.TASTE_RECORD_SCHEMA,
        "media_sha256": receipt.sha256_file(out), **applications[1].pin(),
    }


def test_captions_records_taste_under_publish_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    out = tmp_path / "out.mp4"
    locked = False
    lock_calls = []
    publish_lock = captions._publish_lock
    record_write = taste_render.write_taste_record

    @contextmanager
    def track_lock(media: Path, timeout_s: float):
        nonlocal locked
        lock_calls.append((media, timeout_s))
        with publish_lock(media, timeout_s):
            locked = True
            try:
                yield
            finally:
                locked = False

    def publish(clip: Path, **kwargs) -> dict:
        out.write_bytes(b"render")
        return {"out": str(out), "frames": 60}

    def record_under_lock(media: Path, application) -> Path:
        assert locked
        return record_write(media, application)

    monkeypatch.setattr(captions, "caption_clip", publish)
    monkeypatch.setattr(captions, "_publish_lock", track_lock)
    monkeypatch.setattr(taste_render, "write_taste_record", record_under_lock)
    assert cli.main(["captions", str(tmp_path / "clip.mp4"), "--out", str(out), "--json"]) == 0
    assert lock_calls == [(out, 160)]
    assert read_taste_record(out) is not None
