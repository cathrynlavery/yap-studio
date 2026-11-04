"""Verifier patch audio uses the same resampling as the rendered audio."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import synth
from test_assemble import refuse_unexpected_matte_work as refuse_unexpected_matte_work
from test_cli_captions import fake_layers  # noqa: F401
from test_cut import make_project, time_segment, write_edit, write_json
from test_stage_captions import set_elements

from yap_studio import audio, cli, ffmpeg, receipt
from yap_studio.edl import load_edit_list
from yap_studio.errors import ValidationError
from yap_studio.project.receipts import StageRun, read_receipt
from yap_studio.project.status import ProjectContext, ingest_inputs
from yap_studio.stages import assemble as assembly
from yap_studio.stages.captions import captions
from yap_studio.stages.cut import _spans, cut, resolve_cut
from yap_studio.stages.graphics import graphics
from yap_studio.stages.music import music
from yap_studio.style import resolve
from yap_studio.verify import verify
from yap_studio.voice import decide, make, propose
from yap_studio.voice.splice import blend, decode, write_wav
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


@pytest.fixture
def patched_assembly(tmp_path, request):
    selected = getattr(request, "param", False)
    source = synth.voiced_source(
        tmp_path / "source.mov", [(1.2, 1.8), (2.5, 2.8), (5.2, 5.8), (6.5, 6.8)],
        frequencies=[220, 220, 220, 880], seconds=8,
    )
    if selected:
        early = synth.float_audio(tmp_path / "early.wav", np.zeros(8 * 48000))
        multiple = tmp_path / "multiple.mov"
        ffmpeg.run([
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(early),
            "-itsoffset", "0.2", "-i", str(source), "-map", "1:v:0",
            "-map", "0:a:0", "-map", "1:a:0", "-c", "copy", str(multiple),
        ], timeout_s=60)
        source = multiple
    root = make_project(tmp_path / "project", [source], word_rows=[[
        ("bright", 1.2, 1.8), ("coal", 2.5, 2.8),
        ("light", 5.2, 5.8), ("cold", 6.5, 6.8),
    ]])
    project = json.loads((root / "project.json").read_text())
    project.update(asr={"engine": "none"}, music={"mode": "none"},
                   audio={"mode": "preserve"}, style="style.json")
    project["takes"][0].update(voiceover=True, audio_stream=int(selected))
    style = resolve("preset:clean_minimal").data
    style["broll"]["color_match"] = "off"
    style["music"]["sound_effects_fallback"] = "none"
    write_json(root / "style.json", style)
    write_json(root / "project.json", project)
    previous = read_receipt(root / "ingest")
    with StageRun(root, "ingest") as run:
        for name, digest in ingest_inputs(ProjectContext(root)).items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in previous["outputs"]})
        run.finish()
    write_edit(root, [time_segment("s1", "t1", 0, 3), time_segment("s2", "t1", 5, 8)],
               end_hold_s=0)
    propose(root, "t1:1", "t1:1", "cold")
    fix = make(root, ["f1"])["fixes"][0]
    decide(root, "f1", "approve", via="chat")
    cut(root)
    cover = root / "takes/t1.mov"
    set_elements(root, {"id": "broll-cover", "kind": "insert", "start": {"time": 0},
                        "end": {"time": 6}, "media": {
                            "path": str(cover.relative_to(root)),
                            "sha256": receipt.sha256_file(cover),
                        }}, lane="broll")
    captions(root)
    graphics(root)
    music(root)
    return root, fix


@pytest.mark.parametrize("stage", ["cut", "final"])
def test_retained_reference_keeps_fix_past_original_audio(tmp_path, stage):
    source = synth.voiced_source(
        tmp_path / "source.mov", [(0.2, 0.9), (1.6, 1.95)], seconds=3,
    )
    short = tmp_path / "short.mov"
    ffmpeg.run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(source),
        "-map", "0:v:0", "-map", "0:a:0", "-af", "atrim=duration=2",
        "-c:v", "copy", "-c:a", "pcm_s16le", str(short),
    ], timeout_s=60)
    root = make_project(tmp_path / "project", [short], word_rows=[[
        ("cold", 0.2, 0.9), ("coal", 1.6, 1.95),
    ]])
    project = json.loads((root / "project.json").read_text())
    project.update(asr={"engine": "none"}, music={"mode": "none"},
                   audio={"mode": "preserve"})
    project["takes"][0]["voiceover"] = True
    write_json(root / "project.json", project)
    previous = read_receipt(root / "ingest")
    with StageRun(root, "ingest") as run:
        for name, digest in ingest_inputs(ProjectContext(root)).items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in previous["outputs"]})
        run.finish()
    write_edit(root, [time_segment("s1", "t1", 0, 3)], end_hold_s=0)
    propose(root, "t1:1", "t1:1", "cold")
    fix = make(root, ["f1"])["fixes"][0]
    decide(root, "f1", "approve", via="chat")
    meta = fix["patch"]
    first, last = meta["start_sample"], meta["start_sample"] + meta["samples"]
    _, original = audio.decode(short, sr=48000)
    assert len(original) == 2 * 48000 < last < 3 * 48000
    cut(root)
    _, voice = audio.decode(root / "cut/voice.wav", sr=48000)
    assert np.any(voice[len(original):last])
    if stage == "final":
        cover = root / "takes/t1.mov"
        set_elements(root, {"id": "broll-cover", "kind": "insert", "start": {"time": 0},
                            "end": {"time": 3}, "media": {
                                "path": str(cover.relative_to(root)),
                                "sha256": receipt.sha256_file(cover),
                            }}, lane="broll")
        captions(root)
        graphics(root)
        music(root)
        assembly.assemble(root)
        reference = load_edit_list(root / "final/verify-edl.json").sources["t1"].path
    else:
        reference = root / "cut/source-audio/t1.wav"
    _, actual = audio.decode(reference, sr=48000)
    assert len(actual) >= last, "retained reference truncates the approved voice fix"
    # The cut's voice track is quantized to 16-bit PCM.
    np.testing.assert_allclose(actual[first:last], voice[first:last], atol=1 / 32768, rtol=0)


@pytest.mark.parametrize("patched_assembly", [True], indirect=True)
def test_remuxed_reference_rebases_patch_to_its_video_clock(patched_assembly, monkeypatch):
    root, fix = patched_assembly
    plan = resolve_cut(root)
    assert plan.media["t1"].video_offset_s == pytest.approx(0.2)
    check = assembly.verify
    seen = []

    def check_clock(media, **kwargs):
        reference = load_edit_list(kwargs["edl"]).sources["t1"].path
        info = ffmpeg.probe(reference)
        assert info.video_offset_s == 0
        _, actual = audio.decode(reference, sr=48000)
        if "source_audio" in kwargs:
            actual = kwargs["source_audio"]("t1", actual, 48000)
        meta = fix["patch"]
        first = meta["start_sample"] - round(plan.media["t1"].video_offset_s * 48000)
        patch = decode(root / meta["path"], 1)
        _, original = audio.decode(plan.sources["t1"], stream=1, sr=48000)
        original = original[9600:9600 + len(actual)]
        expected = blend(original[:, None], patch, {**meta, "start_sample": first},
                         source_start_sample=0)[:, 0]
        np.testing.assert_allclose(actual[first:first + len(patch)],
                                   expected[first:first + len(patch)], atol=1e-6,
                                   err_msg="patch follows the original container clock")
        seen.append(reference)
        return check(media, **kwargs)

    monkeypatch.setattr(assembly, "verify", check_clock)
    assembly.assemble(root)
    assert len(seen) == 1


@pytest.mark.parametrize("patched_assembly", [False, True], indirect=True)
def test_published_cut_verification_reproduces_approved_fix_near_join(patched_assembly, capsys):
    root, fix = patched_assembly
    stored = json.loads((root / "cut/verify.json").read_text())
    assert not [item for item in stored["findings"] if item["check"] == "join.words"]
    report = verify(str(root / "cut/cut.mp4"), edl=str(root / "cut/spans.json"),
                    asr="none", loudness="note", detail="full")
    assert report["checks"]["join.words"]["ran"]
    failures = [item for item in report["findings"] if item["check"] == "join.words"]
    assert not failures, "standalone cut verification reconstructs the old speech"
    spans = json.loads((root / "cut/spans.json").read_text())
    assert load_edit_list(root / "cut/spans.json").sources["t1"].path.resolve() == (
        root / "takes/t1.mov"
    ).resolve()
    reference = root / "cut" / spans["verification_audio"]["t1"]
    recorded = read_receipt(root / "cut")
    assert str(reference.relative_to(root)) in {item["path"] for item in recorded["outputs"]}
    _, actual = audio.decode(reference, sr=48000)
    plan = resolve_cut(root)
    _, original = audio.decode(plan.sources["t1"],
                               stream=plan.context.project.takes[0].audio_stream, sr=48000)
    expected = source_transform(plan, [(fix, decode(root / fix["patch"]["path"], 1))])(
        "t1", original, 48000,
    )
    np.testing.assert_array_equal(actual, expected)
    assert cli.main(["verify", str(root / "cut/cut.mp4"), "--edl", str(root / "cut/spans.json"),
                     "--asr", "none", "--json"]) == 0
    assert not [item for item in json.loads(capsys.readouterr().out)["findings"]
                if item["check"] == "join.words"]


def test_cut_without_fixes_keeps_existing_span_references(voice_project):
    expected = _spans(resolve_cut(voice_project), voice_project / "cut", [])
    cut(voice_project)
    assert json.loads((voice_project / "cut/spans.json").read_text()) == expected
    assert not any("source-audio/" in item["path"]
                   for item in read_receipt(voice_project / "cut")["outputs"])


@pytest.mark.parametrize("references", [[], {"t2": "audio.wav"}, {"t1": 2}])
def test_verification_audio_requires_known_sources_and_paths(tmp_path, references):
    source = synth.voiced_source(tmp_path / "source.mov", [(0.2, 0.8)], seconds=1)
    edl = tmp_path / "spans.json"
    write_json(edl, {"sources": {"t1": str(source)}, "segments": [
        {"source": "t1", "start": 0, "end": 1},
    ], "verification_audio": references})
    with pytest.raises(ValidationError, match="verification_audio"):
        verify(str(source), edl=str(edl), asr="none")


def test_published_verification_reproduces_approved_fix_near_join(patched_assembly):
    root, _ = patched_assembly
    result = assembly.assemble(root)
    report = verify(str(root / result["out"]), edl=str(root / "final/verify-edl.json"),
                    asr="none", loudness="note", detail="full")
    assert report["checks"]["join.words"]["ran"]
    failures = [item for item in report["findings"] if item["check"] == "join.words"]
    assert not failures, "standalone verification reconstructs the old speech"


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
