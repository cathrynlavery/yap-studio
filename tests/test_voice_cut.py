"""Approved patches affect only their samples and participate in cut freshness."""

import json

import numpy as np
import pytest
import synth
from test_cut import time_segment, write_edit, write_json
from voice_helpers import linked

from yap_studio.errors import YapError
from yap_studio.project.status import ProjectContext, cut_inputs
from yap_studio.stages.cut import cut
from yap_studio.voice import consent, decide, list_fixes, make, propose


@pytest.mark.parametrize("channels,gain,change", [(1, 0, 0), (1, 3, 0), (2, 0, 0), (1, 0, 4)])
def test_cut_level_uses_the_reference_channel_layout(tmp_path, monkeypatch, channels, gain, change):
    from importlib import import_module

    from test_voice_splice import fixture_sound

    from yap_studio.stages import take_levels
    from yap_studio.voice.splice import _speech, edge_check, level
    from yap_studio.voice.store import add_cut_checks

    original = np.repeat(fixture_sound()[96000:144000], channels, axis=1)
    reference = _speech(original)
    lufs, _ = take_levels.measure(reference, tmp_path, "reference")
    assert lufs is not None
    sound = np.repeat(original, 2, axis=1) if channels == 1 else original.copy()
    sound *= 10 ** ((gain + change) / 20)
    fix = {"id": "f1", "take": "t1", "say": "test", "patch": {"channels": channels},
           "checks": {"items": [edge_check(original, 0, len(original)), {
               "id": "splice.level", "status": "pass", "reference_lufs": lufs,
               "reference_rms_db": level(reference),
           }]}}
    monkeypatch.setattr(import_module("yap_studio.voice.make"), "_heard", lambda *_, **kwargs:
                        {"id": "splice.heard", "status": "skipped"})
    report = add_cut_checks(
        {"status": "pass", "findings": []}, sound,
        [{"id": "f1", "start_sample": 0, "end_sample": len(sound)}], [fix],
        plan=None, work=tmp_path, levels={"takes": {"t1": {"gain_db": gain}}},
    )
    check = next(item for item in report["voice_fixes"][0]["checks"]["items"]
                 if item["id"] == "splice.level")
    assert check["difference_lu"] == pytest.approx(change, abs=0.15)
    assert check["status"] == ("warn" if change else "pass")


def test_approval_exact_samples_stale_captions_and_withdraw(voice_project, fake_voice, monkeypatch):
    linked(voice_project, monkeypatch)
    cut(voice_project)
    baseline = synth.read_wav(voice_project / "cut/voice.wav")[1].copy()
    before = cut_inputs(ProjectContext(voice_project))
    propose(voice_project, "t1:2", "t1:3", "cold plunge")
    fix = make(voice_project, ["f1"], yes=True, max_credits=20)["fixes"][0]
    assert cut_inputs(ProjectContext(voice_project)) == before
    cut(voice_project, force=True)
    np.testing.assert_array_equal(synth.read_wav(voice_project / "cut/voice.wav")[1], baseline)
    decide(voice_project, "f1", "approve", via="review")
    approved = cut_inputs(ProjectContext(voice_project))
    assert "voice:fixes" in approved
    assert approved != before
    cut(voice_project)
    changed = synth.read_wav(voice_project / "cut/voice.wav")[1]
    first, last = (
        fix["patch"]["start_sample"],
        fix["patch"]["start_sample"] + fix["patch"]["samples"],
    )
    np.testing.assert_array_equal(changed[:first], baseline[:first])
    np.testing.assert_array_equal(changed[last:], baseline[last:])
    assert np.any(changed[first:last] != baseline[first:last])
    words = json.loads((voice_project / "cut/cut-words.json").read_text())["words"]
    assert [row["text"] for row in words if row["id"] in ("t1:2", "t1:3")] == ["cold", "plunge"]
    record = json.loads((voice_project / "cut/receipt.json").read_text())
    assert record["voice_fixes"] == [{"id": "f1", "kind": "clone", "start_sample": first,
                                      "end_sample": last}]
    report = json.loads((voice_project / "cut/verify.json").read_text())
    assert report["voice_fixes"][0]["id"] == "f1"
    consent(voice_project, withdraw=True)
    withdrawn = cut_inputs(ProjectContext(voice_project))
    assert "voice:fixes" in withdrawn
    assert withdrawn["voice:fixes"] != approved["voice:fixes"]
    cut(voice_project)
    np.testing.assert_array_equal(synth.read_wav(voice_project / "cut/voice.wav")[1], baseline)


@pytest.mark.parametrize("whole_take", [False, True])
def test_cut_skips_fix_for_removed_phrase(
    voice_project, fake_voice, monkeypatch, tmp_path, whole_take,
):
    if whole_take:
        from test_cut import make_project

        source = voice_project / "takes/t1.mov"
        words = json.loads((voice_project / "ingest/t1/words.json").read_text())["words"]
        rows = [(word["text"], word["start"], word["end"]) for word in words]
        voice_project = make_project(tmp_path / "two-takes", [source, source],
                                     word_rows=[rows, rows])
        project = json.loads((voice_project / "project.json").read_text())
        for take in project["takes"]:
            take["voiceover"] = True
        write_json(voice_project / "project.json", project)
    removed = [time_segment("s1", "t1", 0, 2), {
        "id": "s2", "take": "t1", "from": {"word": "t1:4"}, "to": {"word": "t1:11"},
    }]
    if whole_take:
        removed = [time_segment("s1", "t2", 0, 12)]
    write_edit(voice_project, removed)
    cut(voice_project)
    expected = {name: (voice_project / "cut" / name).read_bytes()
                for name in ("voice.wav", "cut-words.json", "spans.json")}
    write_edit(voice_project, [time_segment("s1", "t1", 0, 12)])
    linked(voice_project, monkeypatch)
    propose(voice_project, "t1:2", "t1:3", "Home")
    make(voice_project, ["f1"], yes=True, max_credits=20)
    decide(voice_project, "f1", "approve", via="chat")
    before = (voice_project / "voice/fixes.json").read_bytes()
    write_edit(voice_project, removed)
    cut(voice_project)
    for name, content in expected.items():
        assert (voice_project / "cut" / name).read_bytes() == content, name
    assert (voice_project / "voice/fixes.json").read_bytes() == before
    record = json.loads((voice_project / "cut/receipt.json").read_text())
    assert record["voice_fixes"] == []
    assert record["inputs"]["voice:fixes"] == (
        cut_inputs(ProjectContext(voice_project))["voice:fixes"]
    )
    report = json.loads((voice_project / "cut/verify.json").read_text())
    assert not report.get("voice_fixes")


@pytest.mark.parametrize("whole_occurrence", [False, True])
def test_cut_still_refuses_partially_kept_fix(
    voice_project, fake_voice, monkeypatch, whole_occurrence,
):
    linked(voice_project, monkeypatch)
    propose(voice_project, "t1:2", "t1:3", "cold plunge")
    make(voice_project, ["f1"], yes=True, max_credits=20)
    decide(voice_project, "f1", "approve", via="chat")
    segments = [time_segment("partial", "t1", 0, 3)]
    if whole_occurrence:
        segments.insert(0, time_segment("whole", "t1", 0, 12))
    write_edit(voice_project, segments)
    with pytest.raises(YapError) as caught:
        cut(voice_project)
    assert caught.value.code == "voice_words_dropped"
    assert not (voice_project / "cut/voice.wav").exists()


def test_patch_beside_join_and_missing_patch(voice_project, fake_voice, monkeypatch):
    write_edit(voice_project, [time_segment("s1", "t1", 0, 4), time_segment("s2", "t1", 6, 12)])
    linked(voice_project, monkeypatch)
    propose(voice_project, "t1:2", "t1:3", "cold plunge")
    fix = make(voice_project, ["f1"], yes=True, max_credits=20)["fixes"][0]
    decide(voice_project, "f1", "approve", via="chat")
    cut(voice_project)
    report = json.loads((voice_project / "cut/verify.json").read_text())
    assert not [row for row in report["findings"] if row["check"] == "join.words"]
    from test_cut import write_json

    from yap_studio.stages.cut import resolve_cut
    from yap_studio.voice.store import require_assembly_picture

    project = json.loads((voice_project / "project.json").read_text())
    project["takes"][0]["voiceover"] = False
    write_json(voice_project / "project.json", project)
    with pytest.raises(YapError, match="Voice fix f1: Your face is on screen"):
        require_assembly_picture(resolve_cut(voice_project), [], {})
    (voice_project / fix["patch"]["path"]).write_bytes(b"changed")
    with pytest.raises(YapError, match="missing or changed"):
        cut(voice_project)
    (voice_project / fix["patch"]["path"]).unlink()
    with pytest.raises(YapError, match="missing or changed"):
        cut(voice_project)


def test_reapproval_refuses_approved_overlap(voice_project, fake_voice, monkeypatch):
    linked(voice_project, monkeypatch)
    propose(voice_project, "t1:2", "t1:3", "cold plunge")
    make(voice_project, ["f1"], yes=True, max_credits=20)
    decide(voice_project, "f1", "reject", via="review")
    propose(voice_project, "t1:2", "t1:3", "cold water")
    make(voice_project, ["f2"], yes=True, max_credits=20)
    decide(voice_project, "f2", "approve", via="review")
    before = list_fixes(voice_project)
    with pytest.raises(YapError, match="Reject f2 first") as caught:
        decide(voice_project, "f1", "approve", via="chat")
    assert caught.value.code == "voice_fix_overlap"
    assert list_fixes(voice_project) == before
    decide(voice_project, "f2", "reject", via="chat")
    assert decide(voice_project, "f1", "approve", via="chat")["status"] == "approved"
    assert decide(voice_project, "f1", "approve", via="chat")["status"] == "approved"


def test_cut_refuses_hand_edited_approved_overlap(voice_project, fake_voice, monkeypatch):
    from yap_studio.stages.cut import resolve_cut
    from yap_studio.voice.store import patches_for_plan

    linked(voice_project, monkeypatch)
    propose(voice_project, "t1:2", "t1:3", "cold plunge")
    make(voice_project, ["f1"], yes=True, max_credits=20)
    decide(voice_project, "f1", "approve", via="chat")
    propose(voice_project, "t1:6", "t1:7", "water")
    make(voice_project, ["f2"], yes=True, max_credits=20)
    decide(voice_project, "f2", "approve", via="chat")
    data = list_fixes(voice_project)
    first, second = data["fixes"]
    # The word ranges stay separate, but the saved patches share one sample.
    second["patch"]["start_sample"] = (
        first["patch"]["start_sample"] + first["patch"]["samples"] - 1
    )
    write_json(voice_project / "voice/fixes.json", data)
    with pytest.raises(YapError) as caught:
        patches_for_plan(resolve_cut(voice_project))
    assert caught.value.code == "voice_fix_overlap"
    with pytest.raises(YapError) as caught:
        cut(voice_project)
    assert caught.value.code == "voice_fix_overlap"
    assert not (voice_project / "cut/voice.wav").exists()


def test_broken_check_record_is_a_yap_error(voice_project, fake_voice, monkeypatch):
    from test_cut import write_json

    from yap_studio.voice import list_fixes

    linked(voice_project, monkeypatch)
    propose(voice_project, "t1:2", "t1:3", "cold plunge")
    make(voice_project, ["f1"], yes=True, max_credits=20)
    data = json.loads((voice_project / "voice/fixes.json").read_text())
    data["fixes"][0]["checks"]["items"] = [{"id": "splice.level", "status": "pass"}]
    write_json(voice_project / "voice/fixes.json", data)
    with pytest.raises(YapError) as caught:
        list_fixes(voice_project)
    assert caught.value.code == "voice_record_invalid"


@pytest.mark.parametrize("say", ["cold plunge", "Home"])
def test_span_replacements_keep_ingest_anchors(voice_project, say):
    from dataclasses import replace

    from yap_studio.stages.cut import _spans, resolve_cut

    write_edit(voice_project, [{
        "id": "s1", "take": "t1", "from": {"word": "t1:2"}, "to": {"word": "t1:3"},
    }])
    plan = resolve_cut(voice_project)
    snapshot = plan.source_words["t1"]
    plan = replace(plan, source_words={"t1": replace(snapshot, words=tuple(
        replace(word, start=word.start + 0.3, end=word.end + 0.3) for word in snapshot.words
    ))})
    fix = {"take": "t1", "words": ["t1:2", "t1:3"], "say": say}
    row = _spans(plan, voice_project / "cut", [fix])["segments"][0]
    assert row["first_word"] == say.split()[0]
    assert row["first_word_anchor_s"] == 2.2
    if say == "Home":
        assert "last_word" not in row and "last_word_anchor_s" not in row
    else:
        assert row["last_word"] == "plunge"
        assert row["last_word_anchor_s"] == 3.8
    assert plan.source_words["t1"].words[2].text == "coal"


def test_caption_phrase_with_fewer_words_keeps_anchors():
    from yap_studio.captions import CaptionWord, _display_words
    from yap_studio.voice.store import replaced_text

    rows = [
        {"id": f"t1:{i}", "segment": "s1", "text": text, "start": i, "end": i + 1}
        for i, text in enumerate(["standing", "in", "Hope", "Depot"])
    ]
    fix = {"words": [row["id"] for row in rows], "say": "Home Depot"}
    projected = replaced_text(rows, [fix])
    assert [row["id"] for row in projected] == [row["id"] for row in rows]
    displayed = _display_words(
        [CaptionWord(row["id"], row["text"], row["start"], row["end"]) for row in projected], {}
    )
    assert len(displayed) == 1 and displayed[0].text == "Home Depot"
    assert displayed[0].start == 0 and displayed[0].end == 4


def test_merged_fix_cut_packs_spoken_words_and_scans(
    voice_project, fake_voice, monkeypatch, capsys,
):
    from yap_studio.cli import main
    from yap_studio.watch import run_scan

    linked(voice_project, monkeypatch)
    propose(voice_project, "t1:2", "t1:3", "Home")
    make(voice_project, ["f1"], yes=True, max_credits=20)
    decide(voice_project, "f1", "approve", via="chat")
    cut(voice_project)
    words = voice_project / "cut/cut-words.json"
    original = words.read_bytes()
    merged = [row for row in json.loads(original)["words"] if row["id"] in {"t1:2", "t1:3"}]
    assert [row["text"] for row in merged] == ["Home", ""]
    out = voice_project / "watch"
    assert main(["watch", "pack", str(voice_project / "cut/cut.mp4"), "--words", str(words),
                 "--out", str(out), "--asr", "none", "--json"]) == 0
    capsys.readouterr()
    assert run_scan(out, dry_run=True)["recorded"] is False
    transcript = json.loads((out / "transcript.json").read_text())["words"]
    assert all(row["text"].strip() for row in transcript)
    assert [row for row in transcript if row["text"] == "Home"][0]["end"] == merged[0]["end"]
    assert "Home" in (out / "transcript.txt").read_text()
    assert words.read_bytes() == original


def test_fix_preserves_take_level_gains(voice_project, fake_voice, monkeypatch, tmp_path):
    from test_cut import make_project, write_json

    source = voice_project / "takes/t1.mov"
    words = json.loads((voice_project / "ingest/t1/words.json").read_text())["words"]
    rows = [(word["text"], word["start"], word["end"]) for word in words]
    root = make_project(tmp_path / "two-takes", [source, source], word_rows=[rows, rows])
    project = json.loads((root / "project.json").read_text())
    for take in project["takes"]:
        take["voiceover"] = True
    write_json(root / "project.json", project)
    write_edit(root, [time_segment("s1", "t1", 0, 5), time_segment("s2", "t2", 6, 12)])
    linked(root, monkeypatch)
    cut(root)
    baseline = synth.read_wav(root / "cut/voice.wav")[1].copy()
    levels = json.loads((root / "cut/receipt.json").read_text())["take_levels"]
    propose(root, "t1:2", "t1:3", "cold plunge")
    fix = make(root, ["f1"], yes=True, max_credits=20)["fixes"][0]
    decide(root, "f1", "approve", via="chat")
    cut(root)
    assert json.loads((root / "cut/receipt.json").read_text())["take_levels"] == levels
    actual = synth.read_wav(root / "cut/voice.wav")[1]
    first = fix["patch"]["start_sample"]
    last = first + fix["patch"]["samples"]
    np.testing.assert_array_equal(actual[:first], baseline[:first])
    np.testing.assert_array_equal(actual[last:], baseline[last:])


@pytest.mark.parametrize("peak", [0.56, 0.23])
def test_cut_refuses_patch_above_ceiling_after_take_gain(
    voice_project, fake_voice, monkeypatch, tmp_path, peak,
):
    from test_cut import make_project

    from yap_studio.stages import take_levels
    from yap_studio.voice import store
    from yap_studio.voice.model import VoiceError

    source = voice_project / "takes/t1.mov"
    words = json.loads((voice_project / "ingest/t1/words.json").read_text())["words"]
    rows = [(word["text"], word["start"], word["end"]) for word in words]
    root = make_project(tmp_path / "two-takes", [source, source], word_rows=[rows, rows])
    project = json.loads((root / "project.json").read_text())
    for take in project["takes"]:
        take["voiceover"] = True
    write_json(root / "project.json", project)
    write_edit(root, [time_segment("s1", "t1", 0, 5), time_segment("s2", "t2", 0, 5)])
    linked(root, monkeypatch)
    propose(root, "t2:2", "t2:3", "cold plunge")
    make(root, ["f1"], yes=True, max_credits=20)
    decide(root, "f1", "approve", via="chat")
    patches_for_plan = store.patches_for_plan

    def higher_peak(*args, **kwargs):
        patches = patches_for_plan(*args, **kwargs)
        for _, sound in patches:
            sound[len(sound) // 2] = peak
        return patches

    monkeypatch.setattr(store, "patches_for_plan", higher_peak)
    monkeypatch.setattr(take_levels, "measure_take", lambda sound, speech, work, name: {
        "seconds": len(sound) / 48000, "lufs": -10 if name == "t1" else -22,
        "peak_dbfs": -30, "basis": "words",
    })
    with pytest.raises(VoiceError) as caught:
        cut(root)
    assert str(caught.value) == (
        "Voice fix f1 would clip after this take's level match. "
        "Make it again with --new or keep the original."
    )
    assert not (root / "cut/voice.wav").exists()


@pytest.mark.parametrize("when", ["resolve", "audio"])
@pytest.mark.parametrize("approved", [True, False])
def test_cut_uses_one_voice_snapshot(
    voice_project, fake_voice, monkeypatch, when, approved,
):
    from yap_studio.stages import cut as cut_stage

    linked(voice_project, monkeypatch)
    propose(voice_project, "t1:2", "t1:3", "cold plunge")
    make(voice_project, ["f1"], yes=True, max_credits=20)
    decide(voice_project, "f1", "approve" if approved else "reject", via="chat")
    write_edit(voice_project, [{
        "id": "s1", "take": "t1", "from": {"word": "t1:2"}, "to": {"word": "t1:3"},
    }])
    cut(voice_project)
    expected = {name: (voice_project / "cut" / name).read_bytes()
                for name in ("voice.wav", "cut-words.json", "spans.json")}
    expected_inputs = cut_inputs(ProjectContext(voice_project))
    name = "_resolve" if when == "resolve" else "_audio_piece"
    original = getattr(cut_stage, name)
    changed = False

    def change_decision(*args, **kwargs):
        nonlocal changed
        if not changed:
            decide(voice_project, "f1", "reject" if approved else "approve", via="review")
            changed = True
        return original(*args, **kwargs)

    monkeypatch.setattr(cut_stage, name, change_decision)
    cut(voice_project, force=True)
    assert changed
    for name, data in expected.items():
        assert (voice_project / "cut" / name).read_bytes() == data, name
    record = json.loads((voice_project / "cut/receipt.json").read_text())
    assert record["inputs"] == expected_inputs
    assert bool(record["voice_fixes"]) == approved
    assert cut_inputs(ProjectContext(voice_project)) != expected_inputs
