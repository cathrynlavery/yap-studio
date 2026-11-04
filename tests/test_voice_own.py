"""Recorded words are the first source for a voice fix."""

import json
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import synth
from test_cut import make_project, replace_take, time_segment, write_edit, write_json
from voice_helpers import calls

from yap_studio import cli
from yap_studio.errors import YapError
from yap_studio.project.receipts import StageRun
from yap_studio.project.status import ProjectContext, ingest_inputs
from yap_studio.stages.cut import cut, resolve_cut
from yap_studio.voice import decide, make, propose


@pytest.fixture
def own_project(tmp_path):
    source = synth.voiced_source(
        tmp_path / "source.mov", [(i + 0.2, i + 0.8) for i in range(12)], seconds=12,
    )
    words = "bright morning coal flunge quiet garden water before after cold plunge light".split()
    rows = [(word, i + 0.2, i + 0.8) for i, word in enumerate(words)]
    root = make_project(tmp_path / "project", [source, source], word_rows=[rows, rows],
                        ingest_manifest=True)
    project = json.loads((root / "project.json").read_text())
    project["takes"][0]["voiceover"] = True
    write_json(root / "project.json", project)
    write_edit(root, [time_segment("s1", "t1", 0, 6)])
    return root


def change_words(root, changes):
    inputs = ingest_inputs(ProjectContext(root))
    with StageRun(root, "ingest") as run:
        for name, value in inputs.items():
            run.record_inputs(name, value)
        for take in ("t1", "t2"):
            data = json.loads((root / f"ingest/{take}/words.json").read_text())
            for word in data["words"]:
                word.update(changes.get(word["id"], {}))
            staged = root / f"changed-{take}.json"
            write_json(staged, data)
            run.replace_output(root / f"ingest/{take}/words.json", staged)
        run.finish()
    edit = json.loads((root / "edit.json").read_text())
    write_edit(root, edit["segments"])


def test_finds_same_take_outside_cut_without_account(own_project, fake_voice):
    fix = propose(own_project, "t1:2", "t1:3", "Cold, plunge!")
    assert fix["from"]["kind"] == "own"
    assert fix["from"]["take"] == "t1"
    assert fix["from"]["words"] == ["t1:9", "t1:10"]
    assert fix["from"]["start_s"] == 9.2
    assert fix["from"]["text"] == "cold plunge"
    assert fix["source"] == {"start_s": 2.2, "end_s": 3.8}
    assert fix["estimate"]["credits"] == 0
    assert calls(fake_voice) == []


def test_audio_identical_standin_can_supply_recorded_words(own_project, fake_voice):
    words = (own_project / "ingest/t1/words.json").read_bytes()
    replace_take(own_project / "takes/t1.mov", copy_audio=True)
    assert "identical audio stream" in " ".join(resolve_cut(own_project).warnings)
    fix = propose(own_project, "t1:2", "t1:3", "cold plunge")
    assert fix["from"]["take"] == "t1"
    assert fix["from"]["words"] == ["t1:9", "t1:10"]
    assert make(own_project, ["f1"])["fixes"][0]["status"] == "ready"
    assert (own_project / "ingest/t1/words.json").read_bytes() == words
    assert calls(fake_voice) == []


def test_finds_other_take_not_used_by_cut(own_project, fake_voice):
    change_words(own_project, {"t1:9": {"conf": 0.69}})
    fix = propose(own_project, "t1:2", "t1:3", "cold plunge")
    assert fix["from"]["take"] == "t2"
    assert fix["from"]["words"] == ["t2:9", "t2:10"]
    assert calls(fake_voice) == []


def test_late_starting_mono_donor_can_supply_a_proposal(tmp_path, fake_voice):
    source = synth.voiced_source(
        tmp_path / "source.mov", [(0.2, 0.8), (1.2, 1.8), (2.2, 2.8)], seconds=4,
    )
    donor = synth.watch_audio_offset(tmp_path / "donor.mov", source, offset_s=0.2)
    rows = [("coal", 0.4, 1), ("cold", 1.4, 2), ("light", 2.4, 3)]
    root = make_project(tmp_path / "project", [donor], word_rows=[rows])
    project = json.loads((root / "project.json").read_text())
    project["takes"][0]["voiceover"] = True
    write_json(root / "project.json", project)
    write_edit(root, [time_segment("s1", "t1", 0, 4)])
    fix = propose(root, "t1:0", "t1:0", "cold")
    assert fix["from"]["kind"] == "own"
    assert fix["from"]["words"] == ["t1:1"]
    assert calls(fake_voice) == []


def test_candidates_rank_take_confidence_then_distance(own_project, fake_voice):
    change_words(own_project, {
        "t1:4": {"text": "cold", "conf": 0.8}, "t1:5": {"text": "plunge", "conf": 0.8},
        "t1:9": {"conf": 0.9}, "t1:10": {"conf": 0.9},
    })
    fix = propose(own_project, "t1:2", "t1:3", "cold plunge")
    assert fix["from"]["words"] == ["t1:9", "t1:10"]
    (own_project / "voice/fixes.json").unlink()
    change_words(own_project, {"t1:4": {"conf": 0.9}, "t1:5": {"conf": 0.9}})
    assert propose(own_project, "t1:2", "t1:3", "cold plunge")["from"]["words"] == [
        "t1:4", "t1:5",
    ]


def test_clipped_candidate_is_skipped(own_project, fake_voice, monkeypatch):
    module = import_module("yap_studio.voice.own")
    decode = module.audio.decode

    def clipped(path, **kwargs):
        rate, sound = decode(path, **kwargs)
        if path.name == "t1.mov":
            sound[round(9.5 * rate)] = 1
        return rate, sound

    monkeypatch.setattr(module.audio, "decode", clipped)
    assert propose(own_project, "t1:2", "t1:3", "cold plunge")["from"]["take"] == "t2"


@pytest.mark.parametrize("duration", [0.9, 2.0, 3.0])
def test_automatic_candidates_must_fit_before_selection(own_project, fake_voice, duration):
    changes = {}
    for take in ("t1", "t2"):
        changes.update({
            f"{take}:3": {"end": 3.2},
            f"{take}:4": {"text": "cold", "start": 4.2, "end": 4.5},
            f"{take}:5": {"text": "plunge", "start": 8.4, "end": 8.7},
            f"{take}:9": {"start": 9.0, "end": 9.3, "conf": 0.8},
            f"{take}:10": {"start": 9.5, "end": 9.0 + duration, "conf": 0.8},
            f"{take}:11": {"start": 12.1, "end": 12.2},
        })
    change_words(own_project, changes)
    fix = propose(own_project, "t1:2", "t1:3", "cold plunge")
    if duration == 3.0:
        assert fix["from"]["kind"] == "clone"
    else:
        assert fix["from"]["words"] == ["t1:9", "t1:10"]
        assert make(own_project, ["f1"])["fixes"][0]["status"] == "ready"
    assert not any(argv[0] == "text-to-speech" for argv in calls(fake_voice))


def test_chosen_words_must_say_the_replacement(own_project, fake_voice):
    with pytest.raises(YapError, match="do not say"):
        propose(own_project, "t1:2", "t1:3", "cold plunge",
                from_take_words=("t1:4", "t1:5"))
    assert calls(fake_voice) == []


def test_chosen_own_words_and_explicit_clone(own_project, fake_voice):
    fix = propose(own_project, "t1:2", "t1:3", "cold plunge",
                  from_take_words=("t2:9", "t2:10"))
    assert fix["from"]["take"] == "t2"
    (own_project / "voice/fixes.json").unlink()
    assert propose(own_project, "t1:2", "t1:3", "cold plunge", clone=True)["from"][
        "kind"
    ] == "clone"


def test_no_clean_words_falls_back_to_clone(own_project, fake_voice, monkeypatch):
    from voice_helpers import linked

    linked(own_project, monkeypatch)
    change_words(own_project, {"t1:9": {"conf": None}, "t2:10": {"conf": 0.2}})
    assert propose(own_project, "t1:2", "t1:3", "cold plunge")["from"]["kind"] == "clone"
    make(own_project, ["f1"], yes=True, max_credits=20)
    assert sum(argv[0] == "text-to-speech" for argv in calls(fake_voice)) == 1


def test_other_fix_words_cannot_supply_audio(own_project, fake_voice):
    write_edit(own_project, [time_segment("s1", "t1", 0, 12)])
    propose(own_project, "t1:9", "t1:10", "bright morning")
    write_edit(own_project, [time_segment("s1", "t1", 0, 6)])
    fix = propose(own_project, "t1:2", "t1:3", "cold plunge")
    assert fix["from"]["take"] == "t2"


def test_changed_donor_refuses_make(own_project, fake_voice):
    propose(own_project, "t1:2", "t1:3", "cold plunge")
    change_words(own_project, {"t1:9": {"text": "changed"}})
    with pytest.raises(YapError, match="do not say"):
        make(own_project, ["f1"])
    assert calls(fake_voice) == []


def test_unknown_word_placeholders_do_not_match(monkeypatch):
    module = import_module("yap_studio.voice.own")
    word = SimpleNamespace(id="t1:9", text="字", conf=1, start=1, end=2)
    snapshot = SimpleNamespace(words=[word], take_sha256="a" * 64)
    context = SimpleNamespace(project=SimpleNamespace(takes=[SimpleNamespace(id="t1")]))
    monkeypatch.setattr(module, "take_words", lambda *_: snapshot)
    monkeypatch.setattr(module, "take_sound", lambda *_: (
        np.full((3 * 48000, 1), 0.1, dtype=np.float32), 0,
    ))
    fix = {"id": "f1", "take": "t1", "words": ["t1:0"], "say": "別",
           "source": {"start_s": 0, "end_s": 0.5}}
    assert module.find_own(SimpleNamespace(context=context), fix, []) is None


def test_normalised_phrase_can_use_more_source_words(monkeypatch):
    module = import_module("yap_studio.voice.own")
    words = [SimpleNamespace(id=f"t1:{i}", text=text, conf=1,
                             start=1 + i / 10, end=1 + (i + 1) / 10)
             for i, text in enumerate("going to going to going to going to going to".split())]
    snapshot = SimpleNamespace(words=words, take_sha256="a" * 64)
    context = SimpleNamespace(project=SimpleNamespace(takes=[SimpleNamespace(id="t1")]))
    monkeypatch.setattr(module, "take_words", lambda *_: snapshot)
    monkeypatch.setattr(module, "take_sound", lambda *_: (
        np.full((3 * 48000, 1), 0.1, dtype=np.float32), 0,
    ))
    monkeypatch.setattr(import_module("yap_studio.voice.make"), "take_audio", lambda *_:
                        np.full((3 * 48000, 1), 0.1, dtype=np.float32))
    target = SimpleNamespace(id="t1:99", start=0, end=1)
    plan = SimpleNamespace(context=context, source_words={
        "t1": SimpleNamespace(words=[target, *words]),
    })
    fix = {"id": "f1", "take": "t1", "words": ["t1:99"], "say": "gonna " * 5,
           "source": {"start_s": 0, "end_s": 1}}
    origin = module.find_own(plan, fix, [])
    assert origin is not None and origin["words"] == [word.id for word in words]


def test_fix_words_cannot_supply_their_own_audio(own_project, fake_voice):
    assert propose(own_project, "t1:2", "t1:3", "coal flunge")["from"]["take"] == "t2"


def test_own_make_approve_cut_without_consent_or_paid_call(own_project, fake_voice):
    propose(own_project, "t1:2", "t1:3", "cold plunge")
    fix = make(own_project, ["f1"])["fixes"][0]
    assert fix["generation"]["cost"]["credits"] == 0
    assert {item["id"] for item in fix["checks"]["items"]} == {
        "splice.click", "splice.level", "splice.room_tone", "splice.pace", "splice.heard",
    }
    decide(own_project, "f1", "approve", via="chat")
    cut(own_project)
    for name in ("receipt.json", "verify.json"):
        record = json.loads((own_project / f"cut/{name}").read_text())
        assert record["voice_fixes"][0]["kind"] == "own"
    assert not (own_project / "voice/consent.json").exists()
    assert calls(fake_voice) == []


def test_own_source_still_needs_picture_cover(own_project, fake_voice):
    project = json.loads((own_project / "project.json").read_text())
    project["takes"][0]["voiceover"] = False
    write_json(own_project / "project.json", project)
    with pytest.raises(YapError) as caught:
        propose(own_project, "t1:2", "t1:3", "cold plunge")
    assert caught.value.code == "voice_mouth_on_screen"
    assert calls(fake_voice) == []


def test_cli_own_source_and_list(own_project, fake_voice, capsys):
    assert cli.main(["voice", "propose", str(own_project), "--from", "t1:2", "--to", "t1:3",
                     "--say", "cold plunge", "--from-take-words", "t2:9", "t2:10"]) == 0
    assert "from your own recording at 0:09" in capsys.readouterr().out
    assert cli.main(["voice", "list", str(own_project)]) == 0
    assert "own" in capsys.readouterr().out
    assert cli.main(["voice", "make", str(own_project), "--fix", "f1"]) == 0
    assert calls(fake_voice) == []
