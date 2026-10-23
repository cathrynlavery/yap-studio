"""Synthetic transcript corrections through the CLI and display stages."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_cli_captions import fake_layers as fake_layers
from test_stage_captions import layer_project as layer_project
from test_stage_captions import set_elements, state
from test_stage_graphics import title
from test_timeline import write

from yap_studio import receipt
from yap_studio.captions import CaptionWord, build_phrases
from yap_studio.cli import main
from yap_studio.project.model import load_project
from yap_studio.project.status import ProjectContext, captions_inputs, graphics_inputs
from yap_studio.sfx import bleep
from yap_studio.stages.assemble import _bleep_words
from yap_studio.stages.captions import caption_snapshot, captions
from yap_studio.stages.graphics import graphics_snapshot


@pytest.fixture
def word_project(tmp_path: Path) -> Path:
    source = tmp_path / "source.mov"
    source.write_bytes(b"synthetic source")
    write(tmp_path / "project.json", {
        "schema": "yap-studio/project@1", "name": "sample",
        "takes": [{"id": "t1", "path": source.name}],
    })
    write(tmp_path / "ingest/t1/words.json", {
        "schema": "yap-studio/words@1", "take": "t1",
        "take_sha256": receipt.sha256_file(source), "time_base": "source",
        "audio_offset_s": 0, "engine": "none", "model": None,
        "transcript_sha256": "1" * 64,
        "words": [{"id": f"t1:{i}", "text": text, "start": i, "end": i + 0.5}
                  for i, text in enumerate(["small", "small."])],
    })
    return tmp_path


@pytest.mark.parametrize(("selector", "field", "key"), [
    ("--word", "word_text_fixes", "t1:0"),
    ("--heard", "word_fixes", "small"),
])
def test_cli_set_list_remove(word_project, capsys, selector, field, key) -> None:
    words_file = word_project / "ingest/t1/words.json"
    original = words_file.read_bytes()
    command = ["words", "fix", str(word_project)]
    assert main([*command, selector, key, "--to", "simple", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result[field] == {key: "simple"}
    assert getattr(load_project(word_project / "project.json"), field) == {key: "simple"}
    assert main([*command, "--list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[field] == {key: "simple"}
    assert main([*command, "--list"]) == 0
    assert f"{key} -> simple" in capsys.readouterr().out
    assert main([*command, selector, key, "--remove", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[field] == {}
    assert words_file.read_bytes() == original
    assert (word_project / "source.mov").read_bytes() == b"synthetic source"


def test_cli_lists_both_kinds_and_plain_mutations(word_project, capsys) -> None:
    command = ["words", "fix", str(word_project)]
    assert main([*command, "--word", "t1:0", "--to", "tiny"]) == 0
    assert "t1:0 -> tiny" in capsys.readouterr().out
    assert main([*command, "--heard", "small", "--to", "simple"]) == 0
    capsys.readouterr()
    assert main([*command, "--list", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["word_text_fixes"] == {"t1:0": "tiny"}
    assert result["word_fixes"] == {"small": "simple"}
    assert main([*command, "--word", "t1:0", "--remove"]) == 0
    assert "Removed" in capsys.readouterr().out


@pytest.mark.parametrize("identifier", ["t1:99", "t2:0", "../../:0", "bad"])
@pytest.mark.parametrize("operation", [["--to", "simple"], ["--remove"]])
def test_cli_refuses_unknown_word_without_writing(
    word_project, capsys, identifier, operation,
) -> None:
    project_file = word_project / "project.json"
    original = project_file.read_bytes()
    assert main(["words", "fix", str(word_project), "--word", identifier,
                 *operation, "--json"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]
    assert project_file.read_bytes() == original


@pytest.mark.parametrize("heard", ["SMALL", "Small,"])
def test_heard_update_and_remove_match_existing_spelling(word_project, capsys, heard) -> None:
    command = ["words", "fix", str(word_project)]
    assert main([*command, "--heard", "small", "--to", "simple", "--json"]) == 0
    capsys.readouterr()
    assert main([*command, "--heard", heard, "--to", "tiny", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["word_fixes"] == {"small": "tiny"}
    assert main([*command, "--heard", heard, "--remove", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["word_fixes"] == {}


@pytest.mark.parametrize("options", [
    [], ["--word", "t1:0"], ["--to", "simple"],
    ["--word", "t1:0", "--heard", "small", "--to", "simple"],
    ["--word", "t1:0", "--to", "   "],
    ["--heard", "   ", "--to", "simple"],
    ["--list", "--word", "t1:0"],
    ["--word", "t1:0", "--remove", "--to", "simple"],
])
def test_cli_refuses_invalid_selection_as_json(word_project, capsys, options) -> None:
    original = (word_project / "project.json").read_bytes()
    assert main(["words", "fix", str(word_project), *options, "--json"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]
    assert (word_project / "project.json").read_bytes() == original


@pytest.mark.parametrize(("replacement", "expected"), [
    ("captchas", "captchas,"), ("Captcha", "Captcha,"), ("captchas!", "captchas!"),
])
def test_occurrence_fix_keeps_timing_and_punctuation(replacement, expected) -> None:
    source = [CaptionWord("t1:0", "CAPTS,", 0.1, 0.4),
              CaptionWord("t1:1", "CAPTS.", 0.5, 0.8),
              CaptionWord("t1:2", "CAPTS!", 0.9, 1.2)]
    phrases = build_phrases(source, {"max_words_per_phrase": 8,
                                    "max_characters_per_phrase": 80},
                            word_fixes={"capts": "captchas"},
                            word_text_fixes={"t1:0": replacement}, text_rules=True)
    shown = [word for phrase in phrases for word in phrase.words]
    assert [word.text for word in shown] == [expected, "captchas.", "captchas!"]
    assert [(word.id, word.start, word.end) for word in shown] == [
        (word.id, word.start, word.end) for word in source]


def test_occurrence_fix_wins_over_overlapping_heard_phrase() -> None:
    source = [CaptionWord("t1:0", "blue", 0, 0.2),
              CaptionWord("t1:1", "bird", 0.3, 0.5)]
    phrases = build_phrases(source, {"max_words_per_phrase": 8,
                                    "max_characters_per_phrase": 80},
                            word_fixes={"blue bird": "birdsong", "finch": "robin"},
                            word_text_fixes={"t1:1": "finch"})
    assert phrases[0].text == "blue finch"


@pytest.mark.parametrize("camera", [False, True])
@pytest.mark.parametrize("selector", ["--word", "--heard"])
def test_fix_stales_captions_then_removal_restores_hash(
    layer_project, fake_layers, capsys, camera, selector,
) -> None:
    if camera:
        style = json.loads((layer_project / "style.json").read_bytes())
        style["captions"]["camera_3d"] = True
        write(layer_project / "style.json", style)
    set_elements(layer_project, title())
    before = captions_inputs(ProjectContext(layer_project))
    title_before = graphics_inputs(ProjectContext(layer_project))
    if not camera:
        captions(layer_project)
        assert state(layer_project, "captions")["state"] == "done"
    command = ["words", "fix", str(layer_project), selector,
               "t1:0" if selector == "--word" else "steps"]
    assert main([*command, "--to", "tiny", "--json"]) == 0
    capsys.readouterr()
    assert captions_inputs(ProjectContext(layer_project)) != before
    assert graphics_inputs(ProjectContext(layer_project)) == title_before
    if camera:
        snapshot = caption_snapshot(ProjectContext(layer_project))
        assert any(word["text"].startswith("tiny") for word in snapshot.words)
    if not camera:
        assert state(layer_project, "captions")["state"] == "stale"
        captions(layer_project)
        assert "tiny" in (layer_project / "captions/captions.srt").read_text()
    assert main([*command, "--remove", "--json"]) == 0
    capsys.readouterr()
    assert captions_inputs(ProjectContext(layer_project)) == before
    assert graphics_inputs(ProjectContext(layer_project)) == title_before


def test_fixed_swear_is_masked_in_phrases_srt_and_audio(layer_project, fake_layers, capsys):
    style = json.loads((layer_project / "style.json").read_bytes())
    style["music"] = {"bleep_swears": True}
    write(layer_project / "style.json", style)
    assert main(["words", "fix", str(layer_project), "--word", "t1:0",
                 "--to", "shit", "--json"]) == 0
    capsys.readouterr()
    snapshot = caption_snapshot(ProjectContext(layer_project))
    assert snapshot.phrases[0].words[0].text == "s***"
    captions(layer_project)
    assert "s***" in (layer_project / "captions/captions.srt").read_text()
    context = ProjectContext(layer_project)
    retained = SimpleNamespace(small_files={name: (layer_project / name).read_bytes()
                                           for name in ("project.json", "cut/cut-words.json")})
    found, = _bleep_words(context, retained)
    assert found.ids == ("t1:0",) and found.masked == ("s***",)


def test_bleep_occurrence_fixes_keep_source_indices_with_repeated_ids() -> None:
    source = [CaptionWord("t1:0", "ship", 0, 0.2),
              CaptionWord("t1:0", "ship", 1, 1.2)]
    found = bleep.find_display(source, bleep.listed({}), word_text_fixes={"t1:0": "shit"})
    assert [item.indices for item in found] == [(0,), (1,)]
    assert [item.ids for item in found] == [("t1:0",), ("t1:0",)]


def test_opening_title_matches_fixed_spoken_word(layer_project, capsys) -> None:
    assert main(["words", "fix", str(layer_project), "--word", "t1:0",
                 "--to", "Tiny", "--json"]) == 0
    capsys.readouterr()
    style = json.loads((layer_project / "style.json").read_bytes())
    style["hook"]["style"] = "kinetic"
    write(layer_project / "style.json", style)
    set_elements(layer_project, title(start={"time": 0}, lines=["Tiny steps"], hold_s=1.5))
    fixed = graphics_snapshot(ProjectContext(layer_project)).titles[0]
    assert "Tiny" in fixed.plan.variables["kinetic_lines"]
    assert json.loads(fixed.plan.variables["word_times"]) == [0.1, 0.3]
