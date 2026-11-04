"""Creator recipes shape clone requests without changing the displayed words."""

import json
from importlib import import_module

import pytest
from test_voice_make import prepared
from voice_helpers import calls

from yap_studio import cli
from yap_studio.errors import YapError
from yap_studio.voice import consent, link, make, propose


@pytest.fixture
def recipe_file(tmp_path, monkeypatch):
    module = import_module("yap_studio.voice.recipe")
    monkeypatch.setattr(module, "data_dir", lambda: tmp_path / "creator")
    target = tmp_path / "creator/voice/recipe.json"
    target.parent.mkdir(parents=True)
    return target


def save_recipe(target, **fields):
    target.write_text(json.dumps({"schema": "yap-studio/voice-recipe@1", **fields}))


def speech(log):
    return [args for args in calls(log) if args[0] == "text-to-speech"]


def test_neutral_recipe_keeps_plain_request(voice_project, fake_voice, monkeypatch, recipe_file):
    prepared(voice_project, monkeypatch)
    make(voice_project, ["f1"], yes=True, max_credits=100)
    args, = speech(fake_voice)
    assert args[args.index("--text") + 1] == "cold plunge"
    assert args[args.index("--model-id") + 1] == "eleven_multilingual_v2"
    assert "--previous-text" in args and "--next-text" in args
    assert "--voice-settings" not in args


@pytest.mark.parametrize("say,previous,expected", [
    ("cold house", "before", "[strong test] /test-ipa/ test-alias"),
    ("cold plunge", "before", "[test accent] /test-ipa/ plunge"),
    ("Cold house", "before.", "[strong test] Cold test-alias"),
    ("Cold house", "", "[strong test] Cold test-alias"),
    ("cold household", "before", "[test accent] /test-ipa/ household"),
])
def test_recipe_text_rules(recipe_file, say, previous, expected):
    module = import_module("yap_studio.voice.recipe")
    save_recipe(recipe_file, tag="[test accent]", strong_tag="[strong test]",
                strong_tag_words=["house"], pronunciations=[
                    {"word": "cold", "type": "ipa", "value": "test-ipa"},
                    {"word": "house", "type": "alias", "value": "test-alias"},
                ])
    assert module.speech_text(say, previous, module.load_recipe()) == expected


def test_recipe_matches_literal_words(recipe_file):
    module = import_module("yap_studio.voice.recipe")
    save_recipe(recipe_file, tag="[test accent]", strong_tag="[strong test]",
                strong_tag_words=["ok"], pronunciations=[
                    {"word": "ok", "type": "alias", "value": "test-alias"},
                ])
    recipe = module.load_recipe()
    assert module.speech_text("okay", "before", recipe) == "[test accent] okay"
    assert module.speech_text("okay OK", "before", recipe) == "[strong test] okay test-alias"
    save_recipe(recipe_file, pronunciations=[
        {"word": "gonna", "type": "alias", "value": "test-alias"},
    ])
    assert module.speech_text("gonna", "before", module.load_recipe()) == "test-alias"


def test_recipe_argv_cost_cache_and_plain(voice_project, fake_voice, monkeypatch, recipe_file):
    save_recipe(recipe_file, model_id="test-model", tag="[test accent]", context=False,
                voice_settings={"stability": 0.5}, pronunciations=[
                    {"word": "plunge", "type": "alias", "value": "test-alias"},
                ])
    prepared(voice_project, monkeypatch)
    preview = make(voice_project, ["f1"], dry_run=True)
    sent = "[test accent] cold test-alias"
    assert preview["cost"]["characters"] == len(sent)
    assert preview["cost"]["credits"] == len(sent)
    assert "test-alias" not in " ".join(preview["cost"]["lines"])
    first = make(voice_project, ["f1"], yes=True, max_credits=100)["fixes"][0]
    args, = speech(fake_voice)
    assert args[args.index("--text") + 1] == sent
    assert args[args.index("--model-id") + 1] == "test-model"
    assert json.loads(args[args.index("--voice-settings") + 1]) == {"stability": 0.5}
    assert "--previous-text" not in args and "--next-text" not in args
    assert first["say"] == "cold plunge"
    save_recipe(recipe_file, model_id="test-model", tag="[changed test]", context=False)
    second = make(voice_project, ["f1"], yes=True, max_credits=100)["fixes"][0]
    assert second["generation"]["request_sha256"] != first["generation"]["request_sha256"]
    assert second["generation"]["seed"] == first["generation"]["seed"]
    assert len(speech(fake_voice)) == 2
    assert cli.main(["voice", "make", str(voice_project), "--fix", "f1", "--plain",
                     "--yes", "--max-credits", "100"]) == 0
    last = speech(fake_voice)[-1]
    assert last[last.index("--text") + 1] == "cold plunge"
    assert last[last.index("--seed") + 1] == args[args.index("--seed") + 1]


@pytest.mark.parametrize("metadata", ["saved", "missing", "other_request"])
def test_cached_charge_belongs_to_its_recipe(
    voice_project, fake_voice, monkeypatch, recipe_file, metadata,
):
    prepared(voice_project, monkeypatch)
    generations = []
    for tag, charged in (("[test accent]", "7"), ("[changed test]", "13")):
        save_recipe(recipe_file, tag=tag)
        monkeypatch.setenv("FAKE_ELEVENLABS_CHARGED", charged)
        monkeypatch.setenv("FAKE_ELEVENLABS_SPEECH_SECONDS", "1.4" if charged == "7" else "1.3")
        generations.append(make(voice_project, ["f1"], yes=True, max_credits=100)["fixes"][0])
    first, second = (fix["generation"] for fix in generations)
    assert first["cost"] == {"credits": 7, "source": "history"}
    assert second["cost"] == {"credits": 13, "source": "history"}
    assert first["request_sha256"] != second["request_sha256"]
    assert first["output_sha256"] != second["output_sha256"]
    cached_charge = voice_project / f".yap/cache/voice/{first['request_sha256']}.json"
    entry = json.loads(cached_charge.read_text())
    assert entry["request_sha256"] == first["request_sha256"]
    assert entry["cost"] == first["cost"]
    if metadata == "missing":
        cached_charge.unlink(missing_ok=True)
    elif metadata == "other_request":
        cached_charge.write_text(json.dumps({**entry, "request_sha256": second["request_sha256"],
                                            "cost": second["cost"]}))
    save_recipe(recipe_file, tag="[test accent]")
    latest = make(voice_project, ["f1"], yes=True, max_credits=100)["fixes"][0]
    assert latest["generation"]["request_sha256"] == first["request_sha256"]
    assert latest["generation"]["output_sha256"] == first["output_sha256"]
    expected = first["cost"] if metadata == "saved" else {
        "credits": latest["estimate"]["credits"], "source": "estimate",
    }
    assert latest["generation"]["cost"] == expected
    assert len(speech(fake_voice)) == 2


@pytest.mark.parametrize("fields", [
    {"context": "yes"}, {"pace": "fast"}, {"model_id": ""}, {"voice_settings": []},
    {"pronunciations": [{"word": "word", "type": "unknown", "value": "test"}]},
    {"strong_tag_words": "house"}, {"unknown": True},
])
def test_bad_recipe_is_refused(voice_project, fake_voice, recipe_file, fields):
    save_recipe(recipe_file, **fields)
    with pytest.raises(YapError, match="recipe") as caught:
        propose(voice_project, "t1:2", "t1:3", "cold plunge", clone=True)
    assert caught.value.code == "voice_recipe_invalid"
    assert speech(fake_voice) == []


def test_recipe_voice_name_defaults(voice_project, fake_voice, monkeypatch, recipe_file):
    from voice_helpers import linked

    linked(voice_project, monkeypatch)
    save_recipe(recipe_file, voice_name="Test voice")
    assert consent(voice_project, yes=True)["voice"]["name"] == "Test voice"
    monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_NAME", "Test voice")
    assert link(voice_project, yes=True)["voice"]["name"] == "Test voice"


def test_respelled_warning_and_review_use_plain_words(
    voice_project, fake_voice, monkeypatch, recipe_file,
):
    from yap_studio.review.voice_fixes import review_fixes
    from yap_studio.stages import cut as cut_stage
    from yap_studio.transcript import ScriptedBackend
    from yap_studio.transcript.model import Transcript, Word

    save_recipe(recipe_file, tag="[test accent]", pronunciations=[
        {"word": "plunge", "type": "alias", "value": "test-alias"},
    ])
    prepared(voice_project, monkeypatch)
    backend = ScriptedBackend(lambda _: Transcript(words=[Word("different", 0.1, 0.8)]))
    monkeypatch.setattr(cut_stage, "_edge_backend", lambda _: backend)
    fix = make(voice_project, ["f1"], yes=True, max_credits=100)["fixes"][0]
    heard = next(item for item in fix["checks"]["items"] if item["id"] == "splice.heard")
    assert heard["status"] == "warn" and "--plain" in heard["message"]
    assert heard["expected"] == "cold plunge"
    review, = review_fixes(voice_project, [])
    assert review["say"] == "cold plunge"
    assert "test-alias" not in json.dumps(review)
    assert "[test accent]" not in json.dumps(review)
