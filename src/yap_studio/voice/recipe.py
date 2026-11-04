"""Creator-owned clone settings with neutral defaults."""

from __future__ import annotations

import copy
import json
import re

from yap_studio.review.store import data_dir
from yap_studio.voice.model import VoiceError, _require

SCHEMA = "yap-studio/voice-recipe@1"
WORD = re.compile(r"\w+(?:['’]\w+)*")
DEFAULTS = {
    "model_id": "eleven_multilingual_v2", "voice_settings": None, "voice_name": None,
    "tag": "", "strong_tag": "", "strong_tag_words": [], "context": True,
    "pace": "fit", "pronunciations": [],
}


def validate_recipe(data: dict) -> dict:
    """Validate every supplied setting before a request can be paid for."""
    try:
        json.dumps(data, allow_nan=False)
        _require(isinstance(data, dict) and data["schema"] == SCHEMA)
        _require(not set(data) - {"schema", *DEFAULTS})
        result = {**copy.deepcopy(DEFAULTS), **{k: v for k, v in data.items() if k != "schema"}}
        _require(isinstance(result["model_id"], str) and result["model_id"].strip())
        _require(result["voice_name"] is None or (
            isinstance(result["voice_name"], str) and result["voice_name"].strip()))
        _require(result["voice_settings"] is None or isinstance(result["voice_settings"], dict))
        _require(isinstance(result["tag"], str) and isinstance(result["strong_tag"], str))
        _require(type(result["context"]) is bool and result["pace"] in ("fit", "off"))
        _require(isinstance(result["strong_tag_words"], list))
        _require(all(isinstance(word, str) and WORD.fullmatch(word)
                     for word in result["strong_tag_words"]))
        _require(isinstance(result["pronunciations"], list))
        seen = set()
        for row in result["pronunciations"]:
            _require(isinstance(row, dict) and set(row) == {"word", "type", "value"})
            _require(isinstance(row["word"], str) and WORD.fullmatch(row["word"]))
            word = row["word"].casefold()
            _require(word not in seen)
            seen.add(word)
            _require(row["type"] in ("ipa", "alias"))
            _require(isinstance(row["value"], str) and row["value"].strip())
        return result
    except (AssertionError, KeyError, TypeError, ValueError):
        raise VoiceError(
            "The voice recipe is not valid. "
            "Check voice/recipe.json in your Yap Studio data folder.",
            code="voice_recipe_invalid",
        ) from None


def load_recipe() -> dict:
    """Read only the creator's data folder, never a project or repository recipe."""
    target = data_dir() / "voice/recipe.json"
    try:
        data = json.loads(target.read_bytes())
    except FileNotFoundError:
        data = {"schema": SCHEMA}
    except (OSError, ValueError):
        raise VoiceError("The voice recipe could not be read.",
                         code="voice_recipe_invalid") from None
    return validate_recipe(data)


def speech_text(say: str, previous: str, recipe: dict, *, plain: bool = False) -> str:
    """Apply whole-word pronunciations without respelling a sentence's first word."""
    if plain:
        return say
    rules = {row["word"].casefold(): row for row in recipe["pronunciations"]}
    first = re.search(r"\w", say)
    initial = not previous.strip() or previous.rstrip().endswith((".", "!", "?"))

    def spelling(match):
        rule = rules.get(match.group().casefold())
        if rule is None or initial and first is not None and match.start() == first.start():
            return match.group()
        return f"/{rule['value']}/" if rule["type"] == "ipa" else rule["value"]

    text = WORD.sub(spelling, say)
    strong = {word.casefold() for word in recipe["strong_tag_words"]}
    spoken = {match.group().casefold() for match in WORD.finditer(say)}
    tag = (recipe["strong_tag"] if recipe["strong_tag"] and strong.intersection(spoken)
           else recipe["tag"])
    return f"{tag.strip()} {text}" if tag.strip() else text
