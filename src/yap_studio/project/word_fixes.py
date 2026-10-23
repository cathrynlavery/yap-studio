"""Shared display-fix mutation for the CLI and review page."""

from pathlib import Path

from yap_studio import paths
from yap_studio.captions import _stripped
from yap_studio.errors import ValidationError
from yap_studio.project.model import load_words, parse_document


def change_word_fix(root: Path, data: dict, *, word: str | None = None,
                    heard: str | None = None, to: str | None = None,
                    remove: bool = False) -> tuple[str, str]:
    """Validate and change project data under the caller's project lock."""
    project = parse_document("project", data)
    if (word is None) == (heard is None) or remove == (to is not None):
        raise ValidationError("words fix", "choose a word or heard text, then replace or remove")
    if any(value is not None and not value.strip() for value in (heard, to)):
        raise ValidationError("words fix", "heard and replacement text must not be blank")
    if word is not None:
        take = word.split(":")[0]
        if take not in {item.id for item in project.takes}:
            raise ValidationError("word", f"unknown word id: {word}")
        words = load_words(paths.validate_input(root / f"ingest/{take}/words.json"))
        if word not in {item.id for item in words.words}:
            raise ValidationError("word", f"unknown word id: {word}")
    field = "word_text_fixes" if word is not None else "word_fixes"
    key = word if word is not None else heard.strip()
    fixes = data.setdefault(field, {})
    if heard is not None:
        normalized = tuple(_stripped(part) for part in key.split())
        key = next((existing for existing in fixes if tuple(
            _stripped(part) for part in existing.split()) == normalized), key)
    if remove:
        fixes.pop(key, None)
    else:
        fixes[key] = to.strip()
    parse_document("project", data)
    return field, key
