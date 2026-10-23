"""Canonical word-timestamp transcript model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yap_studio.errors import ValidationError
from yap_studio.paths import atomic_write_text

SCHEMA = "yap-studio/transcript@1"


@dataclass(frozen=True)
class Word:
    """One timestamped word in a transcript."""

    text: str
    start: float
    end: float
    conf: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible word representation."""
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "conf": self.conf,
        }


@dataclass
class Transcript:
    """A portable transcript with word-level timestamps."""

    words: list[Word]
    language: str | None = None
    backend: str | None = None
    model: str | None = None
    duration_s: float | None = None

    @property
    def text(self) -> str:
        """Return transcript words joined with single spaces."""
        return " ".join(word.text for word in self.words)

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned JSON representation."""
        return {
            "schema": SCHEMA,
            "language": self.language,
            "backend": self.backend,
            "model": self.model,
            "duration_s": self.duration_s,
            "words": [word.to_dict() for word in self.words],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transcript:
        """Build a transcript from its versioned JSON representation."""
        schema = data.get("schema")
        if schema != SCHEMA:
            raise ValidationError("schema", f"expected {SCHEMA!r}, got {schema!r}")
        words = [
            Word(
                text=str(item["text"]),
                start=float(item["start"]),
                end=float(item["end"]),
                conf=float(item["conf"]) if item.get("conf") is not None else None,
            )
            for item in data.get("words", [])
        ]
        return cls(
            words=words,
            language=data.get("language"),
            backend=data.get("backend"),
            model=data.get("model"),
            duration_s=(
                float(data["duration_s"]) if data.get("duration_s") is not None else None
            ),
        )

    def save(self, path: str | Path) -> Path:
        """Save the transcript atomically as UTF-8 JSON."""
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"
        return atomic_write_text(path, text)

    @classmethod
    def load(cls, path: str | Path) -> Transcript:
        """Load a versioned transcript JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
