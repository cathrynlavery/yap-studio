"""Portable transcripts, speech backends, and word comparison helpers."""

from .backends import (
    ENGINE_FAMILIES,
    ENGINES,
    Backend,
    FasterWhisperBackend,
    ParakeetMlxBackend,
    ParakeetOnnxBackend,
    ScriptedBackend,
    available_engines,
    resolve_backend,
    resolve_second_backend,
    second_engine_unmeasured_reason,
)
from .isolated import IsolatedBackend, SecondPassUnmeasured
from .mappers import from_parakeet, from_parakeet_tokens, from_whisper, load_any
from .model import SCHEMA, Transcript, Word
from .words import OPTIONAL, compare_words, norm_words, words_between

__all__ = [
    "ENGINE_FAMILIES",
    "ENGINES",
    "OPTIONAL",
    "SCHEMA",
    "Backend",
    "FasterWhisperBackend",
    "IsolatedBackend",
    "ParakeetMlxBackend",
    "ParakeetOnnxBackend",
    "ScriptedBackend",
    "SecondPassUnmeasured",
    "Transcript",
    "Word",
    "available_engines",
    "compare_words",
    "from_parakeet",
    "from_parakeet_tokens",
    "from_whisper",
    "load_any",
    "norm_words",
    "resolve_backend",
    "resolve_second_backend",
    "second_engine_unmeasured_reason",
    "words_between",
]
