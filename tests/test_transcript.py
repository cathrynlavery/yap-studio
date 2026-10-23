"""Transcript model and speech-engine format mapper tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from yap_studio.errors import InputError, ValidationError
from yap_studio.transcript import (
    SCHEMA,
    Transcript,
    Word,
    from_parakeet,
    from_parakeet_tokens,
    from_whisper,
    load_any,
)


def _parakeet_data() -> dict[str, object]:
    return {
        "text": "Bright planets.",
        "sentences": [
            {
                "text": "Bright planets.",
                "start": 0.08,
                "end": 1.04,
                "duration": 0.96,
                "confidence": 0.91,
                "tokens": [
                    {"text": " plan", "start": 0.56, "end": 0.72, "confidence": 0.98},
                    {"text": " B", "start": 0.08, "end": 0.24, "confidence": 0.95},
                    {"text": "rig", "start": 0.24, "end": 0.32, "confidence": 1.0},
                    {"text": "ht", "start": 0.32, "end": 0.40, "confidence": 0.93},
                    {"text": "ets", "start": 0.72, "end": 1.04, "confidence": 0.96},
                ],
            }
        ],
    }


def test_transcript_round_trip(tmp_path: Path) -> None:
    transcript = Transcript(
        words=[Word("hello", 0.1, 0.4, 0.98), Word("world", 0.5, 0.9)],
        language="en",
        backend="scripted",
        model="fixture",
        duration_s=1.0,
    )
    path = transcript.save(tmp_path / "transcript.json")
    assert path == tmp_path / "transcript.json"
    assert json.loads(path.read_text())["schema"] == SCHEMA
    assert Transcript.load(path) == transcript
    assert transcript.text == "hello world"


def test_wrong_schema_is_rejected() -> None:
    with pytest.raises(ValidationError, match="schema"):
        Transcript.from_dict({"schema": "yap-studio/transcript@0", "words": []})


def test_from_parakeet_joins_sorted_tokens_and_keeps_minimum_confidence() -> None:
    transcript = from_parakeet(_parakeet_data(), model="fixture")
    assert transcript.backend == "parakeet-mlx"
    assert transcript.model == "fixture"
    assert transcript.words == [
        Word("Bright", 0.08, 0.40, 0.93),
        Word("planets", 0.56, 1.04, 0.96),
    ]


def test_from_parakeet_tokens_joins_tokens_and_maps_logprob() -> None:
    transcript = from_parakeet_tokens(
        [" B", "rig", "ht", " plan", "ets"],
        [0.0, 0.24, 0.32, 0.56, 0.88],
        logprobs=[-0.1, -0.2, -0.15, -0.05, -0.3],
        duration_s=1.0,
        model="fixture",
    )
    assert transcript.words[0] == Word("Bright", 0.0, 0.4, pytest.approx(math.exp(-0.2)))
    assert transcript.words[1] == Word("planets", 0.56, 0.96, pytest.approx(math.exp(-0.3)))
    assert transcript.duration_s == 1.0


def test_from_parakeet_tokens_caps_ends() -> None:
    transcript = from_parakeet_tokens(
        [" A", " B", " C"],
        [0.95, 1.0, 1.08],
        frame_s=0.08,
        duration_s=1.12,
    )
    assert [word.end for word in transcript.words] == pytest.approx([1.0, 1.08, 1.12])


@pytest.mark.parametrize(
    ("tokens", "timestamps", "logprobs"),
    [([" A"], [], None), ([" A"], [0.0], []), ([" A", " B"], [0.0], [-0.1])],
)
def test_from_parakeet_tokens_rejects_mismatched_lengths(
    tokens: list[str], timestamps: list[float], logprobs: list[float] | None
) -> None:
    with pytest.raises(ValidationError):
        from_parakeet_tokens(tokens, timestamps, logprobs=logprobs)


def test_from_whisper_accepts_probability_and_skips_missing_timing() -> None:
    transcript = from_whisper(
        {
            "language": "en",
            "segments": [
                {
                    "words": [
                        {"word": " Bright", "start": 0.1, "end": 0.5, "probability": 0.9},
                        {"word": " ignored", "start": None, "end": 0.8},
                    ]
                }
            ],
        }
    )
    assert transcript.words == [Word("Bright", 0.1, 0.5, 0.9)]
    assert transcript.language == "en"


def test_from_whisper_accepts_whisperx_score() -> None:
    transcript = from_whisper(
        {"segments": [{"words": [{"word": "quiet", "start": 1, "end": 2, "score": 0.8}]}]}
    )
    assert transcript.words == [Word("quiet", 1.0, 2.0, 0.8)]


@pytest.mark.parametrize("shape", ["canonical", "parakeet", "whisper", "onnx"])
def test_load_any_detects_supported_shapes(tmp_path: Path, shape: str) -> None:
    canonical = Transcript([Word("bright", 0.0, 0.4)]).to_dict()
    samples = {
        "canonical": canonical,
        "parakeet": _parakeet_data(),
        "whisper": {
            "segments": [{"words": [{"word": " bright", "start": 0.0, "end": 0.4}]}]
        },
        "onnx": {"tokens": [" bright"], "timestamps": [0.0], "duration_s": 0.4},
    }
    path = tmp_path / f"{shape}.json"
    path.write_text(json.dumps(samples[shape]))
    assert load_any(path).words[0].text.lower() == "bright"


def test_load_any_rejects_unknown_shape(tmp_path: Path) -> None:
    path = tmp_path / "unknown.json"
    path.write_text('{"text": "nothing timestamped"}')
    with pytest.raises(InputError, match="unrecognised transcript format"):
        load_any(path)
