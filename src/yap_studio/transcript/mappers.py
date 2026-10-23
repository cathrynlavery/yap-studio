"""Mappers from supported speech-engine transcript formats."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from yap_studio.errors import InputError, ValidationError

from .model import SCHEMA, Transcript, Word


def _minimum_confidence(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def from_parakeet(
    data: dict[str, Any],
    *,
    backend: str = "parakeet-mlx",
    model: str | None = None,
) -> Transcript:
    """Map sentence tokens from a Parakeet JSON result to words."""
    tokens: list[dict[str, Any]] = []
    for sentence in data.get("sentences", []):
        for token in sentence.get("tokens", []):
            confidence = token.get("confidence", 1.0)
            tokens.append(
                {
                    "text": str(token["text"]),
                    "start": float(token["start"]),
                    "end": float(token["end"]),
                    "conf": float(confidence) if confidence is not None else None,
                }
            )
    tokens.sort(key=lambda token: (token["start"], token["end"]))

    words: list[Word] = []
    for token in tokens:
        text = token["text"]
        if text.startswith(" ") or not words:
            words.append(
                Word(
                    text=text.strip(),
                    start=token["start"],
                    end=token["end"],
                    conf=token["conf"],
                )
            )
            continue
        previous = words[-1]
        words[-1] = Word(
            text=previous.text + text,
            start=previous.start,
            end=max(previous.end, token["end"]),
            conf=_minimum_confidence(previous.conf, token["conf"]),
        )
    return Transcript(
        words=words,
        language=data.get("language"),
        backend=backend,
        model=model,
        duration_s=data.get("duration_s"),
    )


def from_parakeet_tokens(
    tokens: Sequence[str],
    timestamps: Sequence[float],
    *,
    logprobs: Sequence[float] | None = None,
    frame_s: float = 0.08,
    duration_s: float | None = None,
    backend: str = "parakeet-onnx",
    model: str | None = None,
) -> Transcript:
    """Map token start timestamps from onnx-asr to word spans."""
    if len(tokens) != len(timestamps):
        raise ValidationError("tokens", "tokens and timestamps must have the same length")
    if logprobs is not None and len(tokens) != len(logprobs):
        raise ValidationError("logprobs", "tokens and logprobs must have the same length")

    groups: list[dict[str, Any]] = []
    for index, (raw_text, raw_start) in enumerate(zip(tokens, timestamps, strict=True)):
        text = str(raw_text)
        start = float(raw_start)
        logprob = float(logprobs[index]) if logprobs is not None else None
        if text.startswith(" ") or not groups:
            groups.append(
                {
                    "text": text.strip(),
                    "start": start,
                    "last_start": start,
                    "min_logprob": logprob,
                }
            )
        else:
            group = groups[-1]
            group["text"] += text
            group["last_start"] = start
            if logprob is not None:
                previous = group["min_logprob"]
                group["min_logprob"] = logprob if previous is None else min(previous, logprob)

    words: list[Word] = []
    for index, group in enumerate(groups):
        end = group["last_start"] + frame_s
        if index + 1 < len(groups):
            end = min(end, groups[index + 1]["start"])
        if duration_s is not None:
            end = min(end, duration_s)
        min_logprob = group["min_logprob"]
        words.append(
            Word(
                text=group["text"],
                start=group["start"],
                end=end,
                conf=math.exp(min_logprob) if min_logprob is not None else None,
            )
        )
    return Transcript(words=words, backend=backend, model=model, duration_s=duration_s)


def from_whisper(
    data: dict[str, Any],
    *,
    backend: str = "faster-whisper",
    model: str | None = None,
) -> Transcript:
    """Map word timestamps from a Whisper-family JSON result."""
    words: list[Word] = []
    for segment in data.get("segments", []):
        for item in segment.get("words", []):
            if item.get("start") is None or item.get("end") is None:
                continue
            confidence = item.get("probability", item.get("score"))
            words.append(
                Word(
                    text=str(item.get("word", "")).strip(),
                    start=float(item["start"]),
                    end=float(item["end"]),
                    conf=float(confidence) if confidence is not None else None,
                )
            )
    return Transcript(
        words=words,
        language=data.get("language"),
        backend=backend,
        model=model,
        duration_s=data.get("duration_s"),
    )


def load_any(path: str | Path) -> Transcript:
    """Load a transcript JSON file by detecting its supported shape."""
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(str(source), "unreadable transcript JSON") from exc
    if not isinstance(data, dict):
        raise InputError(str(source), "unrecognised transcript format")
    if data.get("schema") == SCHEMA:
        return Transcript.from_dict(data)
    if "sentences" in data:
        return from_parakeet(data)
    if "segments" in data:
        return from_whisper(data)
    if "tokens" in data and "timestamps" in data:
        return from_parakeet_tokens(
            data["tokens"],
            data["timestamps"],
            logprobs=data.get("logprobs"),
            duration_s=data.get("duration_s"),
        )
    raise InputError(str(source), "unrecognised transcript format")
