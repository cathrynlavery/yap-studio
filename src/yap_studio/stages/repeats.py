"""Find repeated phrases and points in the retained script."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from yap_studio.project.model import RULE_DEFAULTS
from yap_studio.text_rules import STOP_WORDS


def _token(text: str) -> str:
    return "".join(char for char in text.lower() if char.isalnum())


_STOP = frozenset(_token(word) for word in STOP_WORDS) | {
    "it", "like", "yeah", "think", "know", "mean",
}


@dataclass(frozen=True)
class RepeatWord:
    """One retained word occurrence on the cut clock."""

    id: str
    text: str
    time_s: float
    end_s: float
    segment_id: str = ""


@dataclass(frozen=True)
class _Sentence:
    words: list[RepeatWord]
    tokens: list[tuple[str, RepeatWord]]
    content: set[str]


def _sentences(words: Sequence[RepeatWord]) -> list[_Sentence]:
    groups: list[list[RepeatWord]] = []
    pending: list[RepeatWord] = []
    for word in words:
        # Punctuation keeps a sentence whole only between breaths and story edges.
        if pending and (word.segment_id != pending[-1].segment_id
                        or word.time_s - pending[-1].end_s
                        >= RULE_DEFAULTS["beat_breath_s"] - 1e-9):
            groups.append(pending)
            pending = []
        pending.append(word)
        if word.text.rstrip("\"')]}’”»").endswith((".", "?", "!")):
            groups.append(pending)
            pending = []
    if pending:
        groups.append(pending)
    result = []
    for group in groups:
        tokens = [(token, word) for word in group if (token := _token(word.text))]
        result.append(_Sentence(group, tokens, {token for token, _ in tokens} - _STOP))
    return result


def _phrase(left: _Sentence, right: _Sentence) -> tuple[int, int, int]:
    """Return the longest qualifying run's token offsets and length."""
    prefix = [0]
    for token, _ in left.tokens:
        prefix.append(prefix[-1] + (token not in _STOP))
    previous = [0] * (len(right.tokens) + 1)
    best = (0, 0, 0)
    for i, (token, _) in enumerate(left.tokens):
        current = [0] * len(previous)
        for j, (other, _) in enumerate(right.tokens):
            if token != other:
                continue
            length = current[j + 1] = previous[j] + 1
            first, second = i + 1 - length, j + 1 - length
            if length < 3 or length <= best[2] or prefix[i + 1] - prefix[first] < 2:
                continue
            best = (first, second, length)
        previous = current
    return best


def _occurrence(sentence: _Sentence, time_s: float) -> dict[str, Any]:
    return {"time_s": time_s, "sentence": " ".join(word.text for word in sentence.words),
            "words": [word.id for word in sentence.words]}


def find_repeats(words: Sequence[RepeatWord]) -> list[dict[str, Any]]:
    """Report one repeat per sentence pair, without changing the retained story."""
    result = []
    for left, right in combinations(_sentences(words), 2):
        first, second, length = _phrase(left, right)
        if length:
            if right.tokens[second][1].time_s - left.tokens[first][1].time_s < 3:
                continue
            result.append({
                "kind": "phrase",
                "shared": " ".join(token for token, _ in left.tokens[first:first + length]),
                "first": _occurrence(left, left.tokens[first][1].time_s),
                "second": _occurrence(right, right.tokens[second][1].time_s),
            })
            continue
        shared = left.content & right.content
        if (len(shared) >= 3 and len(shared) * 2 >= min(len(left.content), len(right.content))
                and right.words[0].time_s - left.words[0].time_s >= 3):
            result.append({
                "kind": "point", "shared": sorted(shared),
                "first": _occurrence(left, left.words[0].time_s),
                "second": _occurrence(right, right.words[0].time_s),
            })
    return result


def _time(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}:{seconds:02d}"


def repeat_warnings(repeats: Sequence[dict[str, Any]]) -> list[str]:
    """Format at most eight warnings and the count of remaining sentence pairs."""
    warnings = []
    for repeat in repeats[:8]:
        first, second = repeat["first"], repeat["second"]
        if repeat["kind"] == "phrase":
            warnings.append(
                f'Repeated: "{repeat["shared"]}" at {_time(first["time_s"])} and again at '
                f'{_time(second["time_s"])} ("{second["sentence"]}"). '
                "Keep it once unless the repeat is on purpose."
            )
        else:
            shared = ", ".join(f'"{word}"' for word in repeat["shared"])
            warnings.append(
                f'Same point twice: {_time(first["time_s"])} and {_time(second["time_s"])} '
                f'share {shared}. Keep the clearer one unless the repeat is on purpose.'
            )
    if len(repeats) > 8:
        warnings.append(f"{len(repeats) - 8} more repeated sentence pairs.")
    return warnings
