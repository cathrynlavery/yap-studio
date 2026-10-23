"""Word normalisation and transcript-window comparison."""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Literal

from .model import Transcript, Word

NUMBER_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "11": "eleven",
    "12": "twelve",
    "15": "fifteen",
    "20": "twenty",
    "30": "thirty",
    "100": "hundred",
}
EQUIV = {
    "kinda": ["kind", "of"],
    "gonna": ["going", "to"],
    "wanna": ["want", "to"],
    "ok": ["okay"],
}
OPTIONAL = frozenset({"uh", "um", "uhm", "umm", "mm", "erm", "hmm"})


def norm_words(text: str) -> list[str]:
    """Normalise text to the word vocabulary used by join checks."""
    normalized = re.sub(r"[^a-z0-9' ]", " ", text.lower().replace("-", " "))
    output: list[str] = []
    for word in normalized.split():
        word = word.strip("'")
        if not word:
            continue
        if word in NUMBER_WORDS:
            output.append(NUMBER_WORDS[word])
        else:
            output.extend(EQUIV.get(word, [word]))
    return output


def title_words(text: str) -> list[str]:
    """Read dollar and percent signs as words, then apply norm_words."""
    text = re.sub(r"\$(\d+(?:[,.]\d+)*)", r"\1 dollars", text)
    return norm_words(text.replace("%", " percent"))


def unicode_words(text: str, *, drop_apostrophes: bool = False) -> list[str]:
    """Normalise Unicode word runs while retaining their combining marks."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    result: list[str] = []
    current: list[str] = []
    for char in normalized:
        if drop_apostrophes and char in {"'", "’"}:
            continue
        if char.isalnum():
            current.append(char)
        elif unicodedata.category(char).startswith("M"):
            if not current:
                return []
            current.append(char)
        elif current:
            word = "".join(current)
            result.append(NUMBER_WORDS.get(word, word))
            current = []
    if current:
        word = "".join(current)
        result.append(NUMBER_WORDS.get(word, word))
    return result


def compare_words(
    expected: list[str],
    got: list[str],
    *,
    fillers_optional: bool = False,
) -> dict[str, object]:
    """Compare normalised words, tolerating partial words only at window edges."""
    if fillers_optional:
        expected = [word for word in expected if word not in OPTIONAL]
        got = [word for word in got if word not in OPTIONAL]
    matcher = difflib.SequenceMatcher(None, expected, got)
    operations = [operation for operation in matcher.get_opcodes() if operation[0] != "equal"]
    edge_only = bool(operations) and all(
        (
            (i1 == 0 and j1 == 0 and i2 <= 1)
            or (i1 >= len(expected) - 1 and j2 == len(got))
        )
        and (i2 - i1) <= 1
        and (j2 - j1) <= 2
        for _, i1, i2, j1, j2 in operations
    )
    return {
        "exact": expected == got,
        "core_exact": expected == got or edge_only,
        "ratio": round(matcher.ratio(), 4),
        "diff": [
            f"{operation} exp{expected[i1:i2]} got{got[j1:j2]}"
            for operation, i1, i2, j1, j2 in operations
        ],
    }


def words_between(
    transcript: Transcript,
    t0: float,
    t1: float,
    rule: Literal["midpoint", "start", "overlap"] = "midpoint",
) -> list[Word]:
    """Select words intersecting a half-open transcript time window."""
    if rule == "midpoint":
        return [word for word in transcript.words if t0 <= (word.start + word.end) / 2 < t1]
    if rule == "start":
        return [word for word in transcript.words if t0 <= word.start < t1]
    if rule == "overlap":
        return [word for word in transcript.words if word.start < t1 and word.end > t0]
    raise ValueError(f"unknown word selection rule: {rule}")
