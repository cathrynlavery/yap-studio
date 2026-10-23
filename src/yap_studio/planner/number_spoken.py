"""Check a number card against speech near its resolved cut window."""

from __future__ import annotations

import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation

from yap_studio.project.model import CutWord
from yap_studio.text_rules import number_merges

SPEECH_LEAD_S = 0.5  # Include the number spoken just before a card enters.
FPS = 30  # Timeline windows use the delivery frame clock.
_NUMBER = re.compile(r"\$?(-?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)(%)?")


def spoken_number_fix(slots: dict, words: Sequence[CutWord], start: int, end: int) -> str | None:
    """Return a heard-number fix unless the target value is spoken in this window."""
    tokens = [token for word in words
              if word.end > start / FPS - SPEECH_LEAD_S and word.start < end / FPS
              for token in word.text.split()]
    replacements = {first: (last, text) for first, last, text in number_merges(tokens)}
    heard = []
    index = 0
    while index < len(tokens):
        last, text = replacements.get(index, (index + 1, tokens[index]))
        # The general display rule keeps bare "one" as a word; it is still spoken here.
        if text.casefold().strip(".,!?") == "one":
            text = "1"
        text = text.strip(".,!?;:\"'()[]")
        match = _NUMBER.fullmatch(text)
        if match:
            value = Decimal(match[1].replace(",", ""))
            if index and tokens[index - 1].casefold() in {"minus", "negative"}:
                value = -value
                text = "-" + text
            heard.append((value, bool(match[2]), text))
        index = last
    try:
        target = Decimal(str(slots.get("end")))
    except InvalidOperation:
        target = None
    percent = "%" in str(slots.get("suffix", ""))
    if any(value == target and (not percent or spoken_percent)
           for value, spoken_percent, _ in heard):
        return None
    listing = ", ".join(dict.fromkeys(text for _, _, text in heard)) or "none"
    return (f"use a number she says here (heard: {listing}), or anchor the card to the line "
            "that says it")
