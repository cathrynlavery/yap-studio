"""Measure card text using the committed bundled font advance tables."""

from __future__ import annotations

import json
import math
from functools import cache
from pathlib import Path
from typing import TypedDict

# Conservative browser fallback bound: full-width CJK is 1 em, wide Cyrillic about 1.12 em.
FALLBACK_EM = 1.25
UNBUNDLED_MARGIN = 1.15
# Installed monospace cells are commonly about 0.6 em, even for narrow letters such as i.
UNBUNDLED_MIN_EM = 0.7
# Newly bundled fonts must not move text in existing styles.
UNBUNDLED_BOUND_FAMILIES = (
    "Alata", "Anton SC", "Archivo Black", "Gelasio", "Instrument Serif", "Inter",
    "Permanent Marker", "Space Mono", "Titan One",
)


class _FontTable(TypedDict):
    advances: dict[str, float]
    maximum: float
    coverage: list[list[int]]
    blocks: dict[str, float]


@cache
def _metrics() -> dict[str, dict[str, _FontTable]]:
    families = json.loads(
        Path(__file__).with_suffix(".json").read_text(encoding="utf-8")
    )["fonts"]
    return {
        family: {
            weight: _FontTable(
                advances={char: advance for char, advance in table.items() if len(char) == 1},
                maximum=table["_max"], coverage=table["_coverage"],
                blocks=table.get("_blocks", {}),
            )
            for weight, table in tables.items()
        }
        for family, tables in families.items()
    }


def _nearest(tables: dict[str, _FontTable], weight: int) -> _FontTable:
    return tables[min(tables, key=lambda key: (abs(int(key) - weight), -int(key)))]


@cache
def _tables(family: str, weight: int) -> tuple[_FontTable, ...]:
    metrics = _metrics()
    if family in metrics:
        return (_nearest(metrics[family], weight),)
    return tuple(_nearest(metrics[name], weight) for name in UNBUNDLED_BOUND_FAMILIES)


def _covered(table: _FontTable, char: str) -> bool:
    codepoint = ord(char)
    return any(start <= codepoint <= end for start, end in table["coverage"])


def _advance(table: _FontTable, char: str) -> float:
    if not _covered(table, char):
        return FALLBACK_EM
    block_maximum = table["blocks"].get(str(ord(char) // 64), table["maximum"])
    return table["advances"].get(char, block_maximum)


def text_width_em(text: str, family: str, weight: int, *, tracking_em: float = 0.0) -> float:
    """Sum covered glyph advances, conservative fallback bounds, and CSS tracking.

    Installed fonts cannot be measured at plan time. Bound each character using the
    original bundled faces, including monospace, with a margin and minimum cell width. The
    template's browser overflow guard provides the final fit guarantee.
    """
    tables = _tables(family, weight)
    if family in _metrics():
        advances = (_advance(tables[0], char) for char in text)
    else:
        advances = (
            max(UNBUNDLED_MIN_EM, max(_advance(table, char) for table in tables)
                * UNBUNDLED_MARGIN)
            if any(_covered(table, char) for table in tables) else FALLBACK_EM
            for char in text
        )
    return math.fsum(advances) + len(text) * tracking_em


def widest_digit_em(family: str, weight: int) -> float:
    """Return the widest digit's advance for a fixed reel window."""
    return max(text_width_em(char, family, weight) for char in "0123456789")
