"""Generate conservative glyph advance tables from the bundled normal font faces."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    import os
    import shlex

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(
        "scripts/gen_text_metrics.py needs Python 3.11 or newer. "
        f"Run it with: cd {shlex.quote(root)} && uv run python scripts/gen_text_metrics.py"
    )

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yap_studio.render.fonts import BUNDLED, FONTS_DIR  # noqa: E402

WEIGHTS = (400, 600, 700)
EXTRA_CHARACTERS = (
    "\u00a0\u00b0\u00b1\u00d7\u00f7"
    "\u00c0\u00c1\u00c4\u00c5\u00c6\u00c7\u00c8\u00c9\u00ca\u00cb"
    "\u00cd\u00ce\u00cf\u00d1\u00d3\u00d6\u00d8\u00da\u00dc\u00df"
    "\u00e0\u00e1\u00e4\u00e5\u00e6\u00e7\u00e8\u00e9\u00ea\u00eb"
    "\u00ed\u00ee\u00ef\u00f1\u00f3\u00f6\u00f8\u00fa\u00fc\u00ff"
    "\u0152\u0153\u2018\u2019\u201c\u201d\u2013\u2014\u2026\u20ac"
)
CHARACTERS = "".join(chr(code) for code in range(0x20, 0x7F)) + EXTRA_CHARACTERS


def _advances(font) -> dict[str, float | dict[str, float] | list[list[int]]]:
    cmap = font.getBestCmap()
    if not cmap:
        raise ValueError("a bundled font has no Unicode cmap")
    units_per_em = font["head"].unitsPerEm
    metrics = font["hmtx"].metrics
    table = {
        char: (metrics[cmap[ord(char)]][0] * 10000 + units_per_em - 1) // units_per_em / 10000
        for char in CHARACTERS if ord(char) in cmap
    }
    sampled_max = max(table.values())
    widest = max(advance for advance, _ in metrics.values())
    maximum = (widest * 10000 + units_per_em - 1) // units_per_em / 10000
    coverage: list[list[int]] = []
    blocks: dict[str, float] = {}
    for codepoint in sorted(cmap):
        block = str(codepoint // 64)
        advance = metrics[cmap[codepoint]][0]
        rounded = (advance * 10000 + units_per_em - 1) // units_per_em / 10000
        blocks[block] = max(blocks.get(block, 0.0), rounded)
        if coverage and codepoint == coverage[-1][1] + 1:
            coverage[-1][1] = codepoint
        else:
            coverage.append([codepoint, codepoint])
    # Preserve sampled advances; coverage decides when the browser uses a fallback face.
    return {char: table.get(char, sampled_max) for char in CHARACTERS} | {
        "_max": maximum, "_coverage": coverage, "_blocks": blocks,
    }


def main() -> int:
    """Write metrics, or check the committed bytes without changing them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "src/yap_studio/render/text_metrics.json"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    families = {}
    for name, bundled in BUNDLED.items():
        tables = {}
        for face in bundled.faces:
            if face.style != "normal":
                continue
            with TTFont(FONTS_DIR / face.file) as font:
                if "fvar" in font:
                    axis = next(axis for axis in font["fvar"].axes if axis.axisTag == "wght")
                    for weight in WEIGHTS:
                        if axis.minValue <= weight <= axis.maxValue:
                            with instantiateVariableFont(font, {"wght": weight}) as instance:
                                tables[str(weight)] = _advances(instance)
                else:
                    tables[str(font["OS/2"].usWeightClass)] = _advances(font)
        families[name] = tables
    expected = (
        json.dumps({"schema": "yap-studio/text-metrics@1", "fonts": families},
                   sort_keys=True, indent=1) + "\n"
    ).encode()
    if args.check:
        if not args.out.is_file() or args.out.read_bytes() != expected:
            print(f"Text metrics differ: {args.out.name}", file=sys.stderr)
            return 1
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
