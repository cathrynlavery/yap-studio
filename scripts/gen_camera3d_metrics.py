"""Write the camera pilot's font advances and line-box baselines at build time."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    import os
    import shlex

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(
        "scripts/gen_camera3d_metrics.py needs Python 3.11 or newer. "
        f"Run it with: cd {shlex.quote(root)} && uv run python scripts/gen_camera3d_metrics.py"
    )

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "src/yap_studio/render/text_metrics.json"
DEST = ROOT / "src/yap_studio/templates/camera3d/metrics.json"
FONTS = ROOT / "src/yap_studio/fonts"


def baseline(path: Path) -> float:
    from fontTools.ttLib import TTFont

    with TTFont(path) as font:
        upm = font["head"].unitsPerEm
        os2, hhea = font["OS/2"], font["hhea"]
        if os2.fsSelection & (1 << 7):
            ascent, descent = os2.sTypoAscender, os2.sTypoDescender
        else:
            ascent, descent = hhea.ascent, hhea.descent
        a, d = ascent / upm, -descent / upm
        return round((1 - a - d) / 2 + a, 5)


def main() -> None:
    source = json.loads(SOURCE.read_text())
    result = {"adv": {}, "base": {}}
    for key, family, weight, file in (
        ("caption", "Inter", "600", "inter-variable.woff2"),
        ("hero", "Instrument Serif", "400", "instrument-serif-regular.woff2"),
    ):
        table = source["fonts"][family][weight]
        result["adv"][key] = {char: value for char, value in table.items() if len(char) == 1}
        result["base"][key] = baseline(FONTS / file)
    DEST.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
