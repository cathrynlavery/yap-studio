"""Generate the project contracts from the engine's declarative validators."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    import os
    import shlex

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(
        "scripts/gen_project_schemas.py needs Python 3.11 or newer. "
        f"Run it with: cd {shlex.quote(root)} && uv run python scripts/gen_project_schemas.py"
    )

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from yap_studio.project.model import SCHEMAS_DIR, SPECS, json_schema  # noqa: E402


def main() -> int:
    """Write schemas, or check the committed bytes without changing them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=SCHEMAS_DIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    differs = []
    for name in SPECS:
        path = args.out_dir / f"{name}.schema.json"
        expected = (json.dumps(json_schema(name), sort_keys=True, indent=2) + "\n").encode()
        if args.check:
            if not path.is_file() or path.read_bytes() != expected:
                differs.append(path.name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(expected)
    if differs:
        print("Schemas differ: " + ", ".join(differs), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
