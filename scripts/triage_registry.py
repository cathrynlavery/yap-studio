"""Classify and test-render every HyperFrames registry item at the pinned release."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    import os
    import shlex

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(
        "scripts/triage_registry.py needs Python 3.11 or newer. "
        f"Run it with: cd {shlex.quote(root)} && uv run python scripts/triage_registry.py"
    )

import argparse
import json
import tempfile
import time
from pathlib import Path

from yap_studio import paths
from yap_studio.errors import YapError
from yap_studio.hf_registry import catalog, install
from yap_studio.hf_registry.triage import check, document, dumps, survey
from yap_studio.render import hyperframes
from yap_studio.render.hyperframes import HYPERFRAMES_VERSION

ROOT = Path(__file__).resolve().parents[1]
TRIAGE = ROOT / "src/yap_studio/hf_registry/triage.json"


def main() -> int:
    """Survey and render the pinned registry, then write its triage atomically."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stills", type=Path, metavar="DIR",
                        help="write one PNG still per rendered item")
    args = parser.parse_args()
    started = time.monotonic()
    try:
        renderer = hyperframes.ensure_installed(
            notice=lambda message: print(message, file=sys.stderr),
        )
        with tempfile.TemporaryDirectory(prefix="hf-registry-triage-") as temporary:
            work = Path(temporary)
            records = survey(
                catalog.registry_url(), executable=renderer.executable, work=work,
                home=paths.validate_output_dir(install.registry_home()),
            )
            check(records, install=renderer, work=work, stills=args.stills)
            paths.atomic_write_text(TRIAGE, dumps(document(
                records, commit=catalog.load_catalog()["registry"]["commit"],
                version=HYPERFRAMES_VERSION,
            )))
    except (YapError, OSError, ValueError) as exc:
        print(exc.message if isinstance(exc, YapError) else str(exc), file=sys.stderr)
        action = exc.suggested_action if isinstance(exc, YapError) else None
        print(action or "Check the registry and run the triage again.", file=sys.stderr)
        return 1
    print(json.dumps({
        "items": len(records),
        "usable": sum(record["use"] != "none" for record in records.values()),
        "passed": sum(record.get("check", {}).get("status") == "pass"
                      for record in records.values()),
        "failed": sum(record.get("check", {}).get("status") == "fail"
                      for record in records.values()),
        "seconds": round(time.monotonic() - started, 1),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
