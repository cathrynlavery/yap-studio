"""Require main CI and Release checks on the exact checked out release commit."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    import os
    import shlex

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(
        "scripts/check_release_ci.py needs Python 3.11 or newer. "
        f"Run it with: cd {shlex.quote(root)} && uv run python scripts/check_release_ci.py"
    )

import json
import subprocess


def require_ci(pages: list[dict], commit: str, name: str, event: str) -> None:
    runs = [run for page in pages for run in page["workflow_runs"]
            if run["head_sha"] == commit and run["head_branch"] == "main"
            and run["event"] == event]
    latest = max(runs, key=lambda run: run["id"], default=None)
    if latest is None or latest["status"] != "completed" or latest["conclusion"] != "success":
        result = "missing run" if latest is None else f'{latest["status"]}/{latest["conclusion"]}'
        raise ValueError(f"{name} must pass on the tagged commit before release. "
                         f"Latest run: {result}. Rerun after checks pass. "
                         "Start release checks with: gh workflow run release-checks.yml --ref main")


def main() -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for workflow, event, name in (
        ("ci.yml", "push", "Main CI"),
        ("release-checks.yml", "workflow_dispatch", "Release checks"),
    ):
        result = subprocess.run(
            ["gh", "api", "--paginate", "--slurp",
             f"repos/{{owner}}/{{repo}}/actions/workflows/{workflow}/runs"
             f"?head_sha={commit}&branch=main&event={event}&per_page=100"],
            capture_output=True, text=True, check=True, timeout=60,
        )
        require_ci(json.loads(result.stdout), commit, name, event)
    print(f"Main CI and Release checks passed on {commit}.")


if __name__ == "__main__":
    main()
