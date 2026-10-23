"""Select release suites against their latest successful execution on main."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    import os
    import shlex

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(
        "scripts/release_check_scope.py needs Python 3.11 or newer. "
        f"Run it with: cd {shlex.quote(root)} && uv run python scripts/release_check_scope.py"
    )

import json
import os
import subprocess
from collections.abc import Callable
from fnmatch import fnmatchcase
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("render", "speech", "helpers")


def api_rows(endpoint: str, key: str) -> list[dict]:
    result = subprocess.run(
        ["gh", "api", "--paginate", "--slurp", endpoint],
        capture_output=True, text=True, check=True, timeout=120,
    )
    pages = json.loads(result.stdout)
    rows = [row for page in pages for row in page[key]]
    if len(rows) != pages[0]["total_count"]:
        raise ValueError(f"Incomplete Actions API response for {key}")
    return rows


def latest_baselines(
    runs: list[dict], jobs_for: Callable[[int], list[dict]], suites: list[str],
) -> dict[str, str]:
    baselines = {}
    for run in sorted(runs, key=lambda run: run["id"], reverse=True):
        if (run["head_branch"] != "main" or run["status"] != "completed"
                or run["conclusion"] != "success"
                or str(run["id"]) == os.environ.get("GITHUB_RUN_ID")):
            continue
        jobs = jobs_for(run["id"])
        for suite in suites:
            matched = [job for job in jobs
                       if job["name"] == suite or job["name"].startswith(f"{suite} (")]
            if suite not in baselines and matched and all(
                job["conclusion"] == "success" for job in matched
            ):
                baselines[suite] = run["head_sha"]
        if len(baselines) == len(suites):
            break
    return baselines


def matching_files(changed: list[str], patterns: list[str]) -> list[str]:
    return sorted({name for name in changed if any(fnmatchcase(name, glob) for glob in patterns)})


def changed_files(base: str, head: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", "-z", base, head, "--"],
        cwd=ROOT, capture_output=True, text=True, check=True, timeout=60,
    )
    return [name for name in result.stdout.split("\0") if name]


def main() -> None:
    patterns = json.loads((ROOT / ".github/release-check-paths.json").read_text())
    modes = {suite: os.environ.get(f"{suite.upper()}_MODE", "auto") for suite in SUITES}
    if os.environ.get("GITHUB_EVENT_NAME") == "schedule":
        modes = dict.fromkeys(SUITES, "full")
    if any(mode not in {"auto", "full", "skip"} for mode in modes.values()):
        raise ValueError("Suite modes must be auto, full or skip")
    automatic = [suite for suite in SUITES if modes[suite] == "auto"]
    baselines = {}
    if automatic:
        repo = os.environ["GITHUB_REPOSITORY"]
        # Unfiltered history avoids the API's 1,000-result cap on filtered run searches.
        runs = api_rows(
            f"repos/{repo}/actions/workflows/release-checks.yml/runs?per_page=100",
            "workflow_runs",
        )
        baselines = latest_baselines(runs, lambda run_id: api_rows(
            f"repos/{repo}/actions/runs/{run_id}/jobs?filter=latest&per_page=100", "jobs",
        ), automatic)
    summary = ["## Release check scope", ""]
    outputs = []
    diffs = {}
    for suite, mode in modes.items():
        matched = []
        if mode != "auto":
            enabled, reason = mode == "full", f"{mode} requested"
        elif suite not in baselines:
            enabled, reason = True, "no successful main run that executed this suite"
        else:
            base = baselines[suite]
            if base not in diffs:
                try:
                    diffs[base] = changed_files(base, os.environ["GITHUB_SHA"])
                except subprocess.CalledProcessError:
                    diffs[base] = None
            if diffs[base] is None:
                enabled, reason = True, f"baseline {base} unavailable; running full"
            else:
                matched = matching_files(diffs[base], patterns[suite])
                enabled = bool(matched)
                reason = f"{len(matched)} matching files since {base}"
        outputs.append(f"{suite}={str(enabled).lower()}")
        summary.append(f"### {suite}: {'run' if enabled else 'skip'} ({reason})")
        summary.extend(f"- {name}" for name in matched)
        summary.append("")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write("\n".join(outputs) + "\n")
    report = "\n".join(summary)
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as output:
        output.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
