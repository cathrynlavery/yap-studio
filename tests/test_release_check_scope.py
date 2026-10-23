"""Release scope remembers actual suite executions, including across skipped runs."""

import json
import subprocess
from unittest.mock import Mock

import pytest
from test_workspace import ROOT, _module


@pytest.fixture
def scope():
    return _module(ROOT / "scripts/release_check_scope.py", "release_scope")


def test_baselines_ignore_skips_failed_runs_and_other_branches(scope, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "7")
    run = {"id": 1, "head_branch": "main", "head_sha": "old",
           "status": "completed", "conclusion": "success"}
    runs = [run, dict(run, id=2, head_sha="speech"), dict(run, id=3, head_sha="render"),
            dict(run, id=4, conclusion="failure"), dict(run, id=5, head_branch="feature"),
            dict(run, id=6, status="in_progress"), dict(run, id=7)]
    jobs = Mock(side_effect=lambda run_id: [
        {"name": "render (1)", "conclusion": "success" if run_id in (1, 3) else "skipped"},
        {"name": "render (2)", "conclusion": "success" if run_id in (1, 3) else "skipped"},
        {"name": "speech", "conclusion": "success" if run_id in (1, 2) else "skipped"},
        {"name": "helpers", "conclusion": "success" if run_id == 1 else "skipped"},
    ])
    assert scope.latest_baselines(runs, jobs, list(scope.SUITES)) == {
        "render": "render", "speech": "speech", "helpers": "old",
    }
    assert [call.args[0] for call in jobs.call_args_list] == [3, 2, 1]


def test_missing_or_partially_skipped_suite_has_no_baseline(scope):
    run = {"id": 1, "head_branch": "main", "head_sha": "old",
           "status": "completed", "conclusion": "success"}
    assert scope.latest_baselines([run], lambda _: [
        {"name": "render (1)", "conclusion": "success"},
        {"name": "render (2)", "conclusion": "skipped"},
    ], list(scope.SUITES)) == {}


@pytest.mark.parametrize("suite,path", [
    ("render", "src/yap_studio/templates/nested/card.html"),
    ("render", "src/yap_studio/render/renderer-lock/package-lock.json"),
    ("render", "src/yap_studio/captions.py"),
    ("render", "tests/test_letters_render.py"),
    ("speech", "src/yap_studio/transcript/backends.py"),
    ("speech", "tests/test_cut.py"),
    ("helpers", "src/yap_studio/faces/face_boxes.swift"),
    ("helpers", "src/yap_studio/joins/helper.py"),
    ("helpers", "tests/test_hdr.py"),
])
def test_path_map_matches_suite_but_not_docs(scope, suite, path):
    patterns = json.loads((ROOT / ".github/release-check-paths.json").read_text())
    assert scope.matching_files(["README.md", path, path], patterns[suite]) == [path]


@pytest.mark.parametrize("scenario,expected", [
    ("docs", "render=false\nspeech=false\nhelpers=false\n"),
    ("render", "render=true\nspeech=false\nhelpers=false\n"),
    ("missing", "render=true\nspeech=true\nhelpers=true\n"),
    ("unavailable", "render=true\nspeech=true\nhelpers=true\n"),
    ("forced", "render=true\nspeech=false\nhelpers=true\n"),
    ("schedule", "render=true\nspeech=true\nhelpers=true\n"),
])
def test_scope_outputs_and_summary(scope, monkeypatch, tmp_path, scenario, expected):
    output, summary = tmp_path / "output", tmp_path / "summary"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_SHA", "head")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule" if scenario == "schedule"
                       else "workflow_dispatch")
    for suite in scope.SUITES:
        monkeypatch.setenv(f"{suite.upper()}_MODE", "auto")
    if scenario in ("forced", "schedule"):
        for suite, mode in zip(scope.SUITES, ("full", "skip", "full"), strict=True):
            monkeypatch.setenv(f"{suite.upper()}_MODE", mode)
    api = Mock(return_value=[])
    monkeypatch.setattr(scope, "api_rows", api)
    monkeypatch.setattr(scope, "latest_baselines", Mock(return_value=(
        {} if scenario == "missing" else dict.fromkeys(scope.SUITES, "base")
    )))
    diff = Mock(return_value=["README.md", "src/yap_studio/captions.py"]
                if scenario == "render" else ["README.md"])
    if scenario == "unavailable":
        diff.side_effect = subprocess.CalledProcessError(128, "git")
    monkeypatch.setattr(scope, "changed_files", diff)
    scope.main()
    assert output.read_text() == expected
    report = summary.read_text()
    for suite in scope.SUITES:
        assert f"### {suite}:" in report
    if scenario == "render":
        assert "- src/yap_studio/captions.py" in report
        assert "README.md" not in report
    if scenario in ("forced", "schedule"):
        api.assert_not_called()
        diff.assert_not_called()
    elif scenario != "missing":
        diff.assert_called_once_with("base", "head")


def test_api_walks_all_pages_and_rejects_incomplete_history(scope, monkeypatch):
    pages = [{"total_count": 2, "jobs": [{"id": 1}]},
             {"total_count": 2, "jobs": [{"id": 2}]}]
    run = Mock(return_value=subprocess.CompletedProcess([], 0, json.dumps(pages)))
    monkeypatch.setattr(scope.subprocess, "run", run)
    assert scope.api_rows("endpoint", "jobs") == [{"id": 1}, {"id": 2}]
    assert run.call_args.args[0] == ["gh", "api", "--paginate", "--slurp", "endpoint"]
    run.return_value.stdout = json.dumps(pages[:1])
    with pytest.raises(ValueError, match="Incomplete"):
        scope.api_rows("endpoint", "jobs")


def test_diff_includes_deleted_and_renamed_paths(scope, monkeypatch):
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "old.py\0new.py\0"))
    monkeypatch.setattr(scope.subprocess, "run", run)
    assert scope.changed_files("base", "head") == ["old.py", "new.py"]
    assert run.call_args.args[0] == [
        "git", "diff", "--name-only", "--no-renames", "-z", "base", "head", "--",
    ]
