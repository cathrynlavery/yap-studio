"""CI release gates and isolated workspace acceptance with synthetic fixtures."""

from __future__ import annotations

import json
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_workspace import ROOT, _build_setup_fixture, _module


def test_missing_node_fails_javascript_tests(monkeypatch) -> None:
    import test_review_js

    monkeypatch.setattr(test_review_js, "NODE", None)
    with pytest.raises(AssertionError, match="Node is required"):
        test_review_js.run_node("--version")


def test_ci_cannot_skip_the_browser_pin_flow(monkeypatch) -> None:
    monkeypatch.setenv("CI", "true")
    monkeypatch.delenv("YAP_STUDIO_PUPPETEER", raising=False)
    monkeypatch.delenv("YAP_STUDIO_CHROME", raising=False)
    module = _module(ROOT / "tests/test_review_pin_browser.py", "required_pin_flow")
    assert module.pytestmark.args[0] is False
    with pytest.raises(AssertionError, match="Puppeteer is required in CI"):
        module.test_review_pin_phone_and_desktop(None)


@pytest.mark.parametrize("workflow,event,name", [
    ("ci.yml", "push", "Main CI"),
    ("release-checks.yml", "workflow_dispatch", "Release checks"),
    ("release-checks.yml", "schedule", "Release checks"),
])
@pytest.mark.parametrize("change", [None, "sha", "branch", "event", "pending", "failed",
                                  "missing", "newer_failure", "newer_success"])
def test_release_requires_latest_main_checks_on_exact_commit(
    change, workflow, event, name, monkeypatch,
) -> None:
    module = _module(ROOT / "scripts/check_release_ci.py", "release_ci")
    run = {"id": 1, "head_sha": "abc", "head_branch": "main", "event": event,
           "status": "completed", "conclusion": "success"}
    if change in ("sha", "branch", "event"):
        run[{"sha": "head_sha", "branch": "head_branch", "event": "event"}[change]] = (
            ("workflow_dispatch" if event == "push" else "push") if change == "event" else "other"
        )
    if change == "pending":
        run.update(status="in_progress", conclusion=None)
    if change == "failed":
        run["conclusion"] = "failure"
    runs = [] if change == "missing" else [run]
    if change == "newer_failure":
        runs.insert(0, dict(run, id=2, conclusion="failure"))
    if change == "newer_success":
        runs.append(dict(run, id=0, conclusion="failure"))
    # The matching run can be on a later API page.
    pages = [{"workflow_runs": []}, {"workflow_runs": runs}]
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "abc\n")
    def response(command, **kwargs):
        selected = workflow in command[-1].split("/")
        result = pages if selected else [{"workflow_runs": [dict(
            run, head_sha="abc", head_branch="main", status="completed", conclusion="success",
            event="workflow_dispatch" if event == "push" else "push",
        )]}]
        return subprocess.CompletedProcess(command, 0, json.dumps(result))

    api = Mock(side_effect=response)
    monkeypatch.setattr(module.subprocess, "run", api)
    if change in (None, "newer_success"):
        module.main()
        assert api.call_count == 2
    else:
        with pytest.raises(ValueError, match=f"{name} must pass on the tagged commit") as caught:
            module.main()
        assert "gh workflow run release-checks.yml --ref main" in str(caught.value)
        assert ("missing" if change in ("missing", "sha", "branch", "event")
                else "in_progress" if change == "pending" else "failure") in str(caught.value)
    for call in api.call_args_list:
        command = call.args[0]
        assert "--paginate" in command and "--slurp" in command
        assert "head_sha=abc&branch=main" in command[-1]
        assert "per_page=100" in command[-1]
        if "ci.yml" in command[-1].split("/"):
            assert "&event=push&" in command[-1]
        else:
            assert "&event=" not in command[-1]


@pytest.mark.parametrize("event", ["push", "pull_request", "pull_request_target", "workflow_run"])
def test_release_checks_rejects_other_events(event):
    module = _module(ROOT / "scripts/check_release_ci.py", "release_ci")
    run = {"id": 1, "head_sha": "abc", "head_branch": "main", "event": event,
           "status": "completed", "conclusion": "success"}
    with pytest.raises(ValueError, match="Release checks must pass"):
        module.require_ci([{"workflow_runs": [run]}], "abc", "Release checks",
                          ("workflow_dispatch", "schedule"))


def test_release_checks_uses_latest_across_accepted_events():
    module = _module(ROOT / "scripts/check_release_ci.py", "release_ci")
    run = {"id": 1, "head_sha": "abc", "head_branch": "main", "event": "workflow_dispatch",
           "status": "completed", "conclusion": "success"}
    with pytest.raises(ValueError, match="failure"):
        module.require_ci([{"workflow_runs": [run, dict(
            run, id=2, event="schedule", conclusion="failure",
        )]}], "abc", "Release checks", ("workflow_dispatch", "schedule"))


@pytest.mark.parametrize("failure", [None, "setup", "start", "http", "empty"])
def test_workspace_acceptance_is_fresh_and_stops_home(tmp_path, monkeypatch, failure) -> None:
    module = _module(ROOT / "scripts/accept_workspace.py", "release_accept")
    archive = tmp_path / "workspace.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("setup.py", "# synthetic setup\n")
        bundle.writestr("wheels/synthetic.whl", b"synthetic wheel")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        folder = kwargs["cwd"]
        assert (folder / "wheels/synthetic.whl").read_bytes() == b"synthetic wheel"
        env = kwargs["env"]
        for name in ("HOME", "UV_TOOL_DIR", "UV_TOOL_BIN_DIR", "UV_CACHE_DIR",
                     "YAP_STUDIO_CONFIG", "YAP_STUDIO_CACHE", "YAP_STUDIO_DATA_DIR"):
            assert Path(env[name]).is_relative_to(folder.parent)
        assert "PYTHONPATH" not in env
        if "setup.py" in command:
            assert "--check" in command
            if failure == "setup":
                raise subprocess.CalledProcessError(1, command)
        else:
            assert command[0] == str(folder.parent / "bin/yap-studio")
            if "--detach" in command and failure == "start":
                raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0, '{"url":"http://127.0.0.1:9999/"}')

    monkeypatch.setattr(module.subprocess, "run", run)
    page = BytesIO(b"" if failure == "empty" else b"<html>Yap Studio</html>")
    page.status = 503 if failure == "http" else 200
    fetch = Mock(return_value=page)
    monkeypatch.setattr(module.urllib.request, "urlopen", fetch)
    if failure:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            module.accept(archive)
    else:
        module.accept(archive)
        assert fetch.call_args.args == ("http://127.0.0.1:9999/",)
    if failure == "setup":
        assert len(commands) == 1
    else:
        assert "--stop" in commands[-1]
    assert not Path(commands[0][0]).is_relative_to(tmp_path)  # Setup uses this Python.


@pytest.mark.parametrize("result", ["match", "stale", "unknown", "install_failure"])
def test_setup_check_installs_only_bundled_wheel(tmp_path, monkeypatch, result) -> None:
    module, wheel, engine, report = _build_setup_fixture(tmp_path, monkeypatch)
    report["install"]["build"] = module._wheel_build(wheel) if result == "match" else "other"
    monkeypatch.setattr(module, "_tool_bin_dir", lambda uv: Path(engine).parent)
    monkeypatch.setattr(module, "_doctor", lambda engine: (None, None if result == "unknown"
                                                         else report))
    install = Mock()
    if result == "install_failure":
        install.side_effect = subprocess.CalledProcessError(1, ["uv"])
    monkeypatch.setattr(module.subprocess, "run", install)
    assert module.main(["--check"]) == (0 if result == "match" else 1)
    assert install.call_count == 1
    assert install.call_args.args[0] == module._engine_commands("/usr/bin/uv", wheel,
                                                              Path(engine))[0]
    assert install.call_args.kwargs["stdin"] == subprocess.DEVNULL
