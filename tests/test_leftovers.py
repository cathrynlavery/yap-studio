"""Tests fail and clean up real detached children left behind by a test."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

import pytest

from yap_studio import background
from yap_studio.project.lock import pid_alive

pytest_plugins = ["pytester"]


@pytest.fixture
def nested(pytester: pytest.Pytester):
    tests = Path(__file__).parent
    pytester.makeconftest((tests / "conftest.py").read_text())
    pytester.makepyfile(synth=(tests / "synth.py").read_text())
    yield pytester
    # Keep the regression safe even when the guard itself is broken.
    for facts in pytester.path.glob("*.facts.json"):
        saved = json.loads(facts.read_text())
        for pid in saved["pids"]:
            if background.command_matches(pid, saved["identifier"], saved["marker"]) is True:
                os.kill(pid, signal.SIGKILL)


CHILD = r'''
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

def launch(root, record, marker, identifier, *, stubborn=False, descendant=False, project=None):
    record.parent.mkdir(parents=True, exist_ok=True)
    ready = root / "ready.json"
    code = "\n".join([
        "import json, os, signal, subprocess, sys, time",
        "from pathlib import Path",
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)" if stubborn else "pass",
        "pids = [os.getpid()]",
        "pids.append(subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)', "
            "*sys.argv[2:]]).pid)" if descendant else "pass",
        "ready = Path(sys.argv[1])",
        "temporary = ready.with_name(ready.name + '.tmp')",
        "temporary.write_text(json.dumps(pids))",
        "os.replace(temporary, ready)",
        "time.sleep(60)",
    ])
    command = [sys.executable, "-c", code, str(ready)]
    if project is not None:
        command.extend(("-m", "yap_studio", "run", str(project)))
    child = subprocess.Popen([*command, marker, identifier],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
    deadline = time.monotonic() + 5
    while True:
        try:
            pids = json.loads(ready.read_text())
            break
        except (OSError, json.JSONDecodeError):
            assert child.poll() is None, f"child exited before publishing valid JSON in {ready}"
            assert time.monotonic() < deadline, f"timed out waiting for valid JSON in {ready}"
            time.sleep(0.01)
    facts = {"pids": pids, "identifier": identifier,
             "marker": marker, "record": str(record)}
    (root / "child.facts.json").write_text(json.dumps(facts))
    record.write_text(json.dumps({"pid": child.pid, "id": identifier, "state": "running"}))
    return child
'''


def assert_gone(nested: pytest.Pytester) -> dict:
    facts = json.loads((nested.path / "child.facts.json").read_text())
    deadline = time.monotonic() + 5
    while any(pid_alive(pid) for pid in facts["pids"]) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not any(pid_alive(pid) for pid in facts["pids"]), facts
    return facts


@pytest.mark.parametrize(("kind", "location"), [
    (kind, location) for kind in ("style-pick", "review", "run", "take-job")
    for location in ("owner", "tmp", "override")
    if kind not in ("run", "take-job") or location != "override"
])
def test_leftover_fails_and_stops(nested: pytest.Pytester, kind: str, location: str) -> None:
    nested.makepyfile(test_leak=CHILD + f'''
def test_leaks(_owner_folders, tmp_path, monkeypatch):
    root = Path({str(nested.path)!r})
    folder = {{"owner": _owner_folders, "tmp": tmp_path,
               "override": root / "elsewhere"}}[{location!r}]
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(folder / "data"))
    if {kind!r} == "take-job":
        record = folder / "cache/transcripts/jobs/test.json"
        marker, identifier = "--take-job", str(record)
    elif {kind!r} == "run":
        record = folder / "project/.yap/run/state.json"
        marker, identifier = "--detached-run", str(record.parent / "test-run")
    else:
        record = folder / "data/detached/{kind}-test.json"
        marker, identifier = "--detached-record", str(record)
    launch(root, record, marker, identifier, descendant=True)
''')
    result = nested.runpytest_subprocess(
        "-q", "-p", "no:cacheprovider", f"--basetemp={nested.path / 'run'}",
    )
    result.assert_outcomes(passed=1, errors=1)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    facts = assert_gone(nested)
    output = result.stdout.str()
    assert "test_leak.py::test_leaks left detached processes running:" in output
    assert facts["record"] in output
    assert f"pid {facts['pids'][0]}" in output


@pytest.mark.parametrize(("kind", "location"), [
    ("review", "owner"), ("review", "tmp"), ("review", "override"),
    ("run", "owner"), ("run", "tmp"),
    ("take-job", "owner"), ("take-job", "tmp"),
])
def test_deleted_record_fails_and_stops(
    nested: pytest.Pytester, kind: str, location: str,
) -> None:
    nested.makepyfile(test_orphan=CHILD + f'''
def test_orphan(_owner_folders, tmp_path, monkeypatch):
    root = Path({str(nested.path)!r})
    folder = {{"owner": _owner_folders, "tmp": tmp_path,
               "override": root / "elsewhere"}}[{location!r}]
    project = None
    if {kind!r} == "take-job":
        record = folder / "cache/transcripts/jobs/orphan.json"
        marker, identifier = "--take-job", str(record)
        log = record.with_suffix(".log")
    elif {kind!r} == "run":
        project = folder / "my project"
        record = project / ".yap/run/state.json"
        marker, identifier = "--detached-run", "orphan-run"
        log = record.parent / "run.log"
    else:
        monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(folder / "data"))
        record = folder / "data/detached/review-x.json"
        marker, identifier = "--detached-record", str(record)
        log = record.with_suffix(".log")
    launch(root, record, marker, identifier, descendant=True, project=project)
    log.write_text("synthetic child log")
    # A real detached run keeps run.log when its state record is removed.
    record.unlink()
''')
    result = nested.runpytest_subprocess(
        "-q", "-p", "no:cacheprovider", f"--basetemp={nested.path / 'run'}",
    )
    result.assert_outcomes(passed=1, errors=1)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    facts = assert_gone(nested)
    output = result.stdout.str()
    assert "test_orphan.py::test_orphan left detached processes running:" in output
    assert "(no record)" in output
    assert facts["record"] in output
    assert f"pid {facts['pids'][0]}" in output


@pytest.mark.parametrize("kind", ["review", "run"])
def test_outside_orphan_with_shared_path_prefix_is_left_alone(
    nested: pytest.Pytester, kind: str,
) -> None:
    nested.makepyfile(test_outside=CHILD + f'''
import pytest

@pytest.fixture(scope="session")
def outside():
    children = []
    yield children
    from yap_studio import background
    for child, marker, identifier in children:
        assert background.command_matches(child.pid, identifier, marker) is True
        child.terminate()
        child.wait(timeout=5)

def test_outside(tmp_path, outside):
    # Pay for the process scan, but keep the child in a sibling with a shared prefix.
    (tmp_path / "data/detached").mkdir(parents=True)
    folder = tmp_path.with_name(tmp_path.name + "-outside")
    project = None
    if {kind!r} == "run":
        project = folder / "my project"
        record = project / ".yap/run/state.json"
        marker, identifier = "--detached-run", "outside-run"
    else:
        record = folder / "data/detached/review-x.json"
        marker, identifier = "--detached-record", str(record)
    child = launch(Path({str(nested.path)!r}), record, marker, identifier, project=project)
    outside.append((child, marker, identifier))
    record.unlink()

def test_still_alive(outside):
    assert outside[0][0].poll() is None
''')
    result = nested.runpytest_subprocess(
        "-q", "-p", "no:cacheprovider", f"--basetemp={nested.path / 'run'}",
    )
    result.assert_outcomes(passed=2)
    assert result.ret == pytest.ExitCode.OK
    assert_gone(nested)


@pytest.mark.parametrize("cleanup", ["body", "fixture"])
def test_own_cleanup_passes(nested: pytest.Pytester, cleanup: str) -> None:
    nested.makepyfile(test_clean=CHILD + f'''
import pytest

@pytest.fixture
def page(_owner_folders):
    record = _owner_folders / "data/detached/style-pick-clean.json"
    child = launch(Path({str(nested.path)!r}), record, "--detached-record", str(record))
    yield child
    if {cleanup!r} == "fixture":
        child.terminate()
        child.wait(timeout=5)

def test_cleans(page):
    if {cleanup!r} == "body":
        page.terminate()
        page.wait(timeout=5)
''')
    result = nested.runpytest_subprocess(
        "-q", "-p", "no:cacheprovider", f"--basetemp={nested.path / 'run'}",
    )
    result.assert_outcomes(passed=1)
    assert result.ret == pytest.ExitCode.OK
    assert_gone(nested)


def test_unrelated_live_process_and_stale_records_pass(nested: pytest.Pytester) -> None:
    nested.makepyfile(test_unrelated=CHILD + f'''
child = None

def test_unrelated(_owner_folders):
    global child
    record = _owner_folders / "data/detached/style-pick-unrelated.json"
    root = Path({str(nested.path)!r})
    child = launch(root, record, "--detached-record", str(root / "outside/style-pick-other.json"))
    (record.parent / "review-stale.json").write_text('{{"pid": 2147483647}}')
    (record.parent / "style-pick-malformed.json").write_text('[]')

def test_still_alive():
    assert child.poll() is None
    child.terminate()
    child.wait(timeout=5)
''')
    result = nested.runpytest_subprocess(
        "-q", "-p", "no:cacheprovider", f"--basetemp={nested.path / 'run'}",
    )
    result.assert_outcomes(passed=2)
    assert result.ret == pytest.ExitCode.OK
    assert_gone(nested)


@pytest.mark.parametrize("cleanup", [False, True])
def test_take_job_holder_cleanup(nested: pytest.Pytester, cleanup: bool) -> None:
    from test_transcript_take_job import FAKE_JOB

    nested.makepyfile(fake_job=FAKE_JOB)
    nested.makepyfile(test_job='''
import json, sys, time
from pathlib import Path
from yap_studio import cache
from yap_studio.transcript.take_job import TakeJobs

class Backend:
    name = "parakeet-mlx"
    model = "test-model"
    def version(self): return "test-version"
    def download_notice(self): return None

jobs = None

def test_job(tmp_path):
    global jobs
    take = tmp_path / "take.mp4"
    take.touch()
    jobs = TakeJobs(child_command=[sys.executable, str(Path(__file__).with_name("fake_job.py"))])
    assert jobs.start(take, Backend())
    deadline = time.monotonic() + 8
    while True:
        entries = list(cache.cache_dir("transcripts", "jobs").glob("*.pids"))
        if entries:
            record = entries[0].with_suffix(".json")
            pids = json.loads(entries[0].read_text())
            break
        assert time.monotonic() < deadline
        time.sleep(0.02)
    facts = {"pids": pids, "identifier": str(record),
             "marker": "--take-job", "record": str(record)}
    Path(__file__).with_name("child.facts.json").write_text(json.dumps(facts))
    if CLEANUP:
        jobs.close()
'''.replace("CLEANUP", str(cleanup)))
    result = nested.runpytest_subprocess(
        "-q", "-p", "no:cacheprovider", f"--basetemp={nested.path / 'run'}",
    )
    result.assert_outcomes(passed=1, errors=0 if cleanup else 1)
    facts = assert_gone(nested)
    if not cleanup:
        assert facts["record"] in result.stdout.str()
        assert f"pid {facts['pids'][0]}" in result.stdout.str()


@pytest.mark.parametrize("remove_record", [False, True])
def test_style_fixture_stops_busy_replacement(
    nested: pytest.Pytester, remove_record: bool,
) -> None:
    tests = Path(__file__).parent
    source = (tests / "test_style_pick_detach.py").read_text().replace(
        "ROOT = Path(__file__).parents[1]", f"ROOT = Path({str(tests.parent)!r})",
    )
    nested.makepyfile(style_fixture=source)
    nested.makepyfile(test_busy='''
import json
import os
from pathlib import Path
from style_fixture import isolated, page, records, request, start

def test_busy_replacement(page):
    site = Path(page["PYTHONPATH"].split(os.pathsep)[0]) / "sitecustomize.py"
    with site.open("a") as output:
        output.write("""
import json, signal, sys
from yap_studio import background
from yap_studio.gallery.server import GalleryServer
serve = GalleryServer.serve_forever
def busy_serve(self):
    def busy_stop(signum, frame):
        background.write_json(Path(sys.argv[-1]), {"pid": os.getpid()})
    if os.environ.get("TEST_BUSY_STYLE"):
        signal.signal(signal.SIGTERM, busy_stop)
    serve(self)
GalleryServer.serve_forever = busy_serve
""")
    first = start(page)
    [record] = records(page)
    saved = json.loads(record.read_text())
    saved.pop("document")
    record.write_text(json.dumps(saved))
    second = start({**page, "TEST_BUSY_STYLE": "1"})
    assert second["detached"]["pid"] != first["detached"]["pid"]
    assert request(second["url"])["style"]
    [record] = records(page)
    facts = {"pids": [second["detached"]["pid"]], "identifier": str(record),
             "marker": "--detached-record", "record": str(record)}
    (Path(__file__).parent / "child.facts.json").write_text(json.dumps(facts))
    if REMOVE_RECORD:
        record.unlink()
'''.replace("REMOVE_RECORD", str(remove_record)))
    result = nested.runpytest_subprocess(
        "test_busy.py", "-q", "-p", "no:cacheprovider", f"--basetemp={nested.path / 'run'}",
    )
    result.assert_outcomes(passed=1)
    assert result.ret == pytest.ExitCode.OK
    assert_gone(nested)
