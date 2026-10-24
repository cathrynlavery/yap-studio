"""Private home requests and one-shot listener lifecycle."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_feedback_batch import feedback_project as feedback_project

from yap_studio.errors import ValidationError
from yap_studio.homepage import bridge


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "inbox").mkdir()
    return tmp_path


@pytest.fixture
def story_workspace(workspace, feedback_project):
    from yap_studio.feedback.versions import save_version

    project = workspace / "projects/demo"
    project.parent.mkdir()
    feedback_project.rename(project)
    save_version(project)
    return workspace


@pytest.mark.parametrize("captions", [None, False, True])
@pytest.mark.parametrize("sources", [[], ["folder"], ["album", "stock"]])
def test_horizontal_request_reaches_listener_with_caption_choice(workspace, captions, sources):
    payload = {"shape": "horizontal", "video_type": "talking_head", "takes": ["wide.mp4"],
               "broll_sources": sources}
    if captions is not None:
        payload["captions"] = captions
    bridge.create_request(workspace, "edit", payload)
    event = bridge.wait(workspace, timeout=1)
    assert event["request"]["payload"] == {**payload, "captions": captions is True}


@pytest.mark.parametrize("extra", [
    {"shape": "square"}, {"captions": "yes"}, {"video_type": "voiceover"},
    {"video_type": "screen_recording"}, {"text_hook": "A title"},
    {"top_text": "whole"},
])
def test_horizontal_request_refuses_unsupported_choices(workspace, extra):
    with pytest.raises(ValidationError):
        bridge.create_request(workspace, "edit", {
            "shape": "horizontal", "video_type": "talking_head", "broll_sources": [], **extra,
        })


def test_horizontal_request_allows_chosen_opening(workspace):
    payload = {"shape": "horizontal", "video_type": "talking_head", "broll_sources": [],
               "top_text": "opening", "text_hook": "Make the story clear"}
    request = bridge.create_request(workspace, "edit", payload)
    assert request["payload"] == {**payload, "captions": False}


def test_final_wait_and_ack_cli(story_workspace, capsys):
    from yap_studio import cli

    row = bridge.create_request(story_workspace, "final", {"project": "projects/demo"})
    assert (
        cli.main(["home", "wait", "--workspace", str(story_workspace), "--timeout", "1", "--json"])
        == 0
    )
    event = json.loads(capsys.readouterr().out)
    assert event["request"]["kind"] == "final"
    assert event["request"]["payload"] == {"project": "projects/demo"}
    assert event["next"] == (
        "Build the final video for this project from its approved story cut "
        "and acknowledge this request."
    )
    for state in ("claimed", "done"):
        assert (
            cli.main(
                [
                    "home",
                    "ack",
                    row["id"],
                    "--workspace",
                    str(story_workspace),
                    "--state",
                    state,
                    "--json",
                ]
            )
            == 0
        )
        assert json.loads(capsys.readouterr().out)["state"] == state


@pytest.mark.parametrize("state", ["pending", "delivered", "claimed"])
def test_final_requests_conflict_until_finished(story_workspace, state):
    payload = {"project": "projects/demo"}
    first = bridge.create_request(story_workspace, "final", payload)
    if state == "delivered":
        bridge.wait(story_workspace, timeout=1)
    elif state == "claimed":
        bridge.ack(story_workspace, first["id"], "claimed")
    with pytest.raises(bridge.ConflictError, match="already"):
        bridge.create_request(story_workspace, "final", payload)
    bridge.ack(story_workspace, first["id"], "done")
    assert bridge.create_request(story_workspace, "final", payload)["id"] != first["id"]


def test_concurrent_final_requests_are_serialized_per_project(story_workspace):
    import shutil

    shutil.copytree(story_workspace / "projects/demo", story_workspace / "projects/other")

    def submit(_):
        try:
            return bridge.create_request(story_workspace, "final", {"project": "projects/demo"})
        except bridge.ConflictError:
            return None

    with ThreadPoolExecutor(4) as pool:
        assert len(list(filter(None, pool.map(submit, range(4))))) == 1
    assert (
        bridge.create_request(story_workspace, "final", {"project": "projects/other"})["payload"][
            "project"
        ]
        == "projects/other"
    )


def test_requests_private_atomic_order_and_terminal_ack(workspace):
    with ThreadPoolExecutor(4) as pool:
        rows = list(pool.map(lambda _: bridge.create_request(workspace, "learn", {}), range(8)))
    assert sorted(int(row["id"][1:]) for row in rows) == list(range(1, 9))
    root = workspace / ".yap-studio/homepage"
    assert root.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in root.rglob("*") if p.is_file())
    result = bridge.wait(workspace, timeout=1)
    assert result["event"] == "request" and result["request"]["id"] == "r1"
    assert result["request"]["state"] == "delivered" and result["next"]
    assert not (root / "listener.json").exists()
    bridge.ack(workspace, "r1", "claimed", project="projects/demo", message="Starting")
    row = bridge.ack(workspace, "r1", "done")
    assert row["ack"]["state"] == "done"
    with pytest.raises(ValidationError):
        bridge.ack(workspace, "r1", "claimed")
    with pytest.raises(ValidationError):
        bridge.ack(workspace, "r999", "done")
    with pytest.raises(ValidationError):
        bridge.ack(workspace, "r2", "claimed", project="../outside")


def test_cancel_and_edit_conflict(workspace):
    row = bridge.create_request(workspace, "edit", {"takes": []})
    with pytest.raises(bridge.ConflictError):
        bridge.create_request(workspace, "edit", {})
    assert bridge.cancel(workspace, row["id"])["state"] == "cancelled"
    bridge.create_request(workspace, "edit", {})
    with pytest.raises(bridge.ConflictError):
        bridge.cancel(workspace, row["id"])


@pytest.fixture
def clock(monkeypatch):
    current = [datetime(2026, 10, 22, 12, tzinfo=UTC)]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return current[0]

    monkeypatch.setattr(bridge, "datetime", Clock)
    return current


def test_interrupted_claim_recovers_once_with_previous_ack(workspace, clock):
    from yap_studio.homepage.state import HomeState

    project = workspace / "projects/demo"
    project.mkdir(parents=True)
    row = bridge.create_request(workspace, "edit", {})
    bridge.wait(workspace, timeout=1)
    claimed = bridge.ack(
        workspace, row["id"], "claimed", project="projects/demo", message="Story cut saved."
    )
    assert not (bridge.directory(workspace) / "listener.json").exists()
    clock[0] += timedelta(minutes=10)
    assert HomeState(workspace).snapshot()["requests"][0]["stalled"] is True
    resumed = bridge.resume(workspace, row["id"])
    assert resumed["state"] == "pending" and resumed["resumes"] == 1
    assert resumed["ack"] == claimed["ack"]
    with pytest.raises(bridge.ConflictError):
        bridge.resume(workspace, row["id"])
    event = bridge.wait(workspace, timeout=1)
    assert event["resumed"] is True
    assert event["request"]["id"] == row["id"]
    assert event["request"]["ack"] == claimed["ack"]
    for instruction in ("earlier session", "receipts", "yap-studio status", "paid", "claimed"):
        assert instruction in event["next"]
    clock[0] += timedelta(seconds=121)
    with ThreadPoolExecutor(4) as pool:
        assert all(result == {"event": "timeout"} for result in pool.map(
            lambda _: bridge.wait(workspace, timeout=1), range(4)
        ))
    assert HomeState(workspace).snapshot()["requests"][0]["stalled"] is False
    clock[0] += timedelta(minutes=10)
    assert bridge.resume(workspace, row["id"])["resumes"] == 2
    assert bridge.wait(workspace, timeout=1)["resumed"] is True
    bridge.ack(workspace, row["id"], "claimed")
    assert bridge.ack(workspace, row["id"], "done")["state"] == "done"


@pytest.mark.parametrize("state", ["delivered", "claimed"])
@pytest.mark.parametrize("guard", ["live_listener", "recent_listener", "recent_activity", "lock"])
def test_stall_guards_refuse_resume_and_cancel(workspace, clock, state, guard):
    from yap_studio.homepage.state import HomeState

    project = workspace / "projects/demo"
    project.mkdir(parents=True)
    row = bridge.create_request(workspace, "edit", {})
    bridge.wait(workspace, timeout=1)
    if state == "claimed":
        bridge.ack(workspace, row["id"], state, project="projects/demo")
    clock[0] += timedelta(minutes=10)
    if guard == "live_listener":
        bridge.write_json(bridge.directory(workspace) / "listener.json", {
            "pid": os.getpid(), "heartbeat_at": bridge.timestamp(),
        })
    elif guard == "recent_listener":
        other = bridge.create_request(workspace, "message", {"project": None, "text": "Hello"})
        assert bridge.wait(workspace, timeout=1)["request"]["id"] == other["id"]
    elif guard == "recent_activity":
        clock[0] -= timedelta(seconds=1)
    # A delivered request can carry the previous ack after Send again.
    if state == "delivered" and guard == "lock":
        target = bridge.directory(workspace) / "requests" / f"{row['id']}.json"
        delivered = bridge.read_json(target)
        delivered["ack"] = {"project": "projects/demo", "at": delivered["delivered_at"]}
        bridge.write_json(target, delivered)

    def refused():
        current = next(
            r for r in HomeState(workspace).snapshot()["requests"] if r["id"] == row["id"]
        )
        assert current["stalled"] is False
        for action in (bridge.resume, bridge.cancel):
            with pytest.raises(bridge.ConflictError):
                action(workspace, row["id"])

    if guard == "lock":
        with bridge.project_lock(project):
            refused()
        # The lock file still names this live PID after release. Only the OS lock matters.
        assert HomeState(workspace).snapshot()["requests"][0]["stalled"] is True
    else:
        refused()


@pytest.mark.parametrize("state", ["delivered", "claimed"])
def test_cancel_stalled_request(workspace, clock, state):
    row = bridge.create_request(workspace, "edit", {})
    bridge.wait(workspace, timeout=1)
    if state == "claimed":
        bridge.ack(workspace, row["id"], state)
    clock[0] += timedelta(minutes=10)
    assert bridge.cancel(workspace, row["id"])["state"] == "cancelled"


def test_new_ack_renews_stall_clock(workspace, clock):
    row = bridge.create_request(workspace, "edit", {})
    bridge.wait(workspace, timeout=1)
    clock[0] += timedelta(minutes=9)
    row = bridge.ack(workspace, row["id"], "claimed")
    clock[0] += timedelta(minutes=1)
    assert bridge.stalled(workspace, row) is False
    clock[0] += timedelta(minutes=9)
    assert bridge.stalled(workspace, row) is True


def test_concurrent_resumes_and_delivery_are_serialized(workspace, clock):
    row = bridge.create_request(workspace, "edit", {})
    bridge.wait(workspace, timeout=1)
    bridge.ack(workspace, row["id"], "claimed")
    clock[0] += timedelta(minutes=10)

    def resume(_):
        try:
            return bridge.resume(workspace, row["id"])
        except bridge.ConflictError:
            return None

    with ThreadPoolExecutor(4) as pool:
        assert len(list(filter(None, pool.map(resume, range(4))))) == 1
        events = list(pool.map(lambda _: bridge.wait(workspace, timeout=1), range(4)))
    delivered = [event for event in events if event["event"] == "request"]
    assert len(delivered) == 1 and delivered[0]["resumed"] is True


def test_ack_keeps_project_names_in_rename_order(workspace):
    row = bridge.create_request(workspace, "edit", {})
    bridge.ack(workspace, row["id"], "claimed", project="projects/first")
    renamed = bridge.ack(workspace, row["id"], "claimed", project="projects/second")
    assert renamed["ack"]["renamed_from"] == ["projects/first"]
    repeated = bridge.ack(workspace, row["id"], "claimed", project="projects/second")
    assert repeated["ack"]["renamed_from"] == ["projects/first"]
    third = bridge.ack(workspace, row["id"], "done", project="projects/third")
    assert third["ack"]["renamed_from"] == ["projects/first", "projects/second"]
    repeated = bridge.ack(workspace, row["id"], "done")
    assert repeated["ack"]["renamed_from"] == ["projects/first", "projects/second"]
    assert bridge.requests(workspace)[0]["ack"] == repeated["ack"]


def test_ack_carries_rename_history_across_requests(workspace):
    first = bridge.create_request(workspace, "edit", {})
    bridge.ack(workspace, first["id"], "claimed", project="projects/first")
    bridge.ack(workspace, first["id"], "done", project="projects/second")
    second = bridge.create_request(workspace, "edit", {})
    bridge.ack(workspace, second["id"], "claimed", project="projects/second")
    result = bridge.ack(workspace, second["id"], "done", project="projects/third")
    assert result["ack"]["renamed_from"] == ["projects/first", "projects/second"]
    assert bridge.requests(workspace)[0]["ack"] == result["ack"]


@pytest.mark.parametrize("older_updated", [False, True])
@pytest.mark.parametrize("empty_history", [False, True])
def test_ack_rename_history_uses_latest_other_ack(workspace, older_updated, empty_history):
    rows = []
    for name in ("alpha", "delta"):
        row = bridge.create_request(workspace, "edit", {})
        bridge.ack(workspace, row["id"], "claimed", project=f"projects/{name}")
        rows.append(bridge.ack(workspace, row["id"], "done", project="projects/bravo"))
    selected = rows[0] if older_updated else rows[1]
    for row in rows:
        row["ack"]["at"] = "2026-10-20T10:00:00Z"
    if older_updated:
        selected["ack"]["at"] = "2026-10-20T11:00:00Z"
    if empty_history:
        selected["ack"]["renamed_from"] = []
    for row in rows:
        bridge.write_json(bridge.directory(workspace) / "requests" / f"{row['id']}.json", row)
    current = bridge.create_request(workspace, "edit", {})
    bridge.ack(workspace, current["id"], "claimed", project="projects/bravo")
    result = bridge.ack(workspace, current["id"], "done", project="projects/echo")
    expected = [] if empty_history else [f"projects/{'alpha' if older_updated else 'delta'}"]
    assert result["ack"]["renamed_from"] == [*expected, "projects/bravo"]


def test_ack_project_history_is_unique_and_bounded(workspace):
    row = bridge.create_request(workspace, "edit", {})
    for number in range(25):
        result = bridge.ack(workspace, row["id"], "claimed", project=f"projects/name-{number}")
    assert result["ack"]["renamed_from"] == [f"projects/name-{n}" for n in range(4, 24)]
    result = bridge.ack(workspace, row["id"], "claimed", project="projects/name-5")
    assert len(result["ack"]["renamed_from"]) == 20
    assert "projects/name-5" not in result["ack"]["renamed_from"]
    result = bridge.ack(workspace, row["id"], "claimed", project="projects/latest")
    assert len(result["ack"]["renamed_from"]) == len(set(result["ack"]["renamed_from"]))


@pytest.mark.parametrize(
    "history,expected",
    [
        ("projects/x", []),
        ([1, "projects/old"], ["projects/old"]),
        ([None, {}, "old", "projects/", "projects/nested/name", "projects/.."], []),
    ],
)
def test_ack_ignores_invalid_stored_project_history(workspace, history, expected):
    row = bridge.create_request(workspace, "edit", {})
    row = bridge.ack(workspace, row["id"], "claimed", project="projects/current")
    row["ack"]["renamed_from"] = history
    bridge.write_json(bridge.directory(workspace) / "requests" / f"{row['id']}.json", row)
    result = bridge.ack(workspace, row["id"], "done", project="projects/new")
    assert result["ack"]["renamed_from"] == [*expected, "projects/current"]


def test_message_wait_instruction_and_trimmed_payload(workspace):
    row = bridge.create_request(workspace, "message", {"project": None, "text": "  Hello there \n"})
    assert row["payload"] == {"project": None, "text": "Hello there"}
    second = bridge.create_request(
        workspace, "message", {"project": None, "text": "Another thought"}
    )
    event = bridge.wait(workspace, timeout=1)
    assert event["request"]["id"] == row["id"] and event["request"]["kind"] == "message"
    assert event["next"] == (
        "Read the creator's message, act on it, and reply in one or two plain sentences "
        "with yap-studio home ack ID --state done --message TEXT."
    )
    assert bridge.wait(workspace, timeout=1)["request"]["id"] == second["id"]


@pytest.mark.parametrize("text", [None, True, {}, "", " \t\n", "x" * 2001])
def test_message_text_validation(workspace, text):
    with pytest.raises(ValidationError):
        bridge.create_request(workspace, "message", {"project": None, "text": text})


def test_multiple_messages_can_be_pending(workspace):
    bridge.create_request(workspace, "edit", {})
    with ThreadPoolExecutor(4) as pool:
        rows = list(
            pool.map(
                lambda n: bridge.create_request(
                    workspace, "message", {"project": None, "text": f"Question {n}"}
                ),
                range(4),
            )
        )
    assert len({row["id"] for row in rows}) == 4
    assert all(row["state"] == "pending" for row in rows)


def test_seed_history_then_report_notes_and_approvals(workspace, monkeypatch):
    project = workspace / "projects/demo"
    feedback = project / "feedback"
    feedback.mkdir(parents=True)
    (feedback / "b1.json").write_text("{}")
    approved = {"id": "v1"}
    from types import SimpleNamespace

    monkeypatch.setattr(bridge, "verified_approval", lambda _: SimpleNamespace(**approved))
    assert bridge.wait(workspace, timeout=1) == {"event": "timeout"}
    (feedback / "b2.json").write_text("{}")
    assert bridge.wait(workspace, timeout=1) == {
        "event": "notes",
        "project": "projects/demo",
        "batch": "b2",
    }
    approved["id"] = "v2"
    assert bridge.wait(workspace, timeout=1) == {
        "event": "approved",
        "project": "projects/demo",
        "version": "v2",
    }
    events = bridge.events(workspace)
    assert [row["kind"] for row in events] == ["approved", "notes"]
    assert all(row["delivered_at"] for row in events)


def _listener(workspace: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "yap_studio",
            "home",
            "wait",
            "--workspace",
            str(workspace),
            "--timeout",
            "30",
            "--json",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _ready(workspace: Path, child: subprocess.Popen) -> dict:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        row = bridge.read_json(workspace / ".yap-studio/homepage/listener.json", {})
        if row.get("pid") == child.pid:
            return row
        if child.poll() is not None:
            pytest.fail(str(child.communicate()))
        signal.sigtimedwait(set(), 0.02) if hasattr(signal, "sigtimedwait") else time.sleep(0.02)
    pytest.fail("listener did not become ready")


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_listener_signal_cleanup(workspace, signum):
    child = _listener(workspace)
    try:
        _ready(workspace, child)
        assert bridge.listening(workspace)
        child.send_signal(signum)
        stdout, stderr = child.communicate(timeout=10)
        assert child.returncode == 0, stderr
        assert json.loads(stdout) == {"event": "timeout"}
        assert not (workspace / ".yap-studio/homepage/listener.json").exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()


def test_newest_listener_wins(workspace):
    first = _listener(workspace)
    second = None
    try:
        _ready(workspace, first)
        second = _listener(workspace)
        _ready(workspace, second)
        first.communicate(timeout=10)
        assert bridge.listening(workspace)
        listener = workspace / ".yap-studio/homepage/listener.json"
        assert json.loads(listener.read_text())["pid"] == (second.pid)
    finally:
        for child in (first, second):
            if child is not None and child.poll() is None:
                child.terminate()
                child.communicate(timeout=10)


def test_stale_or_unrelated_listener_is_not_listening(workspace):
    bridge.write_json(
        workspace / ".yap-studio/homepage/listener.json",
        {"pid": os.getpid(), "heartbeat_at": "2000-01-01T00:00:00+00:00"},
    )
    assert not bridge.listening(workspace)


@pytest.mark.parametrize("timeout", [0, 3601])
def test_timeout_bounds(workspace, timeout):
    with pytest.raises(ValidationError):
        bridge.wait(workspace, timeout=timeout)


def test_request_priority_and_event_retention(workspace):
    project = workspace / "projects/demo/feedback"
    project.mkdir(parents=True)
    bridge.wait(workspace, timeout=1)
    for number in range(1, 54):
        (project / f"b{number}.json").write_text("{}")
    pending = bridge.create_request(workspace, "learn", {})
    assert bridge.wait(workspace, timeout=1)["request"]["id"] == pending["id"]
    for number in range(1, 54):
        assert bridge.wait(workspace, timeout=1)["batch"] == f"b{number}"
    events = bridge.events(workspace)
    assert len(events) == 50 and events[0]["id"] == "e53" and events[-1]["id"] == "e4"


def test_fresh_dead_pid_is_not_listening(workspace):
    bridge.write_json(
        workspace / ".yap-studio/homepage/listener.json",
        {"pid": 2**30, "heartbeat_at": bridge.timestamp()},
    )
    assert not bridge.listening(workspace)


def test_unrelated_process_is_never_signaled(workspace, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        bridge.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=f"echo home wait --workspace {workspace}"
        ),
    )
    assert not bridge._matches_listener(123, workspace)


def test_heartbeat_refresh_then_request_delivery(workspace):
    child = _listener(workspace)
    try:
        first = _ready(workspace, child)
        deadline = time.monotonic() + 9
        while time.monotonic() < deadline:
            current = bridge.read_json(workspace / ".yap-studio/homepage/listener.json", {})
            if current.get("heartbeat_at") != first["heartbeat_at"]:
                break
            threading.Event().wait(0.05)
        else:
            pytest.fail("listener heartbeat did not refresh")
        assert bridge.listening(workspace)
        row = bridge.create_request(workspace, "learn", {})
        stdout, stderr = child.communicate(timeout=10)
        assert child.returncode == 0, stderr
        assert json.loads(stdout)["request"]["id"] == row["id"]
        assert not (workspace / ".yap-studio/homepage/listener.json").exists()
    finally:
        if child.poll() is None:
            child.terminate()
            child.communicate(timeout=10)


@pytest.mark.parametrize("value", ["My Videos", "./My Videos"])
def test_listener_command_matches_relative_workspace_with_spaces(workspace, monkeypatch, value):
    from types import SimpleNamespace

    command = f"python -m yap_studio home wait --workspace {value} --json"
    monkeypatch.setattr(bridge.sys, "platform", "darwin")
    monkeypatch.setattr(
        bridge.subprocess,
        "run",
        lambda args, **kw: SimpleNamespace(
            returncode=0, stdout=command if args[0] == "ps" else f"p123\nn{workspace}\n"
        ),
    )
    assert bridge._matches_listener(123, workspace / "My Videos")


@pytest.mark.parametrize(
    "age,state,expected",
    [(121, "delivered", "request"), (30, "delivered", "timeout"), (121, "claimed", "timeout")],
)
def test_delivery_lease(workspace, age, state, expected):
    row = bridge.create_request(workspace, "learn", {})
    first = bridge.wait(workspace, timeout=1)
    assert first["request"]["deliveries"] == 1
    row = first["request"]
    row.update(state=state, delivered_at=(datetime.now(UTC) - timedelta(seconds=age)).isoformat())
    if state == "claimed":
        row["ack"] = {"state": "claimed"}
    with bridge.writer(workspace) as root:
        bridge.write_json(root / "requests" / f"{row['id']}.json", row)
    event = bridge.wait(workspace, timeout=1)
    assert event["event"] == expected
    if expected == "request":
        assert event["redelivered"] is True
        assert event["request"]["deliveries"] == 2
        assert datetime.fromisoformat(event["request"]["delivered_at"]) > datetime.fromisoformat(
            row["delivered_at"]
        )
        assert event["next"] == (
            "This request came back because it was never claimed. "
            "If you are already working on it, claim it and carry on. "
            + first["next"]
        )


def test_pending_requests_precede_expired_deliveries(workspace):
    row = bridge.create_request(workspace, "learn", {})
    row.update(state="delivered", delivered_at="2000-01-01T00:00:00+00:00")
    with bridge.writer(workspace) as root:
        bridge.write_json(root / "requests/r1.json", row)
    pending = bridge.create_request(workspace, "learn", {})
    assert bridge.wait(workspace, timeout=1)["request"]["id"] == pending["id"]
    event = bridge.wait(workspace, timeout=1)
    assert event["request"]["id"] == row["id"]
    assert event["request"]["deliveries"] == 2  # Legacy records had no deliveries count.
    assert event["redelivered"] is True


def test_sigterm_during_poll_returns_delivered_request(workspace, monkeypatch):
    row = bridge.create_request(workspace, "learn", {})
    poll = bridge._poll

    def interrupt_after_delivery(*args):
        result = poll(*args)
        os.kill(os.getpid(), signal.SIGTERM)
        return result

    monkeypatch.setattr(bridge, "_poll", interrupt_after_delivery)
    assert bridge.wait(workspace, timeout=1)["request"]["id"] == row["id"]
    assert not (bridge.directory(workspace) / "listener.json").exists()



def test_approval_hash_cache_is_bounded(workspace, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    calls = []
    original = bridge.sha256_file

    def hashed(path):
        calls.append(path)
        return original(path)

    def record(project, *, verify_media):
        assert verify_media is False
        return {"version": "v1"}

    def versions(project):
        media = project / "media.mp4"
        return (SimpleNamespace(
            id="v1", media_path=media,
            media={"sha256": hashlib.sha256(media.read_bytes()).hexdigest()},
        ),)

    monkeypatch.setattr(bridge, "approval_record", record)
    monkeypatch.setattr(bridge, "load_versions", versions)
    monkeypatch.setattr(bridge, "sha256_file", hashed)
    for number in range(65):
        project = workspace / f"synthetic-{number}"
        project.mkdir()
        (project / "media.mp4").write_bytes(f"synthetic media {number}".encode())
        assert bridge.verified_approval(project).id == "v1"
        assert bridge.verified_approval(project).id == "v1"
    assert len(calls) == 65
    # An older file is evicted rather than retaining every project's hash forever.
    assert bridge.verified_approval(workspace / "synthetic-0").id == "v1"
    assert len(calls) == 66



def test_launch_lock_does_not_block_request_writers(workspace):
    with bridge.writer(workspace, launch=True), ThreadPoolExecutor(1) as pool:
        row = pool.submit(bridge.create_request, workspace, "learn", {}).result(timeout=3)
        assert row["state"] == "pending"
