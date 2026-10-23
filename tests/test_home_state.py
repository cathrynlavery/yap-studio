"""Creator state, inbox identities and pure phase transitions."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_feedback_batch import _json
from test_feedback_batch import feedback_project as feedback_project
from test_feedback_versions import _render_media
from test_workspace import _module

from yap_studio.homepage import state


@pytest.mark.parametrize(
    "stages,run,versions,newer,approved,expected",
    [
        ([], None, [], False, False, "starting"),
        ([("ingest", "running")], None, [], False, False, "transcribing"),
        ([("ingest", "done")], None, [], False, False, "story"),
        ([("cut", "running")], None, [], False, False, "cutting"),
        ([], None, [{"final": False}], False, False, "cut_ready"),
        ([], None, [{"final": False}], True, False, "revising"),
        ([("captions", "running")], None, [], False, False, "building"),
        ([("graphics", "running")], None, [], False, False, "building"),
        ([("music", "running")], None, [], False, False, "building"),
        ([("assemble", "running")], None, [], False, False, "building"),
        ([], {"state": "running", "step": "Preparing"}, [], False, False, "building"),
        ([], None, [{"final": True}], False, False, "final_ready"),
        ([], None, [{"final": True}], False, True, "approved"),
        ([], {"state": "failed", "error": {"message": "Try again"}}, [], False, False, "stopped"),
        ([], {"state": "cancelled"}, [], False, False, "stopped"),
        ([], {"state": "lost"}, [], False, False, "stopped"),
    ],
)
def test_phase_table(stages, run, versions, newer, approved, expected):
    snapshot = {
        "stages": [{"name": name, "state": value, "reason": "Working"} for name, value in stages],
        "run": run,
    }
    result = state.phase(snapshot, versions, newer_feedback=newer, approved=approved)
    assert result["phase"] == expected
    if expected in {"building", "cutting", "transcribing"}:
        assert result["step"]
    if expected == "stopped":
        assert result["message"]


@pytest.mark.parametrize("run_state", ["cancelled", "failed"])
@pytest.mark.parametrize(
    "run_time,stage_time,version_time,expected",
    [
        ("10:00:00", None, "10:01:00", "final_ready"),
        ("10:00:00", "10:02:00", "10:01:00", "building"),
        ("10:03:00", "10:02:00", "10:01:00", "stopped"),
        ("10:03:00", None, "10:01:00", "stopped"),
        ("10:00:00", "09:59:00", "10:01:00", "final_ready"),
        ("10:01:00", None, "10:01:00", "stopped"),
    ],
)
def test_phase_uses_newest_work(run_state, run_time, stage_time, version_time, expected):
    snapshot = {
        "run": {"state": run_state, "finished": f"2026-10-22T{run_time}Z"},
        "stages": [] if stage_time is None else [{
            "name": "assemble", "state": "running",
            "progress": {"started": f"2026-10-22T{stage_time}+00:00", "step": "Rendering"},
        }],
    }
    versions = [{"final": True, "created_at": f"2026-10-22T{version_time}Z"}]
    result = state.phase(snapshot, versions)
    assert result["phase"] == expected
    if expected == "building":
        assert result["step"] == "Rendering"


def test_inbox_rules_cache_and_no_absolute_state(tmp_path, monkeypatch):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for name in ("old.mp4", "new.MOV", "memo.m4a", ".hidden.mp4", "ignore.txt"):
        (inbox / name).write_bytes(b"synthetic")
    os.utime(inbox / "old.mp4", ns=(1, 1))
    (inbox / "link.mp4").symlink_to(inbox / "old.mp4")
    info = (inbox / "old.mp4").stat()
    (inbox / ".used.json").write_text(
        json.dumps(
            {
                "schema": "yap-studio/inbox-used@1",
                "takes": [
                    {
                        "file": "old.mp4",
                        "bytes": info.st_size,
                        "mtime_ns": info.st_mtime_ns,
                        "project": "projects/demo",
                        "used_at": "2026-01-01",
                    }
                ],
            }
        )
    )
    calls = []
    monkeypatch.setattr(state, "probe", lambda p: calls.append(p.name) or ("memo", 2.5))
    view = state.HomeState(tmp_path)
    result = view.snapshot(on_this_mac=True)
    assert {row["name"] for row in result["inbox"]["takes"]} == {"new.MOV", "memo.m4a"}
    assert result["inbox"]["used"] == 1 and result["first_run"]
    view.snapshot(on_this_mac=False)
    assert len(calls) == 2
    assert str(tmp_path) not in json.dumps(result)
    choices = result["video_type"]["questions"][0]["choices"]
    assert [c["value"] for c in choices] == ["talking_head", "voiceover"]
    (inbox / "old.mp4").write_bytes(b"changed")
    assert len(view.snapshot()["inbox"]["takes"]) == 3


def test_real_project_status_is_relative(tmp_path):
    (tmp_path / "inbox").mkdir()
    project = tmp_path / "projects/demo"
    project.mkdir(parents=True)
    (project / "project.json").write_text(
        json.dumps(
            {
                "schema": "yap-studio/project@1",
                "name": "old-manifest-name",
                "takes": [{"id": "t1", "path": "takes/clip.mp4"}],
            }
        )
    )
    result = state.HomeState(tmp_path).snapshot()
    assert result["projects"][0]["phase"] == "starting"
    assert result["projects"][0]["dir"] == "projects/demo"
    assert result["projects"][0]["name"] == "demo"
    assert result["projects"][0]["title"] == "demo"
    assert str(tmp_path) not in json.dumps(result)


def test_project_rename_keeps_old_name_in_state(tmp_path):
    from yap_studio.homepage import bridge

    (tmp_path / "inbox").mkdir()
    project = tmp_path / "projects/2026-10-19-raw"
    _json(
        project / "project.json",
        {
            "schema": "yap-studio/project@1",
            "name": project.name,
            "takes": [{"id": "t1", "path": "takes/clip.mp4"}],
        },
    )
    request = bridge.create_request(tmp_path, "edit", {})
    bridge.ack(tmp_path, request["id"], "claimed", project=f"projects/{project.name}")
    view = state.HomeState(tmp_path)
    assert view.snapshot()["projects"][0]["name"] == project.name
    module = _module(Path(__file__).parents[1] / "workspace/new_project.py", "home_rename")
    renamed = module.rename_project(tmp_path, project, "Tokens")
    bridge.ack(tmp_path, request["id"], "done", project=f"projects/{renamed.name}")
    projects = view.snapshot()["projects"]
    assert [row["name"] for row in projects] == ["2026-10-19-tokens"]
    assert projects[0]["renamed_from"] == [project.name]
    assert projects[0]["request"] == request["id"]


def test_project_rename_follows_history_across_requests(tmp_path):
    from yap_studio.homepage import bridge

    (tmp_path / "inbox").mkdir()
    manifest = {"schema": "yap-studio/project@1", "name": "third",
                "takes": [{"id": "t1", "path": "takes/clip.mp4"}]}
    _json(tmp_path / "projects/third/project.json", manifest)
    for names in (("first", "second"), ("second", "third")):
        request = bridge.create_request(tmp_path, "edit", {})
        for name in names:
            bridge.ack(tmp_path, request["id"], "claimed", project=f"projects/{name}")
        bridge.ack(tmp_path, request["id"], "done")
    projects = state.HomeState(tmp_path).snapshot()["projects"]
    third = next(row for row in projects if row["name"] == "third")
    assert set(third["renamed_from"]) == {"second", "first"}


def test_project_rename_history_keeps_reused_names_separate(tmp_path):
    from yap_studio.homepage import bridge

    (tmp_path / "inbox").mkdir()
    for name in ("charlie", "echo"):
        _json(tmp_path / f"projects/{name}/project.json", {
            "schema": "yap-studio/project@1", "name": name,
            "takes": [{"id": "t1", "path": "takes/clip.mp4"}],
        })
    for previous, current in (("alpha", "bravo"), ("bravo", "charlie"),
                              ("delta", "bravo"), ("bravo", "echo")):
        request = bridge.create_request(tmp_path, "edit", {})
        bridge.ack(tmp_path, request["id"], "claimed", project=f"projects/{previous}")
        bridge.ack(tmp_path, request["id"], "done", project=f"projects/{current}")
    projects = {row["name"]: row for row in state.HomeState(tmp_path).snapshot()["projects"]}
    assert set(projects["charlie"]["renamed_from"]) == {"bravo", "alpha"}
    assert set(projects["echo"]["renamed_from"]) == {"bravo", "delta"}


def test_project_names_combine_only_requests_for_current_project(tmp_path):
    from yap_studio.homepage import bridge

    (tmp_path / "inbox").mkdir()
    _json(
        tmp_path / "projects/current/project.json",
        {
            "schema": "yap-studio/project@1",
            "name": "current",
            "takes": [{"id": "t1", "path": "takes/clip.mp4"}],
        },
    )
    for names in (("first", "second", "current"), ("second", "third", "current"), ("other",)):
        request = bridge.create_request(tmp_path, "edit", {})
        for name in names:
            bridge.ack(tmp_path, request["id"], "claimed", project=f"projects/{name}")
        bridge.ack(tmp_path, request["id"], "done")
    project = state.HomeState(tmp_path).snapshot()["projects"][0]
    assert project["renamed_from"] == ["first", "second", "third"]


@pytest.mark.parametrize(
    "history,expected",
    [
        ("projects/x", []),
        ([1, "projects/old"], ["old"]),
        ([None, {}, "old", "projects/", "projects/nested/name", "projects/.."], []),
    ],
)
def test_project_ignores_invalid_stored_history(tmp_path, history, expected):
    from yap_studio.homepage import bridge

    (tmp_path / "inbox").mkdir()
    _json(
        tmp_path / "projects/current/project.json",
        {
            "schema": "yap-studio/project@1",
            "name": "current",
            "takes": [{"id": "t1", "path": "takes/clip.mp4"}],
        },
    )
    row = bridge.create_request(tmp_path, "edit", {})
    row = bridge.ack(tmp_path, row["id"], "claimed", project="projects/current")
    row["ack"]["renamed_from"] = history
    bridge.write_json(bridge.directory(tmp_path) / "requests" / f"{row['id']}.json", row)
    project = state.HomeState(tmp_path).snapshot()["projects"][0]
    assert project["renamed_from"] == expected


def test_messages_are_grouped_oldest_first_with_replies(tmp_path):
    from yap_studio.homepage import bridge

    (tmp_path / "inbox").mkdir()
    for name in ("first", "other"):
        _json(
            tmp_path / f"projects/{name}/project.json",
            {
                "schema": "yap-studio/project@1",
                "name": name,
                "takes": [{"id": "t1", "path": "takes/clip.mp4"}],
            },
        )
    view = state.HomeState(tmp_path)
    initial = view.snapshot()
    assert initial["messages"] == []
    assert all(p["messages"] == [] for p in initial["projects"])
    first = bridge.create_request(
        tmp_path, "message", {"project": "projects/first", "text": "First question"}
    )
    global_row = bridge.create_request(
        tmp_path, "message", {"project": None, "text": "Workspace note"}
    )
    second = bridge.create_request(
        tmp_path, "message", {"project": "projects/first", "text": "Second question"}
    )
    other = bridge.create_request(
        tmp_path, "message", {"project": "projects/other", "text": "Other question"}
    )
    reply = bridge.ack(tmp_path, first["id"], "done", message="A plain reply")
    bridge.cancel(tmp_path, second["id"])
    # Message threads remain complete when the general request list reaches its limit.
    for _ in range(21):
        bridge.create_request(tmp_path, "learn", {})
    result = view.snapshot()
    projects = {p["name"]: p for p in result["projects"]}
    assert projects["first"]["messages"] == [
        {
            "id": first["id"],
            "text": "First question",
            "state": "done",
            "created_at": first["created_at"],
            "reply": "A plain reply",
            "replied_at": reply["ack"]["at"],
        },
        {
            "id": second["id"],
            "text": "Second question",
            "state": "cancelled",
            "created_at": second["created_at"],
            "reply": None,
            "replied_at": None,
        },
    ]
    assert [m["id"] for m in projects["other"]["messages"]] == [other["id"]]
    assert result["messages"] == [
        {
            "id": global_row["id"],
            "text": "Workspace note",
            "state": "pending",
            "created_at": global_row["created_at"],
            "reply": None,
            "replied_at": None,
        }
    ]


def test_real_versions_feedback_final_and_approval(feedback_project, tmp_path):
    from yap_studio.feedback.release import approve
    from yap_studio.feedback.versions import save_version

    (tmp_path / "inbox").mkdir()
    project = tmp_path / "projects/demo"
    project.parent.mkdir()
    feedback_project.rename(project)
    _json(project / "cut/verify.json", {"status": "PASS", "findings": []})
    save_version(project, now=lambda: datetime(2026, 1, 1, tzinfo=UTC))
    view = state.HomeState(tmp_path)
    assert view.snapshot()["projects"][0]["phase"] == "cut_ready"
    _json(project / "feedback/b1.json", {})
    assert view.snapshot()["projects"][0]["phase"] == "revising"
    final = project / "final/final.mp4"
    final.parent.mkdir()
    _render_media(final, 120, label="synthetic final")
    saved = save_version(project, media=final)
    result = view.snapshot()["projects"][0]
    assert result["phase"] == "final_ready"
    assert [v["final"] for v in result["versions"]] == [False, True]
    approve(project, saved.id)
    result = view.snapshot()["projects"][0]
    assert result["phase"] == "approved"
    assert result["approved"]["file"] == "projects/demo/versions/v2/media.mp4"
    assert view.approved_file("demo") == saved.media_path
    assert str(tmp_path) not in json.dumps(result)


def test_probe_and_thumbnail_are_cached_and_private(tmp_path):
    from yap_studio import ffmpeg

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _render_media(inbox / "clip.mp4", 30)
    view = state.HomeState(tmp_path)
    row = view.snapshot()["inbox"]["takes"][0]
    assert row["kind"] == "take" and row["duration_s"] == pytest.approx(1.0)
    thumbnail = view.thumbnail("clip.mp4")
    assert thumbnail is not None and thumbnail.stat().st_mode & 0o777 == 0o600
    assert ffmpeg.probe(thumbnail, count_frames=False).height == 360
    before = thumbnail.stat().st_mtime_ns
    assert view.thumbnail("clip.mp4").stat().st_mtime_ns == before
    assert view.thumbnail("../clip.mp4") is None


def test_sensitive_paths_in_answers_requests_and_messages_are_relative(tmp_path):
    from yap_studio.homepage import bridge

    (tmp_path / "inbox").mkdir()
    _json(
        tmp_path / "style/answers.json",
        {
            "schema": "yap-studio/style-answers@1",
            "finished_videos": {"answer": "yes", "videos": ["/private/example/movie.mp4"]},
        },
    )
    row = bridge.create_request(tmp_path, "learn", {})
    bridge.ack(
        tmp_path,
        row["id"],
        "claimed",
        project=str(tmp_path / "projects/demo"),
        message="Read /private/example/movie.mp4 first",
    )
    result = state.HomeState(tmp_path).snapshot()
    assert "/private/example" not in json.dumps(result)
    assert str(tmp_path) not in json.dumps(result)


def test_invalid_ledger_is_actionable(tmp_path):
    from yap_studio.errors import ValidationError

    (tmp_path / "inbox").mkdir()
    (tmp_path / "inbox/.used.json").write_text("invalid")
    with pytest.raises(ValidationError, match="Fix inbox/.used.json"):
        state.HomeState(tmp_path).snapshot()


def test_review_link_stays_on_this_mac_when_served_through_tailscale(tmp_path, monkeypatch):
    from yap_studio import background

    (tmp_path / "inbox").mkdir()
    project = tmp_path / "projects/demo"
    project.mkdir(parents=True)
    tailnet = "https://box.tailnet.example.invalid:18001/?t=abc"
    local = "http://127.0.0.1:52000/?t=abc"
    saved = {
        "pid": 1, "media": [str(project)],
        "document": {"videos": [{"url": tailnet, "local_url": local}],
                     "phone": {"via": "tailscale", "url": tailnet}},
    }
    monkeypatch.setattr(background, "records", lambda kind: ["record"])
    monkeypatch.setattr(background, "read_record", lambda record: saved)
    monkeypatch.setattr(background, "command_matches", lambda pid, record: True)
    home = state.HomeState(tmp_path)
    assert home._review(project.resolve()) == {
        "url": local, "phone_url": tailnet, "phone_via": "tailscale",
    }
    del saved["document"]["videos"][0]["local_url"]
    assert home._review(project.resolve()) == {
        "url": tailnet, "phone_url": tailnet, "phone_via": "tailscale",
    }
