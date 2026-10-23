"""Current pointers, snapshot rollback, approval reports and concurrent edit guards."""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path

import pytest
from test_feedback_versions import NOW, _json, _update_cut
from test_feedback_versions import project as project

from yap_studio.errors import InputError
from yap_studio.feedback import release
from yap_studio.feedback.release import (
    approval_record,
    approve,
    approved_version,
    clear_approval,
    rollback,
)
from yap_studio.feedback.versions import VersionError, current_version, save_version


def _changed_rollback_project(project: Path) -> tuple:
    _json(project / "removals.json", {"schema": "yap-studio/removals@1", "removals": []})
    first = save_version(project)
    _json(project / "edit.json", {"schema": "yap-studio/edit@1", "words_sha256": {},
                                 "segments": [], "end_hold_s": 0.5})
    _json(project / "removals.json", {"schema": "yap-studio/removals@1", "removals": [{
        "id": "r1", "take": "t1", "start": 1, "end": 2, "kind": "manual", "status": "applied",
    }]})
    _update_cut(project)
    second = save_version(project)
    _json(project / "versions" / "current.json", {
        "schema": "yap-studio/version-pointer@1", "current": second.id,
        "set_at": "2026-10-05T22:00:00Z", "reason": "rollback",
    })
    return first, second


def test_regression_rollback_second_replacement_failure_restores_all_inputs(
    project: Path, monkeypatch,
) -> None:
    _, second = _changed_rollback_project(project)
    watched = [project / "edit.json", project / "removals.json",
               project / "versions" / "current.json"]
    before = {file: file.read_bytes() for file in watched}
    original = os.replace
    count = 0

    def fail_second(source, target):
        nonlocal count
        if Path(target) in watched[:2]:
            count += 1
            if count == 2:
                raise OSError("second replacement failed")
        return original(source, target)

    monkeypatch.setattr(os, "replace", fail_second)
    with pytest.raises(OSError, match="second replacement"):
        rollback(project, "v1")
    assert {file: file.read_bytes() for file in watched} == before
    assert current_version(project) == second
    assert not (project / "versions" / ".rollback.json").exists()
    assert not (project / "versions" / ".rollback").exists()


def test_regression_interrupted_rollback_completes_forward(project: Path, monkeypatch) -> None:
    first, _ = _changed_rollback_project(project)
    original = os.replace
    count = 0

    class ProcessDied(BaseException):
        pass

    def die_second(source, target):
        nonlocal count
        if Path(target) in {project / "edit.json", project / "removals.json"}:
            count += 1
            if count == 2:
                raise ProcessDied()
        return original(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", die_second)
        with pytest.raises(ProcessDied):
            rollback(project, "v1")
    # Older code has no recovery entry point and leaves the second input unrestored.
    complete = getattr(release, "complete_pending_rollback", lambda _: None)
    result = complete(project)
    for name in ("edit.json", "removals.json"):
        assert json.loads((project / name).read_text()) == json.loads(
            (first.path / name).read_text(),
        )
    assert result["version"] == "v1" and current_version(project) == first
    assert complete(project) is None
    assert not (project / "versions" / ".rollback.json").exists()
    assert not (project / "versions" / ".rollback").exists()


def test_regression_rollback_pointer_failure_restores_all_inputs(
    project: Path, monkeypatch,
) -> None:
    _changed_rollback_project(project)
    watched = [project / "edit.json", project / "removals.json",
               project / "versions" / "current.json"]
    before = {file: file.read_bytes() for file in watched}
    original = release.write_json_cas

    def pointer_failed(path, data, *, base_sha256):
        result = original(path, data, base_sha256=base_sha256)
        if path.name == "current.json":
            raise OSError("pointer publication failed")
        return result

    monkeypatch.setattr(release, "write_json_cas", pointer_failed)
    with pytest.raises(OSError, match="pointer publication"):
        rollback(project, "v1")
    assert {file: file.read_bytes() for file in watched} == before
    assert not (project / "versions" / ".rollback.json").exists()


def _leave_pending_rollback(project: Path, monkeypatch) -> None:
    original = os.replace

    class ProcessDied(BaseException):
        pass

    def interrupted(source, target):
        if Path(target) == project / "removals.json":
            raise ProcessDied()
        return original(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupted)
        with pytest.raises(ProcessDied):
            rollback(project, "v1")


def test_pending_rollback_conflict_restores_other_inputs(project: Path, monkeypatch) -> None:
    _changed_rollback_project(project)
    edit = (project / "edit.json").read_bytes()
    pointer = (project / "versions" / "current.json").read_bytes()
    _leave_pending_rollback(project, monkeypatch)
    removals = json.loads((project / "removals.json").read_text())
    removals["removals"][0]["end"] = 3
    _json(project / "removals.json", removals)
    conflict = (project / "removals.json").read_bytes()
    with pytest.raises(VersionError, match="conflicts: removals.json"):
        release.complete_pending_rollback(project)
    assert (project / "edit.json").read_bytes() == edit
    assert (project / "removals.json").read_bytes() == conflict
    assert (project / "versions" / "current.json").read_bytes() == pointer
    assert not (project / "versions" / ".rollback.json").exists()
    assert not (project / "versions" / ".rollback").exists()


@pytest.mark.parametrize("action", ["rollback", "save", "approve", "review"])
def test_project_actions_complete_pending_rollback(project: Path, monkeypatch, action) -> None:
    from yap_studio.review.launch import version_video
    from yap_studio.review.store import ReviewStore

    first, _ = _changed_rollback_project(project)
    _leave_pending_rollback(project, monkeypatch)
    if action == "rollback":
        rollback(project, "v1")
    elif action == "save":
        with pytest.raises(VersionError, match="edit.json changed"):
            save_version(project)
    elif action == "approve":
        with pytest.raises(VersionError, match="no check report"):
            approve(project, "v1")
    else:
        version_video([str(project)], ReviewStore(project.parent / "reviews"), lambda _: [])
    assert current_version(project) == first
    assert json.loads((project / "removals.json").read_text()) == {
        "schema": "yap-studio/removals@1", "removals": [],
    }
    assert not (project / "versions" / ".rollback.json").exists()


@pytest.mark.parametrize("qa,verify", [({}, {"status": "pass"}),
                                     ({"status": "unknown"}, {"status": "pass"}),
                                     ({"status": "pass"}, {}),
                                     ({"status": "pass"}, {"status": "fail"})])
def test_regression_approval_requires_valid_status_from_every_report(
    project: Path, qa: dict, verify: dict,
) -> None:
    _json(project / "qa" / "report.json", qa)
    _json(project / "cut" / "verify.json", verify)
    save_version(project)
    with pytest.raises(VersionError, match="no valid status|failed its checks"):
        approve(project, "v1")
    assert approved_version(project) is None


def test_regression_approval_records_worst_present_status(project: Path) -> None:
    _json(project / "qa" / "report.json", {"status": "PASS"})
    _json(project / "cut" / "verify.json", {"status": "WaRn"})
    save_version(project)
    assert approve(project, "v1")["report_status"] == "warn"


def test_rollback_restores_inputs_reports_lanes_and_preserves_noop(project: Path) -> None:
    removals = {"schema": "yap-studio/removals@1", "removals": [{
        "id": "r1", "take": "t1", "start": 1, "end": 2, "kind": "manual", "status": "applied",
    }]}
    _json(project / "removals.json", removals)
    _json(project / "timeline" / "agent.json", {"elements": []})
    _json(project / "timeline" / "broll.json", {"elements": []})
    first = save_version(project)
    _json(project / "edit.json", {"schema": "yap-studio/edit@1", "words_sha256": {},
                                 "segments": [], "end_hold_s": 0.5})
    _json(project / "removals.json", {"schema": "yap-studio/removals@1", "removals": []})
    _json(project / "timeline" / "agent.json", {"elements": ["changed"]})
    _json(project / "timeline" / "extra.json", {"elements": []})
    _update_cut(project)
    second = save_version(project)
    assert current_version(project) == second
    report = rollback(project, "v1", now=lambda: NOW)
    assert report == {"version": "v1", "restored": ["edit.json", "removals.json"],
                      "not_restored": ["timeline/agent.json", "timeline/extra.json"],
                      "cut_stale": True}
    assert json.loads((project / "edit.json").read_text()) == json.loads(
        (first.path / "edit.json").read_text(),
    )
    assert json.loads((project / "removals.json").read_text()) == removals
    assert json.loads((project / "timeline" / "agent.json").read_text()) == {
        "elements": ["changed"],
    }
    pointer = project / "versions" / "current.json"
    assert json.loads(pointer.read_text()) == {"schema": "yap-studio/version-pointer@1",
                                             "current": "v1", "set_at": "2026-10-05T22:00:00Z",
                                             "reason": "rollback"}
    assert current_version(project) == first
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (
        pointer, project / "edit.json", project / "removals.json",
    )}
    assert rollback(project, "v1", now=lambda: NOW + timedelta(hours=1)) == report
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before} == before
    _update_cut(project, label="third render")
    third = save_version(project)
    assert current_version(project) == third and not pointer.exists()


def test_rollback_without_removals_snapshot_clears_current_removals(project: Path) -> None:
    save_version(project)
    _json(project / "removals.json", {"schema": "yap-studio/removals@1", "removals": [{
        "id": "r1", "take": "t1", "start": 1, "end": 2, "kind": "manual", "status": "applied",
    }]})
    assert rollback(project, "v1")["restored"] == ["edit.json", "removals.json"]
    assert json.loads((project / "removals.json").read_text()) == {
        "schema": "yap-studio/removals@1", "removals": [],
    }


def test_rollback_can_restore_corrupt_current_json(project: Path) -> None:
    first = save_version(project)
    (project / "edit.json").write_text('{broken\n')
    assert rollback(project, "v1")["restored"] == ["edit.json"]
    assert json.loads((project / "edit.json").read_text()) == json.loads(
        (first.path / "edit.json").read_text(),
    )


def test_rollback_refuses_concurrent_edit_before_pointer_write(project: Path, monkeypatch) -> None:
    save_version(project)
    _json(project / "edit.json", {"schema": "yap-studio/edit@1", "words_sha256": {},
                                 "segments": [], "end_hold_s": 0.5})
    _update_cut(project)
    second = save_version(project)
    original = Path.write_bytes

    def concurrent(path, data):
        if path.name == "edit.json.staged":
            _json(project / "edit.json", {
                "schema": "yap-studio/edit@1", "words_sha256": {}, "segments": [],
                         "end_hold_s": 1.0})
        return original(path, data)

    monkeypatch.setattr(Path, "write_bytes", concurrent)
    with pytest.raises(VersionError, match="targets changed: edit.json"):
        rollback(project, "v1")
    assert current_version(project) == second
    assert json.loads((project / "edit.json").read_text())["end_hold_s"] == 1.0
    assert not (project / "versions" / "current.json").exists()


@pytest.mark.parametrize("snapshot", ["edit.json", "removals.json", "timeline/agent.json",
                                      "verify.json", "qa-report.json"])
def test_rollback_refuses_corrupt_snapshot_before_writing(project: Path, snapshot: str) -> None:
    _json(project / "removals.json", {"schema": "yap-studio/removals@1", "removals": []})
    _json(project / "timeline" / "agent.json", {"elements": []})
    _json(project / "cut" / "verify.json", {"status": "pass"})
    _json(project / "qa" / "report.json", {"status": "pass"})
    first = save_version(project)
    (first.path / snapshot).write_text('{}\n')
    before = (project / "edit.json").read_bytes()
    with pytest.raises(InputError, match="snapshot changed"):
        rollback(project, "v1")
    assert (project / "edit.json").read_bytes() == before
    assert not (project / "versions" / "current.json").exists()


def test_approve_missing_and_failed_reports_and_replace_approval(project: Path) -> None:
    assert current_version(project) is None and approved_version(project) is None
    first = save_version(project)
    with pytest.raises(VersionError, match="This version has no check report; run the checks"):
        approve(project, "v1")
    _json(project / "cut" / "verify.json", {"status": "FAIL", "findings": [
        {"message": "A required check failed"},
    ]})
    _update_cut(project)
    save_version(project)
    with pytest.raises(VersionError, match="This version failed its checks: A required check"):
        approve(project, "v2")
    _json(project / "cut" / "verify.json", {"status": "WARN"})
    _update_cut(project, label="third render")
    third = save_version(project)
    assert approve(project, "v3", now=lambda: NOW) == {
        "schema": "yap-studio/version-approval@1", "version": "v3",
        "media_sha256": third.media["sha256"], "approved_at": "2026-10-05T22:00:00Z",
        "report_status": "warn",
    }
    assert approved_version(project) == third
    _json(project / "cut" / "verify.json", {"status": "PASS"})
    _update_cut(project, label="fourth render")
    fourth = save_version(project)
    approve(project, "v4")
    assert approved_version(project) == fourth
    assert first.id == "v1"
    clear_approval(project)
    assert approved_version(project) is None
    clear_approval(project)


@pytest.mark.parametrize("changed", ["record", "media"])
def test_approval_hash_mismatch_is_not_approved(project: Path, changed: str) -> None:
    _json(project / "cut" / "verify.json", {"status": "pass"})
    first = save_version(project)
    approve(project, "v1")
    if changed == "record":
        path = project / "versions" / "approval.json"
        data = json.loads(path.read_text())
        _json(path, {**data, "media_sha256": "f" * 64})
    else:
        first.media_path.write_bytes(b"changed")
    assert approved_version(project) is None


def test_approval_readers_can_skip_media_hash_but_still_match_manifest(
    project: Path, monkeypatch,
) -> None:
    _json(project / "cut" / "verify.json", {"status": "pass"})
    first = save_version(project)
    approved = approve(project, "v1")
    media_hash_calls = []
    original = release.sha256_file

    def counted_hash(path):
        if path == first.media_path:
            media_hash_calls.append(path)
        return original(path)

    monkeypatch.setattr(release, "sha256_file", counted_hash)
    assert approval_record(project, verify_media=False) == approved
    assert approved_version(project, verify_media=False) == first
    assert len(media_hash_calls) == 0
    assert approval_record(project) == approved
    assert approved_version(project) == first
    assert media_hash_calls == [first.media_path] * 2
    _json(project / "versions" / "approval.json", {**approved, "media_sha256": "f" * 64})
    assert approval_record(project, verify_media=False) is None
    assert approved_version(project, verify_media=False) is None
    assert media_hash_calls == [first.media_path] * 2
