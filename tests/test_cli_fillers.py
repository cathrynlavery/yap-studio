"""CLI proposal, apply and locked creator-decision flows."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from test_fillers import install_engine, make_project

from yap_studio import receipt
from yap_studio.cli import main
from yap_studio.project.status import ProjectContext, fillers_inputs, status
from yap_studio.stages import fillers as stage


def test_fillers_defaults_to_proposals_then_apply(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    assert main(["fillers", str(root), "--json"]) == 0
    proposed = json.loads(capsys.readouterr().out)
    assert [(item["id"], item["kind"], item["status"]) for item in proposed["entries"]] == [
        ("r1", "filler", "proposed"), ("r2", "silence", "proposed"),
    ]
    assert main(["fillers", str(root), "--apply"]) == 0
    output = capsys.readouterr().out
    assert output.splitlines() == [
        "r1 filler applied um: pass",
        "r2 silence applied pause: pass",
        "Fillers: 2 applied, 0 proposed, 0 rejected, 0 skipped.",
    ]


def test_fillers_force_retests_machine_rejection_with_same_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch, clipped=True)
    assert main(["fillers", str(root), "--apply", "--json"]) == 0
    rejected_entries = json.loads(capsys.readouterr().out)["entries"]
    assert [(item["id"], item["kind"], item["status"]) for item in rejected_entries] == [
        ("r1", "filler", "rejected"), ("r2", "silence", "proposed"),
    ]
    assert rejected_entries[1]["evidence"]["words"] == "unverifiable"
    assert rejected_entries[1]["listen"]
    rejected = rejected_entries[0]
    batches = install_engine(monkeypatch)
    assert main(["fillers", str(root), "--apply", "--force", "--json"]) == 0
    applied_entries = json.loads(capsys.readouterr().out)["entries"]
    assert [(item["id"], item["kind"], item["status"]) for item in applied_entries] == [
        ("r1", "filler", "applied"), ("r2", "silence", "applied"),
    ]
    applied = applied_entries[0]
    assert applied["status"] == "applied" and applied["id"] == rejected["id"]
    assert [[path.stem for path in batch] for batch in batches] == [
        [f"round-0-{index}-{role}" for index in range(2) for role in ("cut", "control")],
        [f"round-1-{index}-{role}" for index in range(4) for role in ("cut", "control")],
    ]


def test_fillers_set_r3_rejected_changes_only_that_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    assert main(["fillers", str(root), "--apply", "--json"]) == 0
    capsys.readouterr()
    for first, last in ((2.6, 2.7), (2.8, 2.9)):
        assert main(["fillers", "add", str(root), "--manual", "t1", str(first), str(last),
                     "--json"]) == 0
        capsys.readouterr()
    target = root / "removals.json"
    before = json.loads(target.read_bytes())["removals"]
    def entry_bytes(payload: bytes, identifier: str) -> bytes:
        """Extract an entry independently from its first ID field."""
        text = payload.decode()
        match = re.search(r'\{\s*"id":\s*"' + identifier + '"', text)
        assert match is not None
        _, end = json.JSONDecoder().raw_decode(text, match.start())
        return text[match.start():end].encode()

    untouched = [entry_bytes(target.read_bytes(), item["id"])
                 for item in before if item["id"] != "r3"]
    assert main(["fillers", "set", str(root), "r3", "rejected", "--reason", "keep this beat",
                 "--json"]) == 0
    changed = json.loads(capsys.readouterr().out)["removals"]
    assert [entry_bytes(target.read_bytes(), item["id"])
            for item in changed if item["id"] != "r3"] == untouched
    assert next(item for item in changed if item["id"] == "r3")["reason"] == "keep this beat"
    owned = json.loads((root / "fillers/receipt.json").read_bytes())
    assert next(item for item in owned["outputs"] if item["path"] == "removals.json")["sha256"] == (
        receipt.sha256_file(target)
    )


@pytest.mark.parametrize("shift", [1, -1], ids=["inward", "outward"])
@pytest.mark.parametrize("decision", ["applied", "rejected"])
def test_fillers_set_inspection_proposal_updates_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    shift: int, decision: str,
) -> None:
    from yap_studio.stages.cut import resolve_cut

    root = make_project(tmp_path / "project")
    candidate = stage.Candidate("t1", "filler", ("t1:2",), 1.1, 1.4, (), "s1", False, {})
    monkeypatch.setattr(stage, "_proposals", lambda *_, **__: [candidate])
    install_engine(monkeypatch)
    saved = stage.fillers(root, apply=True)["entries"][0]

    def add_proposal(data: dict) -> str:
        """Seed a saved inspection proposal without changing the applied interval."""
        data["removals"].append({
            **saved, "id": "r2", "start": 1.1 + shift / 30, "status": "proposed",
            "evidence": {"applied_parent": saved["id"]},
        })
        return "r2"

    stage._change(root, add_proposal)
    other_id = stage.add_manual(root, "t1", 2.7, 2.8, reason="chosen trim")["id"]
    unrelated = next(item for item in stage.set_removal(
        root, other_id, "rejected", reason="keep this beat",
    )["removals"] if item["id"] == other_id)
    owned_before = json.loads((root / "fillers/receipt.json").read_bytes())
    pieces = resolve_cut(root).pieces
    assert [(left.src_f1, right.src_f0)
            for left, right in zip(pieces, pieces[1:], strict=False)] == [(33, 42)]
    assert main(["fillers", "set", str(root), "r2", decision, "--json"]) == 0
    entries = json.loads(capsys.readouterr().out)["removals"]
    parent = next(item for item in entries if item["id"] == saved["id"])
    if decision == "applied":
        assert parent["status"] == "rejected"
        assert parent["reason"] == "replaced by r2"
        assert parent["evidence"] == {**saved["evidence"], "decision": True, "replaced_by": "r2"}
        assert (parent["start"], parent["end"]) == (saved["start"], saved["end"])
    else:
        assert parent == saved
    assert next(item for item in entries if item["id"] == "r2")["status"] == decision
    assert next(item for item in entries if item["id"] == other_id) == unrelated
    assert len({item["id"] for item in entries}) == len(entries)
    pieces = resolve_cut(root).pieces
    assert [(left.src_f1, right.src_f0)
            for left, right in zip(pieces, pieces[1:], strict=False)] == [
                (33 + shift if decision == "applied" else 33, 42),
            ]
    owned = json.loads((root / "fillers/receipt.json").read_bytes())
    assert owned["inputs"] == owned_before["inputs"]
    assert next(item for item in owned["outputs"] if item["path"] == "removals.json")["sha256"] == (
        receipt.sha256_file(root / "removals.json")
    )
    assert next(item for item in status(root)["stages"] if item["name"] == "fillers")["state"] == (
        "stale"
    )
    if decision == "applied":
        rerun = stage.fillers(root, apply=True, force=True)["entries"]
        assert {item["id"]: item for item in rerun} == {item["id"]: item for item in entries}


def test_rerun_after_set_and_manual_add_keeps_both_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    assert main(["fillers", str(root), "--apply", "--json"]) == 0
    capsys.readouterr()
    assert main(["fillers", "set", str(root), "r1", "rejected", "--reason", "keep phrase"]) == 0
    capsys.readouterr()
    fillers_state = next(item for item in status(root)["stages"] if item["name"] == "fillers")
    assert fillers_state["state"] == "stale"
    assert main(["fillers", "add", str(root), "--manual", "t1", "2.7", "2.8",
                 "--reason", "chosen trim"]) == 0
    capsys.readouterr()
    assert main(["fillers", str(root), "--apply", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert not result["noop"]
    assert [(item["id"], item["kind"], item["status"], item.get("reason"))
            for item in result["entries"]] == [
                ("r1", "filler", "rejected", "keep phrase"),
                ("r2", "silence", "applied", None),
                ("r3", "manual", "applied", "chosen trim"),
            ]
    assert main(["fillers", str(root), "--apply", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["noop"]


def test_unknown_removal_id_exits_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                      capsys: pytest.CaptureFixture[str]) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    assert main(["fillers", str(root), "--json"]) == 0
    capsys.readouterr()
    before = (root / "removals.json").read_bytes()
    assert main(["fillers", "set", str(root), "r99", "rejected", "--json"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "invalid_input"
    assert (root / "removals.json").read_bytes() == before


def test_hand_edited_removals_names_supported_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    assert main(["fillers", str(root), "--json"]) == 0
    capsys.readouterr()
    target = root / "removals.json"
    target.write_bytes(target.read_bytes() + b"\n")
    monkeypatch.setattr(stage, "resolve_backend", lambda *args: pytest.fail(
        "speech engine should not start for hand-edited removals",
    ))

    assert main(["fillers", str(root), "--apply"]) == 3
    stderr = capsys.readouterr().err
    assert "removals.json was changed outside yap-studio" in stderr
    assert "yap-studio fillers set DIR ID STATUS" in stderr
    assert "yap-studio fillers add DIR --manual TAKE START END" in stderr

    for command in (["fillers", str(root), "--apply", "--json"],
                    ["fillers", "set", str(root), "r1", "rejected", "--json"],
                    ["fillers", "add", str(root), "--manual", "t1", "2.7", "2.8", "--json"]):
        assert main(command) == 3
        error = json.loads(capsys.readouterr().err)["error"]
        assert error["code"] == "unsafe_path"
        assert "yap-studio fillers set DIR ID STATUS" in error["suggested_action"]
        assert "yap-studio fillers add DIR --manual TAKE START END" in error["suggested_action"]


def test_manual_add_before_first_run_survives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    assert main(["fillers", "add", str(root), "--manual", "t1", "2.7", "2.8", "--json"]) == 0
    capsys.readouterr()
    assert main(["fillers", str(root), "--apply", "--json"]) == 0
    assert any(item["id"] == "r1" and item["by"] == "manual"
               for item in json.loads(capsys.readouterr().out)["entries"])


@pytest.mark.parametrize("force", [False, True])
def test_explicit_applied_decision_survives_proposal_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    force: bool,
) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    assert main(["fillers", str(root), "--json"]) == 0
    capsys.readouterr()
    assert main(["fillers", "set", str(root), "r1", "applied", "--json"]) == 0
    capsys.readouterr()
    rerun = ["fillers", str(root), "--json", *(["--force"] if force else [])]
    assert main(rerun) == 0
    assert json.loads(capsys.readouterr().out)["entries"][0]["status"] == "applied"


def test_forced_manual_inspection_keeps_applied_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from yap_studio.stages.cut import resolve_cut

    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    assert main(["fillers", str(root), "--apply", "--json"]) == 0
    capsys.readouterr()
    assert main(["fillers", "add", str(root), "--manual", "t1", "2.2", "2.5", "--json"]) == 0
    added = json.loads(capsys.readouterr().out)["id"]
    monkeypatch.setattr("yap_studio.stages.fillers._proposals", lambda *_, **__: [])
    from yap_studio.transcript import ScriptedBackend, Transcript, Word

    engine = ScriptedBackend(lambda target: Transcript([
        Word("garden", 1.05, 1.15), Word("flowers", 1.2, 1.3),
        *([Word("grow", 1.4, 1.5)] if target.stem.endswith("control") else []),
    ]))
    monkeypatch.setattr("yap_studio.stages.fillers.resolve_backend", lambda *_: engine)
    assert main(["fillers", str(root), "--force", "--json"]) == 0
    entries = json.loads(capsys.readouterr().out)["entries"]
    manual = next(item for item in entries if item["id"] == added)
    assert manual["status"] == "applied" and manual["by"] == "manual"
    assert manual["evidence"]["words"] == "pass"
    assert "t1:5" not in {word_id for piece in resolve_cut(root).pieces for word_id in piece.words}


def test_repeating_same_status_is_a_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    assert main(["fillers", str(root), "--json"]) == 0
    capsys.readouterr()
    before = (root / "removals.json").read_bytes()
    owned = (root / "fillers/receipt.json").read_bytes()
    assert main(["fillers", "set", str(root), "r1", "proposed", "--json"]) == 0
    capsys.readouterr()
    assert (root / "removals.json").read_bytes() == before
    assert (root / "fillers/receipt.json").read_bytes() == owned
    fillers_state = next(item for item in status(root)["stages"] if item["name"] == "fillers")
    assert fillers_state["state"] == "done"


def test_no_engine_human_output_reports_skipped_count(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    root = make_project(tmp_path / "project")
    monkeypatch.setattr("yap_studio.stages.fillers.resolve_backend", lambda *_: None)
    assert main(["fillers", str(root)]) == 0
    output = capsys.readouterr().out
    assert "fillers skipped: no speech engine (1)" in output
    assert "1 skipped" in output


def test_max_pause_persists_from_proposal_through_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    assert ProjectContext(root).project.rules["max_pause_s"] == 0.3

    assert main(["fillers", str(root), "--max-pause", "0.1", "--json"]) == 0
    proposal = json.loads(capsys.readouterr().out)
    assert any(item["kind"] == "silence" and item["status"] == "proposed"
               for item in proposal["entries"])
    assert json.loads((root / "project.json").read_bytes())["rules"]["max_pause_s"] == 0.1
    proposed_inputs = json.loads((root / "fillers/receipt.json").read_bytes())["inputs"]
    assert proposed_inputs == fillers_inputs(ProjectContext(root))

    assert main(["fillers", str(root), "--apply", "--json"]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert not applied["noop"]
    assert any(item["kind"] == "silence" and item["status"] == "applied"
               for item in applied["entries"])
    assert json.loads((root / "project.json").read_bytes())["rules"]["max_pause_s"] == 0.1
    applied_inputs = json.loads((root / "fillers/receipt.json").read_bytes())["inputs"]
    assert applied_inputs["rules"] == proposed_inputs["rules"]
    assert applied_inputs == fillers_inputs(ProjectContext(root))
    assert next(item for item in status(root)["stages"]
                if item["name"] == "fillers")["state"] == "done"


def test_same_max_pause_leaves_project_bytes_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    project_path = root / "project.json"
    before = project_path.read_bytes()
    assert ProjectContext(root).project.rules["max_pause_s"] == 0.3

    assert main(["fillers", str(root), "--max-pause", "0.3", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["entries"]
    assert project_path.read_bytes() == before


def test_max_pause_conflicts_if_rule_changes_before_fillers_context_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    root = make_project(tmp_path / "project")
    project_path = root / "project.json"
    original_context = stage.ProjectContext
    loads = []

    def second_writer(project_dir: Path) -> ProjectContext:
        saved = json.loads(project_path.read_bytes())
        assert saved["rules"]["max_pause_s"] == 0.1
        saved["rules"]["max_pause_s"] = 0.2
        project_path.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
        loads.append(True)
        return original_context(project_dir)

    monkeypatch.setattr(stage, "ProjectContext", second_writer)
    assert main(["fillers", str(root), "--max-pause", "0.1", "--json"]) == 3
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "base_changed"
    assert loads == [True]
    assert json.loads(project_path.read_bytes())["rules"]["max_pause_s"] == 0.2
    assert not (root / "removals.json").exists()
    assert not any((root / "fillers" / name).exists() for name in (
        "receipt.json", "receipt.pending.json", "progress.json", "report.json",
    ))


@pytest.mark.parametrize("value", ["-0.1", "nan", "inf"])
def test_invalid_max_pause_preserves_project_and_fillers_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    value: str,
) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    assert main(["fillers", str(root), "--json"]) == 0
    capsys.readouterr()
    targets = [root / "project.json", root / "removals.json", root / "fillers/receipt.json"]
    before = [target.read_bytes() for target in targets]

    assert main(["fillers", str(root), "--max-pause", value, "--json"]) != 0
    assert all(target.read_bytes() == saved for target, saved in zip(targets, before, strict=True))
    assert "unrecognized arguments" not in capsys.readouterr().err


def test_active_fillers_rejects_max_pause_without_changing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    assert main(["fillers", str(root), "--json"]) == 0
    capsys.readouterr()
    targets = [root / "project.json", root / "removals.json", root / "fillers/receipt.json"]
    before = [target.read_bytes() for target in targets]
    (root / "fillers/progress.json").write_text(
        json.dumps({"pid": os.getpid(), "step": "testing"}) + "\n", encoding="utf-8",
    )

    assert main(["fillers", str(root), "--max-pause", "0.1", "--json"]) == 2
    assert all(target.read_bytes() == saved for target, saved in zip(targets, before, strict=True))
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "stage_running"


def test_tampered_removals_rejects_max_pause_without_changing_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    assert main(["fillers", str(root), "--json"]) == 0
    capsys.readouterr()
    project = root / "project.json"
    before = project.read_bytes()
    removals = root / "removals.json"
    removals.write_bytes(removals.read_bytes() + b"\n")
    receipt_before = (root / "fillers/receipt.json").read_bytes()

    assert main(["fillers", str(root), "--max-pause", "0.1", "--json"]) == 3
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "unsafe_path"
    assert project.read_bytes() == before
    assert (root / "fillers/receipt.json").read_bytes() == receipt_before
