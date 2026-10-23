"""CLI timeline contracts and refusal exit codes using synthetic media only."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_timeline import add, fragment, graphic, insert, overlay
from test_timeline import project as project

from yap_studio import cli


@pytest.mark.parametrize("kind", ["insert", "overlay"])
def test_hand_authored_fragment_add_show_check_remove(
    project: Path, capsys: pytest.CaptureFixture[str], kind: str,
) -> None:
    file = fragment(project, (overlay if kind == "overlay" else insert)(project))
    assert cli.main(["timeline", "add", str(file), str(project), "--lane", "broll", "--json"]) == 0
    added = json.loads(capsys.readouterr().out)
    assert added["valid"] and added["elements"][0]["id"] == "broll-one"
    assert cli.main(["timeline", "show", str(project), "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert (shown["elements"][0]["start_frame"], shown["elements"][0]["end_frame"]) == (66, 108)
    if kind == "overlay":
        assert shown["elements"][0]["box_px"] == [60, 280, 960, 360]
    assert cli.main(["timeline", "check", str(project)]) == 0
    assert "passed" in capsys.readouterr().out
    assert cli.main(["timeline", "remove", str(project), "--lane", "broll",
                     "--id", "broll-one", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["removed"] == ["broll-one"]


def test_add_replace_lane_flag(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    first = fragment(project, insert(project, id="broll-old"))
    assert cli.main(["timeline", "add", str(first), str(project), "--lane", "broll"]) == 0
    capsys.readouterr()
    second = fragment(project, insert(project, id="broll-new"), name="replacement.json")
    assert cli.main(["timeline", "add", str(second), str(project), "--lane", "broll",
                     "--replace-lane"]) == 0
    assert [item["id"] for item in json.loads((project / "timeline/broll.json").read_bytes())[
        "elements"]] == ["broll-new"]


@pytest.mark.parametrize("kind", ["insert", "overlay"])
def test_overlap_exits_three_and_prints_both_lanes(
    project: Path, capsys: pytest.CaptureFixture[str], kind: str,
) -> None:
    add(project, graphic(start={"word": "t1:2"}), lane="agent").require_valid()
    file = fragment(project, (overlay if kind == "overlay" else insert)(project))
    assert cli.main(["timeline", "add", str(file), str(project), "--lane", "broll", "--json"]) == 3
    result = json.loads(capsys.readouterr().out)
    assert not result["valid"]
    assert result["findings"][0]["rule"] == "visual_overlap"
    assert set(result["findings"][0]["lanes"]) == {"agent", "broll"}
    assert not (project / "timeline/broll.json").exists()


def test_warning_exits_zero_and_prints_warning(
    project: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    item = insert(project, start={"time": 0}, end={"time": 0.5})
    assert cli.main(["timeline", "add", str(fragment(project, item)), str(project),
                     "--lane", "broll"]) == 0
    assert "warning:" in capsys.readouterr().out
    assert (project / "timeline/broll.json").is_file()


def test_text_collapses_repeated_findings_but_json_keeps_each(
    project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.stages import timeline as stage

    original = stage.resolve_timeline
    actions = ["Move media into the project."] * 3

    def repeated(root: Path) -> stage.ResolvedTimeline:
        found = original(root)
        template = stage.TimelineFinding(
            "external_media", "warning",
            "broll-1 (lane broll): insert media is outside the project folder",
            ("broll-1",), ("broll",), "Move media into the project.",
        )
        return stage.ResolvedTimeline(found.elements, tuple(
            stage.replace(template, message=(f"broll-{i} (lane broll): insert media is "
                                             "outside the project folder"),
                          elements=(f"broll-{i}",), suggested_action=actions[i - 1])
            for i in range(1, 4)),
            found.snapped_anchors, found.total_frames, found.end_hold_frames, found.beat_snaps)

    monkeypatch.setattr(stage, "resolve_timeline", repeated)
    assert cli.main(["timeline", "check", str(project)]) == 0
    output = capsys.readouterr().out
    assert output.count("insert media is outside the project folder") == 1
    assert "3 elements: broll-1, broll-2, broll-3" in output
    assert output.count("fix:") == 1
    assert "fix: Move media into the project." in output
    assert "the same for the other" not in output
    actions[:] = [f"Move 'broll-{i}' into the project." for i in range(1, 4)]
    assert cli.main(["timeline", "check", str(project)]) == 0
    output = capsys.readouterr().out
    assert output.count("fix:") == 1
    assert "fix: Move 'broll-1' into the project.; the same for the other 2 elements" in output
    assert cli.main(["timeline", "check", str(project), "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["findings"]) == 3


def test_text_keeps_near_duplicate_groups_separate_and_collapses_single_warnings(
    project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.stages import timeline as stage

    original = stage.resolve_timeline
    findings = []
    for ids in (("broll-a", "broll-b"), ("broll-c", "broll-d", "broll-e")):
        names = ", ".join(f"{name} (lane broll)" for name in ids)
        findings.append(stage.TimelineFinding(
            "broll_near_duplicate", "warning", f"{names}: these shots look nearly the same",
            ids, ("broll",) * len(ids), "Keep one, or swap the others for different shots.",
            "these shots look nearly the same",
        ))
    for index in range(1, 4):
        name = f"broll-{index}"
        findings.append(stage.TimelineFinding(
            "external_media", "warning",
            f"{name} (lane broll): insert media is outside the project folder",
            (name,), ("broll",), "Move media into the project.",
            "insert media is outside the project folder",
        ))

    def resolved(root: Path) -> stage.ResolvedTimeline:
        return stage.replace(original(root), findings=tuple(findings))

    monkeypatch.setattr(stage, "resolve_timeline", resolved)
    assert cli.main(["timeline", "check", str(project)]) == 0
    warnings = [line for line in capsys.readouterr().out.splitlines()
                if line.startswith("warning:")]
    assert len(warnings) == 3
    assert "broll-a (lane broll), broll-b (lane broll)" in warnings[0]
    assert "broll-c (lane broll), broll-d (lane broll), broll-e (lane broll)" in warnings[1]
    assert "3 elements: broll-1, broll-2, broll-3" in warnings[2]


def test_invalid_fragment_and_lane_mismatch_write_nothing(
    project: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    item = insert(project, unexpected=True)
    assert cli.main(["timeline", "add", str(fragment(project, item)), str(project),
                     "--lane", "broll", "--json"]) == 2
    assert "error" in json.loads(capsys.readouterr().err)
    item = insert(project, lane="agent")
    assert cli.main(["timeline", "add", str(fragment(project, item)), str(project),
                     "--lane", "broll"]) == 2
    assert "expected 'broll'" in capsys.readouterr().err
    assert not (project / "timeline/broll.json").exists()


def test_stale_cut_refuses_check_with_machine_error(
    project: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    (project / "edit.json").write_text("{}", encoding="utf-8")
    assert cli.main(["timeline", "check", str(project), "--json"]) == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "stale_cut" and error["suggested_action"]


def test_broll_remove_cannot_touch_agent(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    add(project, graphic(), lane="agent").require_valid()
    original = (project / "timeline/agent.json").read_bytes()
    assert cli.main(["timeline", "remove", str(project), "--lane", "broll",
                     "--id", "agent-card"]) == 3
    assert "own prefix" in capsys.readouterr().err
    assert (project / "timeline/agent.json").read_bytes() == original


def test_current_directory_and_actual_module_entrypoint(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(project)
    assert cli.main(["timeline", "add", str(fragment(project, insert(project))),
                     "--lane", "broll", "--json"]) == 0
    capsys.readouterr()
    result = subprocess.run([sys.executable, "-m", "yap_studio", "timeline", "show", "--json"],
                            cwd=project, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["elements"][0]["start_frame"] == 66
