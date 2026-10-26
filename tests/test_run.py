"""The run chain: stage order, the first failed check stops it, a rerun resumes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_assemble import project, refuse_unexpected_matte_work  # noqa: F401
from test_cli_captions import fake_layers  # noqa: F401
from test_cut import write_json
from test_stage_captions import set_elements
from test_stage_graphics import graphic

from yap_studio import cli
from yap_studio.errors import ProcessingError, YapError
from yap_studio.feedback.versions import load_versions, version_layers
from yap_studio.project.receipts import read_receipt
from yap_studio.stages import run as chain
from yap_studio.stages.music import MusicError


def _fake_stages(monkeypatch: pytest.MonkeyPatch, calls: list[str],
                 failing: dict[str, Any] | None = None) -> None:
    """Replace every stage with a recorder; finished stages report no-ops on a rerun."""
    failing = failing if failing is not None else {}
    finished: set[str] = set()

    def stage(name: str, result: dict[str, Any]):
        def call(root: Path, **kwargs: Any) -> dict[str, Any]:
            calls.append(name)
            outcome = failing.get(name)
            if isinstance(outcome, Exception):
                raise outcome
            if outcome is not None:
                return outcome
            noop = name in finished
            finished.add(name)
            return {**result, "noop": noop}
        return call

    def ingest_state(context: Any, spec: Any, previous: dict) -> dict[str, str]:
        if spec.name != "ingest":
            return {"state": "missing", "reason": "no cut yet"}
        calls.append("ingest")
        return failing.get("ingest") or {"state": "done", "reason": "inputs match"}

    monkeypatch.setattr(chain, "ProjectContext", lambda root: None)
    monkeypatch.setattr(chain, "_stage_state", ingest_state)
    monkeypatch.setattr(chain.fillers_stage, "fillers", stage(
        "fillers", {"counts": {"status": {"applied": 2}}}))
    monkeypatch.setattr(chain.cut_stage, "cut", stage(
        "cut", {"status": "verified", "out": "cut/cut.mp4"}))
    monkeypatch.setattr(chain.music_stage, "music", stage("music", {"bed": None}))
    monkeypatch.setattr(chain.captions_stage, "captions", stage("captions", {"phrases": 3}))
    monkeypatch.setattr(chain.graphics_stage, "graphics", stage("graphics", {"elements": []}))
    monkeypatch.setattr(chain.assemble_stage, "assemble", stage(
        "assemble", {"out": "final/demo.mp4"}))

    def verify(root: Path, state: dict[str, Any]) -> tuple[bool, str]:
        calls.append("verify")
        return True, "final verified"

    def version(root: Path, state: dict[str, Any]) -> tuple[bool, str]:
        calls.append("version")
        state["version"] = "v1"
        return False, "v1"

    monkeypatch.setitem(chain._HANDLERS, "verify", verify)
    monkeypatch.setitem(chain._HANDLERS, "version", version)

    def words(root: Path, state: dict[str, Any]) -> tuple[bool, str]:
        calls.append("words")
        state["words"] = {"status": "pass", "dropped": []}
        return True, "all words kept"

    monkeypatch.setitem(chain._HANDLERS, "words", words)


def _shell(tmp_path: Path, *, edit: bool = True) -> Path:
    root = tmp_path / "reel"
    root.mkdir()
    (root / "project.json").write_text("{}")
    if edit:
        (root / "edit.json").write_text("{}")
    return root


def test_run_walks_every_stage_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _fake_stages(monkeypatch, calls)
    result = chain.run(_shell(tmp_path))
    assert calls == list(chain.STEPS)
    assert result["status"] == "finished" and result["stopped"] is None
    assert result["final"] == "final/demo.mp4" and result["version"] == "v1"
    assert [step["state"] for step in result["steps"]] == ["unchanged"] + ["done"] * 6 + [
        "unchanged", "unchanged", "done"]
    assert result["next"].startswith("yap-studio review --versions ")


@pytest.mark.parametrize("allow_violations", [False, True])
def test_caption_warnings_and_allow_violations_reach_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    allow_violations: bool,
) -> None:
    calls: list[str] = []
    _fake_stages(monkeypatch, calls)
    options = []
    warning = "Captions have no clear position at 0:00.1; frame the shot wider."

    def captions(root: Path, **kwargs: Any) -> dict:
        calls.append("captions")
        options.append(kwargs)
        return {"noop": False, "phrases": 2, "warnings": [warning]}

    monkeypatch.setattr(chain.captions_stage, "captions", captions)
    assert cli.main(["run", str(_shell(tmp_path)),
                     *(["--allow-violations"] if allow_violations else [])]) == 0
    assert options == [{"allow_violations": allow_violations}]
    assert "assemble" in calls
    assert warning in capsys.readouterr().out


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("error", [ValueError("backing count mismatch"),
                                 OSError("caption layer write failed")])
def test_run_caption_failure_matches_captions_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    json_output: bool, error: Exception,
) -> None:
    calls: list[str] = []
    _fake_stages(monkeypatch, calls, {"captions": error})
    root = _shell(tmp_path)
    options = ["--json"] if json_output else []
    assert cli.main(["captions", str(root), *options]) == 2
    caption_output = capsys.readouterr().err
    assert cli.main(["run", str(root), *options]) == 2
    output = capsys.readouterr()
    assert calls == ["captions", "ingest", "fillers", "cut", "music", "captions"]
    if json_output:
        stopped = json.loads(output.out)["stopped"]
        assert stopped["step"] == "captions" and stopped["exit"] == 2
        assert {key: value for key, value in stopped.items() if key not in {"step", "exit"}} == (
            json.loads(caption_output)["error"]
        )
        assert stopped["code"] == "caption_failed"
        assert stopped["type"] == "processing_error"
        assert stopped["message"] == str(error)
    else:
        assert f"stopped at captions: {error}" in output.err
        assert "Report this caption failure" in caption_output and (
            "Report this caption failure" in output.err
        )
        assert "Cannot read input" not in output.err


def test_run_preserves_missing_caption_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from yap_studio.errors import InputError

    error = InputError("cut/cut-words.json")
    _fake_stages(monkeypatch, [], {"captions": error})
    root = _shell(tmp_path)
    assert cli.main(["run", str(root), "--json"]) == 2
    stopped = json.loads(capsys.readouterr().out)["stopped"]
    assert stopped == {"step": "captions", **error.to_dict(), "exit": 2}


def test_run_prints_music_setup_notice_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _fake_stages(monkeypatch, [])
    root = _shell(tmp_path)
    count = 0

    def music_with_notice(_root: Path) -> dict[str, Any]:
        nonlocal count
        count += 1
        return {"noop": count > 1, "bed": None,
                "notices": ["Set up ElevenLabs music"] if count == 1 else []}

    monkeypatch.setattr(chain.music_stage, "music", music_with_notice)
    first = chain.run(root)
    assert first["notices"] == ["Set up ElevenLabs music"]
    count = 0
    assert cli.main(["run", str(root)]) == 0
    assert capsys.readouterr().err.count("Set up ElevenLabs music") == 1
    assert cli.main(["run", str(root)]) == 0
    assert "Set up ElevenLabs music" not in capsys.readouterr().err


def test_run_prints_assembly_sound_setup_notice_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _fake_stages(monkeypatch, [])
    root = _shell(tmp_path)
    count = 0

    def assemble_with_notice(_root: Path, **_kwargs: Any) -> dict[str, Any]:
        nonlocal count
        count += 1
        return {"noop": count > 1, "out": "final/demo.mp4",
                "notices": ["Set up ElevenLabs sound effects"] if count == 1 else []}

    monkeypatch.setattr(chain.assemble_stage, "assemble", assemble_with_notice)
    first = chain.run(root)
    assert first["notices"] == ["Set up ElevenLabs sound effects"]
    count = 0
    assert cli.main(["run", str(root)]) == 0
    assert capsys.readouterr().err.count("Set up ElevenLabs sound effects") == 1
    assert cli.main(["run", str(root)]) == 0
    assert "Set up ElevenLabs sound effects" not in capsys.readouterr().err


def test_run_missing_elevenlabs_cli_shows_setup_in_text_and_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from test_music_file import make_project, set_music

    root = make_project(tmp_path)
    set_music(root, mode="elevenlabs")
    real_music = chain.music_stage.music
    _fake_stages(monkeypatch, [])
    monkeypatch.setattr(chain.music_stage, "music", real_music)

    assert cli.main(["run", str(root)]) == 2
    printed = capsys.readouterr()
    assert "install elevenlabs --cli-only" in printed.err
    assert "auth set-token <your-key>" in printed.err
    assert "stopped at music" in printed.err

    assert cli.main(["run", str(root), "--json"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["stopped"]["step"] == "music"
    assert any("install elevenlabs --cli-only" in line for line in result["notices"])
    assert any("auth set-token <your-key>" in line for line in result["notices"])


def test_run_carries_notices_from_other_stage_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = MusicError("Assembly failed.", code="assembly_failed",
                       suggested_action="Fix the final.", notices=["Assembly detail."])
    _fake_stages(monkeypatch, [], {"assemble": error})
    result = chain.run(_shell(tmp_path))
    assert result["stopped"]["step"] == "assemble"
    assert result["notices"] == ["Assembly detail."]


@pytest.mark.parametrize("frames", [None, (0, 20)])
def test_run_json_assemble_processing_failure_has_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    frames: tuple[int, int] | None,
) -> None:
    real_assemble = chain.assemble_stage.assemble
    _fake_stages(monkeypatch, [])
    monkeypatch.setattr(chain.assemble_stage, "assemble", real_assemble)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise ProcessingError("ffmpeg", 1, "mix failed")

    monkeypatch.setattr(chain.assemble_stage, "_assemble_impl", fail)
    root = _shell(tmp_path)
    options = ["--frames", f"{frames[0]}-{frames[1]}"] if frames else []
    assert cli.main(["run", str(root), "--json", *options]) == 2
    stopped = json.loads(capsys.readouterr().out)["stopped"]
    assert stopped["step"] == "assemble" and stopped["code"] == "ffmpeg_exit_1"
    assert "yap-studio doctor" in stopped["suggested_action"]
    rerun = f"yap-studio run {root}" + (f" --frames {frames[0]}-{frames[1]}" if frames else "")
    assert f"{rerun} again" in stopped["suggested_action"]


def test_run_reuses_removals_applied_before_story_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_fillers import install_engine, make_project

    from yap_studio import verify

    # The planted filler uses the scripted engine; cut verification needs no real ASR.
    monkeypatch.setattr(verify, "resolve_backend", lambda *_args, **_kwargs: None)
    root = make_project(tmp_path / "project")
    batches = install_engine(monkeypatch)
    applied = chain.fillers_stage.fillers(root, apply=True)
    assert [(item["id"], item["kind"], item["status"]) for item in applied["entries"]] == [
        ("r1", "filler", "applied"), ("r2", "silence", "applied"),
    ]
    assert [[path.stem for path in batch] for batch in batches] == [
        ["round-0-0-cut", "round-0-0-control", "round-0-1-cut", "round-0-1-control"],
        ["round-1-0-cut", "round-1-0-control", "round-1-1-cut", "round-1-1-control",
         "round-1-2-cut", "round-1-2-control", "round-1-3-cut", "round-1-3-control"],
    ]
    initial_batches = [tuple(batch) for batch in batches]
    reviewed = chain.cut_stage.cut(root)
    assert reviewed["status"] == "verified"
    before = {name: (root / name).read_bytes() for name in (
        "removals.json", "cut/receipt.json", "cut/cut.mp4",
    )}
    real_fillers, real_cut = chain.fillers_stage.fillers, chain.cut_stage.cut
    _fake_stages(monkeypatch, [])
    monkeypatch.setattr(chain.fillers_stage, "fillers", real_fillers)
    monkeypatch.setattr(chain.cut_stage, "cut", real_cut)

    result = chain.run(root)

    assert result["status"] == "finished", result["stopped"]
    states = {step["name"]: step["state"] for step in result["steps"]}
    assert states["fillers"] == states["cut"] == "unchanged"
    assert {name: (root / name).read_bytes() for name in before} == before
    assert [tuple(batch) for batch in batches] == initial_batches


def test_run_keeps_proposals_out_of_reviewed_cut(tmp_path, monkeypatch) -> None:
    from test_fillers import install_engine, make_project

    from yap_studio import verify
    from yap_studio.project.status import ProjectContext, _stage_state

    monkeypatch.setattr(verify, "resolve_backend", lambda *_args, **_kwargs: None)
    root = make_project(tmp_path / "project")
    batches = install_engine(monkeypatch)
    chain.fillers_stage.fillers(root)
    assert chain.cut_stage.cut(root)["status"] == "verified"
    before = {name: (root / name).read_bytes() for name in (
        "removals.json", "cut/cut-words.json", "cut/cut.mp4",
    )}
    initial_batches = [tuple(batch) for batch in batches]
    real_fillers, real_cut = chain.fillers_stage.fillers, chain.cut_stage.cut
    _fake_stages(monkeypatch, [])
    monkeypatch.setattr(chain, "ProjectContext", ProjectContext)
    monkeypatch.setattr(chain, "_stage_state", _stage_state)
    monkeypatch.setattr(chain.fillers_stage, "fillers", real_fillers)
    monkeypatch.setattr(chain.cut_stage, "cut", real_cut)

    result = chain.run(root)

    assert result["status"] == "finished", result["stopped"]
    assert not any(item["status"] == "applied" for item in json.loads(
        (root / "removals.json").read_bytes())["removals"])
    assert {name: (root / name).read_bytes() for name in before} == before
    assert {item["name"]: item["state"] for item in result["steps"]}["fillers"] == "unchanged"
    assert {item["name"]: item["state"] for item in result["steps"]}["cut"] == "unchanged"
    assert [tuple(batch) for batch in batches] == initial_batches


def test_run_current_cut_never_calls_fillers_with_no_removals(tmp_path, monkeypatch):
    _fake_stages(monkeypatch, [])
    monkeypatch.setattr(chain, "_stage_state", lambda *_: {"state": "done"})
    monkeypatch.setattr(chain.fillers_stage, "fillers", lambda *_, **__: pytest.fail("cut is done"))
    result = chain.run(_shell(tmp_path))
    assert result["steps"][1] == {"name": "fillers", "state": "unchanged",
                                  "detail": "kept the cut's removals: 0 applied"}


@pytest.mark.parametrize("no_engine", [False, True])
def test_run_stale_cut_only_proposes(tmp_path, monkeypatch, no_engine) -> None:
    from test_fillers import ROWS, install_engine, make_project

    from yap_studio import verify
    from yap_studio.project.status import STAGES, ProjectContext, _stage_state

    monkeypatch.setattr(verify, "resolve_backend", lambda *_args, **_kwargs: None)
    rows = [*ROWS, ("later", 3.5, 3.8)] if no_engine else None
    root = make_project(tmp_path / "project with spaces", rows)
    if no_engine:
        monkeypatch.setattr(chain.fillers_stage, "resolve_backend", lambda *_: None)
    else:
        install_engine(monkeypatch)
    assert chain.cut_stage.cut(root)["status"] == "verified"
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"][0]["to"]["time"] -= 0.1
    write_json(root / "edit.json", edit)
    spec = next(spec for spec in STAGES if spec.name == "cut")
    assert _stage_state(ProjectContext(root), spec, {"ingest": "done"})["state"] == "stale"
    real_fillers, real_cut = chain.fillers_stage.fillers, chain.cut_stage.cut
    _fake_stages(monkeypatch, [])
    monkeypatch.setattr(chain, "ProjectContext", ProjectContext)
    monkeypatch.setattr(chain, "_stage_state", _stage_state)
    monkeypatch.setattr(chain.fillers_stage, "fillers", real_fillers)
    monkeypatch.setattr(chain.cut_stage, "cut", real_cut)

    result = chain.run(root)

    assert result["status"] == "finished", result["stopped"]
    entries = json.loads((root / "removals.json").read_bytes())["removals"]
    assert not any(item["status"] == "applied" for item in entries)
    proposed = [item for item in entries if item["status"] == "proposed"]
    assert proposed and (not no_engine or all(item["kind"] == "silence" for item in proposed))
    assert any(f"{len(proposed)} removals proposed and not applied" in note
               and f"'{root}'" in note for note in result["warnings"])
    if no_engine:
        assert any("fillers skipped: no speech engine" in note for note in result["warnings"])


@pytest.mark.parametrize("dropped", [False, True])
def test_run_checks_words_before_saving_version(tmp_path, monkeypatch, capsys, dropped) -> None:
    from test_kept_words import _removal, saved_project

    root = saved_project(tmp_path)
    if dropped:
        _removal(root)
        words = json.loads((root / "cut/cut-words.json").read_bytes())
        words["words"].pop(0)
        write_json(root / "cut/cut-words.json", words)
    real_words = chain._HANDLERS.get("words")
    calls = []
    _fake_stages(monkeypatch, calls)
    if real_words is not None:
        monkeypatch.setitem(chain._HANDLERS, "words", real_words)
    monkeypatch.setattr(chain.assemble_stage, "assemble", lambda *_args, **_kwargs: {
        "noop": False, "out": "final/result.mp4",
    })
    result = chain.run(root)
    if dropped:
        assert result["status"] == "stopped"
        assert result["stopped"]["step"] == "words"
        assert result["stopped"]["code"] == "words_dropped"
        assert result["stopped"]["exit"] == 1
        assert result["words"]["dropped"][0]["text"] == "First"
        assert '"First" at 0:00.1 (r13)' in result["stopped"]["message"]
        action = result["stopped"]["suggested_action"]
        assert f"yap-studio fillers set {root} r13 rejected" in action
        assert "version" not in calls and result["version"] is None
        assert len(load_versions(root)) == 1
        assert cli.main(["run", str(root), "--json"]) == 1
        assert json.loads(capsys.readouterr().out)["stopped"]["code"] == "words_dropped"
    else:
        assert result["status"] == "finished"
        assert result["words"]["status"] == "pass"
        assert result["words"]["dropped"] == []


def test_run_pilot_skips_verify_words_and_version(project: Path) -> None:  # noqa: F811
    result = chain.run(project, frames=(30, 89))
    assert result["status"] == "finished", result["stopped"]
    assert result["final"] is None and result["version"] is None
    assert result["pilot"] and (project / result["pilot"]).is_file()
    assert [(item["name"], item["state"], item["detail"]) for item in result["steps"][-3:]] == [
        ("verify", "skipped", "skipped: pilot"),
        ("words", "skipped", "skipped: pilot"),
        ("version", "skipped", "skipped: pilot"),
    ]
    assert not (project / "versions").exists()
    assert result["next"] == f"yap-studio run {project}"


def test_run_pilot_mid_phrase_exits_three(project: Path) -> None:  # noqa: F811
    from test_assemble import _measured_caption_layout

    layout = _measured_caption_layout(project)
    end = layout["captions"][0]["end_frame"] - 2
    result = chain.run(project, frames=(0, end))
    assert result["stopped"]["exit"] == 3
    assert result["stopped"]["suggested_end"] == end + 1
    assert not (project / "pilot/receipt.json").exists()


def test_run_pilot_cli_refusal_and_success(
    project: Path, capsys: pytest.CaptureFixture[str],  # noqa: F811
) -> None:
    from test_assemble import _measured_caption_layout

    layout = _measured_caption_layout(project)
    end = layout["captions"][0]["end_frame"] - 2
    assert cli.main(["run", str(project), "--frames", f"0-{end}"]) == 3
    assert f"--frames 0-{end + 1}" in capsys.readouterr().err
    assert cli.main(["run", str(project), "--frames", "0-999"]) == 3
    capsys.readouterr()
    assert cli.main(["run", str(project), "--frames", "30-89"]) == 0
    output = capsys.readouterr().out
    assert "verify: skipped: pilot" in output
    assert "words: skipped: pilot" in output
    assert "version: skipped: pilot" in output
    assert "Not a final and not saved as a version." in output


def test_run_reports_untranscribed_ingest_speech(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _fake_stages(monkeypatch, calls)
    root = _shell(tmp_path)
    (root / "ingest").mkdir()
    write_json(root / "ingest/ingest.json", {
        "takes": [{"id": "t1", "speech_check": {"untranscribed": [
            {"start": 195.4, "end": 222.0, "seconds": 26.6},
        ]}}],
    })
    result = chain.run(root)
    assert result["warnings"] == [
        "t1: speech with no words at 3:15-3:42 (26.6 s); "
        "listen before you use that part"
    ]


def test_first_failure_stops_and_rerun_resumes(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    failing: dict[str, Any] = {"graphics": YapError(
        "Card c1 covers the face.", code="timeline_refused",
        suggested_action="Move card c1 to a later word.")}
    _fake_stages(monkeypatch, calls, failing)
    root = _shell(tmp_path)
    result = chain.run(root)
    assert calls == ["ingest", "fillers", "cut", "music", "captions", "graphics"]
    assert result["status"] == "stopped" and result["final"] is None
    assert result["stopped"]["step"] == "graphics"
    assert result["stopped"]["message"] == "Card c1 covers the face."
    assert result["stopped"]["suggested_action"] == "Move card c1 to a later word."
    assert result["stopped"]["exit"] == 3
    assert result["next"] == f"yap-studio run {root}"

    failing.clear()
    calls.clear()
    resumed = chain.run(root)
    assert resumed["status"] == "finished"
    states = {step["name"]: step["state"] for step in resumed["steps"]}
    assert all(states[name] == "unchanged" for name in ("ingest", "fillers", "cut", "music",
                                                        "captions"))
    assert states["graphics"] == "done" and states["assemble"] == "done"


def test_failed_cut_stops_with_its_verification_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    root = _shell(tmp_path)
    (root / "cut").mkdir()
    write_json(root / "cut/verify.json", {"findings": [
        {"severity": "warn", "message": "quiet", "suggested_action": "ignore"},
        {"severity": "fail", "message": "Word t1:4 is clipped at 00:02.1.",
         "suggested_action": "End segment s2 one word later."},
    ]})
    _fake_stages(monkeypatch, calls, {"cut": {
        "status": "failed", "noop": False,
        "asserts": [{"id": "frames", "pass": True},
                    {"id": "verify", "pass": False, "expected": "pass", "actual": "fail"}],
    }})
    result = chain.run(root)
    assert calls == ["ingest", "fillers", "cut"]
    stopped = result["stopped"]
    assert stopped["code"] == "cut_failed" and stopped["exit"] == 1
    assert stopped["message"] == ("Cut failed its checks: verify (expected pass, actual fail). "
                                  "Word t1:4 is clipped at 00:02.1.")
    assert stopped["suggested_action"] == "End segment s2 one word later."


@pytest.mark.parametrize(("step", "code", "expected"), [
    ("music", "music_contract_failed", 1),
    ("cut", "cut_frame_mismatch", 1),
    ("cut", "cut_sample_mismatch", 1),
    ("captions", "caption_output_locked", 3),
    ("music", "music_too_short", 2),
])
def test_stage_errors_exit_by_meaning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                      step: str, code: str, expected: int) -> None:
    calls: list[str] = []
    _fake_stages(monkeypatch, calls, {step: YapError("stage said no", code=code,
                                                     suggested_action="do the fix")})
    result = chain.run(_shell(tmp_path))
    assert result["stopped"]["step"] == step and result["stopped"]["exit"] == expected


def test_refused_version_save_exits_three(tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    from yap_studio.feedback.versions import VersionError

    calls: list[str] = []
    _fake_stages(monkeypatch, calls)

    def refuse(root: Path, state: dict[str, Any]) -> tuple[bool, str]:
        raise VersionError("Render unchanged since v1.")

    monkeypatch.setitem(chain._HANDLERS, "version", refuse)
    result = chain.run(_shell(tmp_path))
    assert result["stopped"]["code"] == "version_refused" and result["stopped"]["exit"] == 3


def test_filler_notes_and_cut_warnings_reach_the_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    _fake_stages(monkeypatch, calls, {
        "fillers": {"noop": False, "counts": {"status": {}}, "skipped": 3,
                    "notes": ["fillers skipped: no speech engine", "pause p1 kept"]},
        "cut": {"status": "verified", "noop": False, "out": "cut/cut.mp4",
                "warnings": ["Segment s2 ends mid-word."]},
    })
    root = _shell(tmp_path)
    result = chain.run(root)
    assert result["warnings"] == ["fillers skipped: no speech engine (3)", "pause p1 kept",
                                  "Segment s2 ends mid-word."]
    assert cli.main(["run", str(root)]) == 0
    out = capsys.readouterr().out
    assert "warning: fillers skipped: no speech engine (3)" in out
    assert "warning: Segment s2 ends mid-word." in out


def test_unchanged_cut_still_reports_its_verify_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    root = _shell(tmp_path)
    (root / "cut").mkdir()
    write_json(root / "cut/verify.json", {"findings": [
        {"severity": "warn", "message": "Join at 00:01.2 may click."},
        {"severity": "info", "message": "not a warning"},
    ]})
    _fake_stages(monkeypatch, calls, {"cut": {
        "status": "verified", "noop": True, "out": "cut/cut.mp4",
        "warnings": ["Segment s2 ends mid-word."],
    }})
    assert chain.run(root)["warnings"] == ["Segment s2 ends mid-word.",
                                           "Join at 00:01.2 may click."]


def test_caption_ids_follow_their_words_not_their_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    root = tmp_path / "reel"
    (root / "captions").mkdir(parents=True)
    monkeypatch.setattr("yap_studio.stages.timeline.resolve_timeline",
                        lambda path: SimpleNamespace(elements=[]))

    def ids(*phrases: tuple[list[str], str]) -> dict[str, str]:
        write_json(root / "captions/phrases.json", {"phrases": [
            {"ids": words, "text": text, "start": index, "end": index + 0.5}
            for index, (words, text) in enumerate(phrases)]})
        return {layer["text"]: layer["id"] for layer in chain.layer_windows(root)["layers"]}

    before = ids((["t1:4", "t1:5"], "See you"))
    after = ids((["t1:0", "t1:1"], "Hello there"), (["t1:4", "t1:5"], "See you"))
    assert before["See you"] == after["See you"] == "caption:t1:4-t1:5"
    repeated = chain.layer_windows(root)
    write_json(root / "captions/phrases.json", {"phrases": [
        {"ids": ["t1:4"], "text": "Again", "start": 0, "end": 0.5},
        {"ids": ["t1:4"], "text": "Again", "start": 1, "end": 1.5}]})
    layers = chain.layer_windows(root)["layers"]
    assert [layer["id"] for layer in layers] == ["caption:t1:4", "caption:t1:4#2"]
    assert repeated["schema"] == "yap-studio/layer-windows@1"


def test_caption_backing_keys_keep_layer_window_suffixes(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    (tmp_path / "captions").mkdir()
    write_json(tmp_path / "captions/phrases.json", {"phrases": [
        {"ids": ["t1:4"], "text": "Empty", "start": 0, "end": 0},
        {"ids": ["t1:4"], "text": "Again", "start": 1, "end": 1.5},
        {"text": "Legacy", "start": 2, "end": 2.5},
        {"ids": ["t1:4"], "text": "Again", "start": 3, "end": 3.5},
    ]})
    monkeypatch.setattr("yap_studio.stages.timeline.resolve_timeline",
                        lambda root: SimpleNamespace(elements=[]))
    assert [layer["id"] for layer in chain.layer_windows(tmp_path)["layers"]] == [
        "caption:t1:4", "caption:2", "caption:t1:4#2",
    ]


def test_missing_edit_stops_after_ingest(tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _fake_stages(monkeypatch, calls)
    result = chain.run(_shell(tmp_path, edit=False))
    assert calls == ["ingest"]
    assert result["stopped"]["code"] == "edit_missing"
    assert "yap-studio words" in result["stopped"]["suggested_action"]


def test_unfinished_ingest_stops_before_any_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _fake_stages(monkeypatch, calls,
                 {"ingest": {"state": "stale", "reason": "changed input: take:t1"}})
    result = chain.run(_shell(tmp_path))
    assert calls == ["ingest"]
    assert result["stopped"]["code"] == "ingest_needed" and result["stopped"]["exit"] == 2
    assert result["stopped"]["message"] == "Ingest is stale: changed input: take:t1."
    assert "yap-studio ingest" in result["stopped"]["suggested_action"]


def test_cli_prints_steps_reason_fix_and_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    _fake_stages(monkeypatch, calls, {"cut": {
        "status": "failed", "noop": False, "asserts": [{"id": "verify", "pass": False}],
    }})
    root = _shell(tmp_path)
    assert cli.main(["run", str(root)]) == 1
    captured = capsys.readouterr()
    assert captured.out.splitlines()[:3] == ["ingest: unchanged, source words verified",
                                             "fillers: 2 applied, 0 proposed, 0 rejected",
                                             "cut: failed"]
    assert "stopped at cut: Cut failed its checks: verify" in captured.err
    assert "fix: Read cut/verify.json, correct edit.json, then run again." in captured.err
    assert f"next: yap-studio run {root}" in captured.out

    assert cli.main(["run", str(root), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "yap-studio/run@1" and payload["stopped"]["step"] == "cut"


def test_cli_refuses_folder_without_project(tmp_path: Path,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", str(tmp_path)]) == 2
    assert "run yap-studio init first" in capsys.readouterr().err


@pytest.fixture
def reel(project: Path) -> Path:  # noqa: F811
    """The assembly fixture, with the full piece plan a real cut records."""
    path = project / "cut/receipt.json"
    data = json.loads(path.read_bytes())
    data["plan"].update(fps="30/1", pieces=[{
        "segment_id": "s1", "take": "t1", "out_f0": 0, "n_frames": 120,
        "src_t0": "0", "src_t1": "4",
    }])
    write_json(path, data)
    return project


def test_real_stages_finish_save_once_and_resume_as_noops(reel: Path) -> None:
    result = chain.run(reel, note="First full video")
    assert result["status"] == "finished", result["stopped"]
    assert read_receipt(reel / "final")["status"] == "verified"
    assert (reel / result["final"]).is_file()
    versions = load_versions(reel)
    assert [item.id for item in versions] == ["v1"] and result["version"] == "v1"
    assert versions[0].note == "First full video"
    assert versions[0].media["source"] == result["final"]

    again = chain.run(reel)
    assert again["status"] == "finished"
    assert {step["state"] for step in again["steps"]} == {"unchanged"}
    assert [item.id for item in load_versions(reel)] == ["v1"]


def test_platform_failure_stops_before_saving(reel: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    data = json.loads((reel / "project.json").read_bytes())
    data["platform"] = "reels"
    write_json(reel / "project.json", data)
    presets = chain.load_presets()
    presets["reels"]["duration"]["max_s"] = 1
    monkeypatch.setattr(chain, "load_presets", lambda: presets)
    result = chain.run(reel)
    assert result["stopped"]["step"] == "verify"
    assert result["stopped"]["code"] == "platform_failed" and result["stopped"]["exit"] == 1
    assert result["stopped"]["suggested_action"]
    assert load_versions(reel) == ()


def test_saved_final_carries_caption_and_card_windows(reel: Path) -> None:
    set_elements(reel, graphic())
    result = chain.run(reel)
    assert result["status"] == "finished", result["stopped"]
    layers = version_layers(load_versions(reel)[0])
    assert layers is not None
    by_kind = {layer.kind: layer for layer in layers}
    card = by_kind["graphic"]
    assert (card.id, card.start, card.end) == ("agent-tag", pytest.approx(0.2),
                                               pytest.approx(1.2))
    caption = by_kind["caption"]
    assert caption.id == "caption:t1:0-t1:1" and caption.text == "First words"
    assert 0 <= caption.start < caption.end
