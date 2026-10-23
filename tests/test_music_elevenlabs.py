"""Offline consent, cache, and failure tests for generated music."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from pathlib import Path

import numpy as np
import pytest
from test_music_file import make_project, publish_cut, set_music, tone

from yap_studio import audio, cli, elevenlabs, tools
from yap_studio.affiliate import ELEVENLABS_AFFILIATE_URL
from yap_studio.errors import ProcessingError, YapError
from yap_studio.project.receipts import read_receipt
from yap_studio.project.status import status
from yap_studio.stages.music import MusicError


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = make_project(tmp_path)
    tone(root / "track.wav")
    return root


@pytest.fixture
def fake(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    _no_real_elevenlabs: Callable[[Path], None],
) -> Path:
    executable = Path(__file__).parent / "fakes/elevenlabs-pp-cli"
    monkeypatch.setenv("PATH", str(executable.parent) + os.pathsep + os.environ["PATH"])
    _no_real_elevenlabs(executable)
    log = tmp_path.parent / f"{tmp_path.name}-elevenlabs.log"
    monkeypatch.setenv("FAKE_ELEVENLABS_LOG", str(log))
    return log


def calls(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def generated(log: Path) -> list[dict]:
    return [call for call in calls(log) if call["argv"][:2] == ["music", "generate"]]


def test_missing_cli_auto_notice_once_and_fallback(
    project: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    set_music(project, mode="auto", file="track.wav")
    assert cli.main(["music", str(project), "--json"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["mode"] == "file"
    assert any(ELEVENLABS_AFFILIATE_URL in line for line in first["notices"])
    assert any("affiliate link" in line for line in first["notices"])
    assert any("npx -y @mvanhorn/printing-press-library install elevenlabs --cli-only"
               in line for line in first["notices"])
    assert any("auth set-token <your-key>" in line for line in first["notices"])
    assert cli.main(["music", str(project), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["notices"] == []
    assert (project / "music/bed.wav").exists()


def test_signed_out_and_explicit_mode_error(
    project: Path, fake: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("FAKE_ELEVENLABS_AUTH_EXIT", "4")
    set_music(project, mode="elevenlabs")
    assert cli.main(["music", str(project), "--json"]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "elevenlabs_not_set_up"
    assert any(ELEVENLABS_AFFILIATE_URL in line for line in error["notices"])
    assert any("auth set-token <your-key>" in line for line in error["notices"])


def test_signed_out_auto_falls_back(project: Path, fake: Path,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    from yap_studio.stages.music import music

    monkeypatch.setenv("FAKE_ELEVENLABS_AUTH_EXIT", "4")
    set_music(project, mode="auto", file=None)
    result = music(project)
    assert result["mode"] == "none"
    assert "isn't set up" in result["notices"][0]
    assert not generated(fake)


def test_auto_account_opt_in_and_no_generate(project: Path, fake: Path) -> None:
    set_music(project, mode="auto", file=None)
    from yap_studio.stages.music import music

    assert music(project)["mode"] == "none"
    assert not generated(fake)
    assert any("music: ElevenLabs can compose" in item
               for item in status(project)["needs_human"])


def test_free_tier_falls_back(project: Path, fake: Path,
                              monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_ELEVENLABS_TIER", "free")
    set_music(project, mode="auto", file=None)
    from yap_studio.stages.music import music

    result = music(project)
    assert result["mode"] == "none"
    assert "free plan" in result["notices"][0]
    assert not generated(fake)


def test_unknown_tier_proceeds_and_403_clears_pending(
    project: Path, fake: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("FAKE_ELEVENLABS_TIER", "missing_permissions")
    monkeypatch.setenv("FAKE_ELEVENLABS_GENERATE_EXIT", "4")
    monkeypatch.setenv("FAKE_ELEVENLABS_GENERATE_STDERR", "HTTP 403 subscription required")
    set_music(project, mode="elevenlabs")
    assert cli.main(["music", str(project), "--json"]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "elevenlabs_paid_plan_required"
    assert error["notices"][0] == (
        "Music: ElevenLabs refused the request because music needs a paid plan, "
        "so this video has no generated music.")
    assert not (project / ".yap/music/pending.json").exists()
    assert len(generated(fake)) == 1


def test_request_stdin_argv_timeout_and_receipt(
    project: Path, fake: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.stages.music import music

    original = tools.run_tool
    timeouts = []

    def run(name: str, args: list[str], **kwargs: object):
        if list(args[:2]) == ["music", "generate"]:
            timeouts.append(kwargs["timeout_s"])
            assert "--dry-run" not in args
            assert str(kwargs["stdout_path"]).endswith(".stdout")
        result = original(name, args, **kwargs)
        if list(args[:2]) == ["music", "generate"]:
            assert result.stdout is None
        return result

    monkeypatch.setattr(tools, "run_tool", run)
    set_music(project, mode="elevenlabs", description="Warm piano", tags=["gentle"])
    assert music(project)["mode"] == "elevenlabs"
    call = generated(fake)[0]
    args = call["argv"]
    assert args[:4] == ["music", "generate", "--stdin", "--timeout"]
    assert args[4] == "10m" and "--deliver" in args
    assert not set(args) & {"--json", "--agent", "--compact", "--quiet", "--yes"}
    assert json.loads(call["stdin"]) == read_receipt(project / "music")["elevenlabs"]["request"]
    assert timeouts[0] >= 2400


def test_plan_edit_stale_requires_new(project: Path, fake: Path,
                                      capsys: pytest.CaptureFixture[str]) -> None:
    from yap_studio.stages.music import music

    set_music(project, mode="elevenlabs")
    music(project)
    plan = project / "music/plan.json"
    data = json.loads(plan.read_text())
    data["composition_plan"]["sections"][0]["prompt"] = "different"
    plan.write_text(json.dumps(data))
    assert next(item for item in status(project)["stages"]
                if item["name"] == "music")["state"] == "stale"
    assert cli.main(["music", str(project), "--json"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "music_request_changed"
    assert any("--new" in item for item in status(project)["needs_human"])
    assert cli.main(["music", str(project), "--new", "--json"]) == 0
    assert len(generated(fake)) == 2


def test_recut_within_plan_headroom_reuses_cache_and_longer_cut_needs_new(
    project: Path, fake: Path,
) -> None:
    from yap_studio.stages.music import music

    set_music(project, mode="elevenlabs")
    music(project)
    plan = json.loads((project / "music/plan.json").read_text())
    assert plan["prompt_request"]["music_length_ms"] == 25_000
    publish_cut(project, 270)
    assert not any("--new" in item for item in status(project)["needs_human"])
    assert not music(project)["noop"]
    publish_cut(project, 600)
    assert not any("--new" in item for item in status(project)["needs_human"])
    assert not music(project)["noop"]
    assert len(generated(fake)) == 1
    publish_cut(project, 750)
    assert any("--new" in item for item in status(project)["needs_human"])
    with pytest.raises(MusicError) as caught:
        music(project)
    assert caught.value.code == "music_request_changed"
    assert len(generated(fake)) == 1


def test_recut_reuses_paid_cache_after_sign_out(
    project: Path, fake: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.stages.music import music

    set_music(project, mode="elevenlabs")
    music(project)
    monkeypatch.setenv("FAKE_ELEVENLABS_AUTH_EXIT", "4")
    publish_cut(project, 270)
    result = music(project)
    assert result["mode"] == "elevenlabs" and not result["noop"]
    assert len(generated(fake)) == 1


def test_simultaneous_approvals_reserve_only_one_paid_attempt(
    project: Path, fake: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_music(project, mode="elevenlabs")
    barrier = threading.Barrier(2, timeout=20)
    original = elevenlabs.compose_plan
    generate = elevenlabs.generate_music
    hold = threading.Event()
    sequence = count()

    def compose(request: dict, *, dry_run: bool) -> dict:
        barrier.wait()
        result = original(request, dry_run=dry_run)
        result["sections"][0]["prompt"] += f" {next(sequence)}"
        return result

    def delayed_generate(*args: object, **kwargs: object) -> object:
        hold.wait(0.5)
        return generate(*args, **kwargs)

    monkeypatch.setattr(elevenlabs, "compose_plan", compose)
    monkeypatch.setattr(elevenlabs, "generate_music", delayed_generate)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = [future.result() for future in
                    [pool.submit(lambda: _music_outcome(project)) for _ in range(2)]]
    assert sum(result == "verified" for result in outcomes) == 1
    assert outcomes.count("music_generation_unreconciled") == 1
    assert len(generated(fake)) == 1
    plan = json.loads((project / "music/plan.json").read_text())
    body = {"composition_plan": plan["composition_plan"],
            "model_id": elevenlabs.MUSIC_MODEL}
    assert read_receipt(project / "music")["elevenlabs"]["request_sha256"] == (
        elevenlabs.request_sha256(body))


def _music_outcome(project: Path) -> str:
    from yap_studio.stages.music import music

    try:
        return music(project)["status"]
    except YapError as exc:
        return exc.code


def test_status_with_stale_cut_and_generated_receipt_returns(
    project: Path, fake: Path,
) -> None:
    from yap_studio.stages.music import music

    set_music(project, mode="elevenlabs")
    music(project)
    edit_path = project / "edit.json"
    edit = json.loads(edit_path.read_text())
    edit["end_hold_s"] = 2.0
    edit_path.write_text(json.dumps(edit))
    report = status(project)
    assert next(item for item in report["stages"] if item["name"] == "cut")["state"] == "stale"
    assert isinstance(report["needs_human"], list)


def test_failure_is_unreconciled_until_confirmed(
    project: Path, fake: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    set_music(project, mode="elevenlabs")
    monkeypatch.setenv("FAKE_ELEVENLABS_GENERATE_EXIT", "5")
    assert cli.main(["music", str(project), "--json"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "music_generation_failed"
    assert not (project / "music/bed.wav").exists()
    assert json.loads((project / ".yap/music/pending.json").read_text())["state"] == "unreconciled"
    assert any("--confirm-retry" in item for item in status(project)["needs_human"])
    assert cli.main(["music", str(project), "--json"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "music_generation_unreconciled"
    assert len(generated(fake)) == 1
    monkeypatch.setenv("FAKE_ELEVENLABS_GENERATE_EXIT", "0")
    assert cli.main(["music", str(project), "--confirm-retry", "--json"]) == 0
    assert len(generated(fake)) == 2


def test_confirm_retry_preserves_pending_attempt_on_another_host(
    project: Path, fake: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    set_music(project, mode="elevenlabs")
    pending_path = project / ".yap/music/pending.json"
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema": "yap-studio/music-pending@1", "state": "pending",
              "host": "studio-other-host", "pid": 12345}
    pending_path.write_text(json.dumps(record))

    assert cli.main(["music", str(project), "--confirm-retry", "--json"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "music_generation_unreconciled"
    assert "started on another computer (studio-other-host)" in error["message"]
    assert "studio-other-host is stopped" in error["suggested_action"]
    assert "ElevenLabs history" in error["suggested_action"]
    assert "yap-studio music DIR --confirm-retry --other-computer-stopped" in (
        error["suggested_action"])
    assert json.loads(pending_path.read_text()) == record
    assert not generated(fake)


def test_other_computer_stopped_with_confirm_retry_generates_once(
    project: Path, fake: Path,
) -> None:
    set_music(project, mode="elevenlabs")
    pending_path = project / ".yap/music/pending.json"
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.write_text(json.dumps({"schema": "yap-studio/music-pending@1",
                                        "state": "pending", "host": "studio-other-host",
                                        "pid": 12345}))

    assert cli.main(["music", str(project), "--confirm-retry",
                     "--other-computer-stopped", "--json"]) == 0
    assert len(generated(fake)) == 1
    assert (project / "music/bed.wav").is_file()
    assert not pending_path.exists()


def test_other_computer_stopped_requires_confirm_retry(
    project: Path, fake: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    set_music(project, mode="elevenlabs")
    pending_path = project / ".yap/music/pending.json"
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema": "yap-studio/music-pending@1", "state": "pending",
              "host": "studio-other-host", "pid": 12345}
    pending_path.write_text(json.dumps(record))

    assert cli.main(["music", str(project), "--other-computer-stopped", "--json"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "music_retry_confirmation_required"
    assert "--other-computer-stopped requires --confirm-retry" in error["message"]
    assert "--confirm-retry" in error["suggested_action"]
    assert json.loads(pending_path.read_text()) == record
    assert not generated(fake)


def test_failed_generation_redacts_key_from_pending_and_cli_output(
    project: Path, fake: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    set_music(project, mode="elevenlabs")
    monkeypatch.setenv("FAKE_ELEVENLABS_GENERATE_EXIT", "5")
    monkeypatch.setenv("FAKE_ELEVENLABS_GENERATE_STDERR", "HTTP 503 upstream unavailable")
    assert cli.main(["music", str(project), "--json"]) == 2
    printed = capsys.readouterr()
    pending_text = (project / ".yap/music/pending.json").read_text()
    pending = json.loads(pending_text)
    assert pending["state"] == "unreconciled"
    assert "HTTP 503 upstream unavailable" in pending["error"]
    assert json.loads(printed.err)["error"]["code"] == "music_generation_failed"
    assert len(generated(fake)) == 1
    for exposed in (pending_text, printed.out, printed.err):
        assert "api-key" not in exposed.lower()
        assert "abcd" not in exposed.lower()


def test_cache_reuse_new_and_silent_hold(project: Path, fake: Path) -> None:
    from yap_studio.stages.music import music

    set_music(project, mode="elevenlabs")
    music(project)
    assert music(project)["noop"]
    assert len(generated(fake)) == 1
    assert not music(project, new=True)["noop"]
    assert len(generated(fake)) == 2
    _, samples = audio.decode(project / "music/bed.wav", mono=False)
    assert np.max(np.abs(samples[480_000:])) == 0
    assert np.max(np.abs(samples[432_000:436_800])) > np.max(
        np.abs(samples[475_200:480_000]))


def test_binary_generation_stdout_stays_out_of_receipt_result_and_cli_output(
    project: Path, fake: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    set_music(project, mode="elevenlabs")
    assert cli.main(["music", str(project), "--json"]) == 0
    printed = capsys.readouterr()
    result = json.loads(printed.out)
    receipt_path = project / "music/receipt.json"
    receipt_text = receipt_path.read_text()
    assert result["mode"] == "elevenlabs"
    assert len(generated(fake)) == 1
    assert receipt_path.stat().st_size < 8192
    assert "RIFF" not in receipt_text
    assert "RIFF" not in json.dumps(result)
    assert "RIFF" not in printed.out + printed.err


def test_dry_run_writes_nothing(project: Path, fake: Path) -> None:
    from yap_studio.stages.music import music

    set_music(project, mode="elevenlabs")
    before = {path.relative_to(project) for path in project.rglob("*")}
    result = music(project, dry_run=True)
    assert result["status"] == "dry_run"
    assert {path.relative_to(project) for path in project.rglob("*")} == before
    assert calls(fake)[-1]["argv"][:2] == ["music", "compose-plan"]


@pytest.mark.parametrize("with_plan", [False, True])
@pytest.mark.parametrize("as_json", [False, True])
def test_cli_dry_run_shows_request_body_without_key(
    project: Path, fake: Path, capsys: pytest.CaptureFixture[str],
    with_plan: bool, as_json: bool,
) -> None:
    set_music(project, mode="elevenlabs", description="Warm piano")
    if with_plan:
        plan_path = project / "music/plan.json"
        plan_path.parent.mkdir(exist_ok=True)
        plan_path.write_text(json.dumps({"composition_plan": {
            "sections": [{"duration_ms": 25_000, "prompt": "Warm piano"}],
        }}))
    args = ["music", str(project), "--dry-run"]
    if as_json:
        args.append("--json")
    assert cli.main(args) == 0
    printed = capsys.readouterr()
    output = printed.out + printed.err
    result = json.loads(printed.out)
    previews = result["requests"] if as_json else result
    assert len(previews) == 1
    assert "POST https://api.elevenlabs.io/v1/music" in previews[0]
    assert "Body:" in previews[0]
    assert ('"composition_plan"' if with_plan else '"prompt": "Warm piano"') in previews[0]
    assert "api-key" not in output.lower()
    assert "abcd" not in output.lower()
    if as_json:
        assert result["status"] == "dry_run"
        assert "api-key" not in json.dumps(result).lower()
        assert "abcd" not in json.dumps(result).lower()
    assert calls(fake)[-1]["argv"][-1] == "--dry-run"


def test_redact_cli_text_removes_case_insensitive_key_line() -> None:
    stderr = "POST https://api.elevenlabs.io/v1/music\n  XI-API-KEY: ****abcd\nBody: {}\n"
    assert elevenlabs.redact_cli_text(stderr) == (
        "POST https://api.elevenlabs.io/v1/music\nBody: {}"
    )


def test_failed_compose_plan_redacts_processing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise ProcessingError("elevenlabs-pp-cli music compose-plan", 5,
                              "HTTP 503 upstream unavailable\n  XI-API-KEY: ****abcd")

    monkeypatch.setattr(tools, "run_tool", fail)
    with pytest.raises(ProcessingError) as caught:
        elevenlabs.compose_plan({"prompt": "Warm piano"}, dry_run=False)
    assert "HTTP 503 upstream unavailable" in caught.value.message
    assert "api-key" not in caught.value.message.lower()
    assert "abcd" not in caught.value.message.lower()
    assert "abcd" not in caught.value.full_stderr.lower()


@pytest.mark.parametrize(("tier", "auth_exit"), [("creator", "4"), ("free", "0")])
def test_dry_run_without_paid_account_previews_and_shows_signup(
    project: Path, fake: Path, monkeypatch: pytest.MonkeyPatch,
    tier: str, auth_exit: str,
) -> None:
    from yap_studio.stages.music import music

    monkeypatch.setenv("FAKE_ELEVENLABS_TIER", tier)
    monkeypatch.setenv("FAKE_ELEVENLABS_AUTH_EXIT", auth_exit)
    set_music(project, mode="elevenlabs")
    result = music(project, dry_run=True)
    assert result["status"] == "dry_run"
    assert any("affiliate link" in line for line in result["notices"])
    assert calls(fake)[-1]["argv"][-1] == "--dry-run"
    assert not (project / "music/plan.json").exists()


def test_dry_run_missing_cli_refuses(project: Path) -> None:
    from yap_studio.stages.music import music

    set_music(project, mode="elevenlabs")
    with pytest.raises(MusicError) as caught:
        music(project, dry_run=True)
    assert caught.value.code == "elevenlabs_not_set_up"


def test_default_elevenlabs_fixture_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert tools.find_tool("elevenlabs-pp-cli") is None
    assert "ELEVENLABS_API_KEY" not in os.environ
    assert Path(os.environ["ELEVENLABS_CONFIG"]).is_relative_to(tmp_path)
    assert os.environ["ELEVENLABS_BASE_URL"] == "http://127.0.0.1:9"


def test_dry_run_with_plan_previews_generate_without_writing(
    project: Path, fake: Path,
) -> None:
    from yap_studio.stages.music import music

    set_music(project, mode="elevenlabs")
    music(project)
    before = {path.relative_to(project): path.read_bytes() for path in project.rglob("*")
              if path.is_file()}
    result = music(project, dry_run=True)
    assert result["status"] == "dry_run"
    assert generated(fake)[-1]["argv"][-1] == "--dry-run"
    after = {path.relative_to(project): path.read_bytes() for path in project.rglob("*")
             if path.is_file()}
    assert after == before


def test_yes_saves_mode(project: Path, fake: Path,
                        capsys: pytest.CaptureFixture[str]) -> None:
    set_music(project, mode="auto", file=None)
    assert cli.main(["music", str(project), "--yes", "--json"]) == 0
    capsys.readouterr()
    assert json.loads((project / "project.json").read_text())["music"]["mode"] == "elevenlabs"


def test_cli_dry_run_yes_previews_description_without_saving(
    project: Path, fake: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    set_music(project, mode="auto", file=None)
    before = (project / "project.json").read_bytes()
    assert cli.main(["music", str(project), "--yes", "--description", "Soft keys",
                     "--tags", "warm,slow", "--dry-run", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "dry_run"
    assert (project / "project.json").read_bytes() == before
    assert json.loads(calls(fake)[-1]["stdin"])["prompt"] == "Soft keys, warm, slow"


@pytest.mark.parametrize("mode", ["file", "none"])
def test_cli_dry_run_yes_previews_from_other_mode_without_saving(
    project: Path, fake: Path, capsys: pytest.CaptureFixture[str], mode: str,
) -> None:
    set_music(project, mode=mode, **({"file": "track.wav"} if mode == "file" else {}))
    before = (project / "project.json").read_bytes()
    assert cli.main(["music", str(project), "--yes", "--dry-run", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "dry_run" and len(result["requests"]) == 1
    assert calls(fake)[-1]["argv"][:2] == ["music", "compose-plan"]
    assert (project / "project.json").read_bytes() == before


def test_changed_description_previews_new_compose_request(
    project: Path, fake: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from yap_studio.stages.music import music

    set_music(project, mode="elevenlabs", description="Old piano")
    music(project)
    before = (project / "music/plan.json").read_bytes()
    assert cli.main(["music", str(project), "--dry-run", "--description", "New strings",
                     "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result["requests"]) == 1
    assert calls(fake)[-1]["argv"][:2] == ["music", "compose-plan"]
    assert json.loads(calls(fake)[-1]["stdin"])["prompt"] == "New strings"
    assert (project / "music/plan.json").read_bytes() == before


def test_affiliate_url_is_one_constant_and_labelled_in_docs() -> None:
    root = Path(__file__).resolve().parents[1]
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).split(b"\0")
    code, unlabelled = [], []
    for name in files:
        if not name:
            continue
        path = os.fsdecode(name)
        content = (root / path).read_bytes()
        if b"\0" in content or ELEVENLABS_AFFILIATE_URL.encode() not in content:
            continue
        if not path.endswith(".md"):
            code.append(path)
            continue
        lines = content.decode("utf-8").splitlines()
        for index, line in enumerate(lines):
            if ELEVENLABS_AFFILIATE_URL in line:
                nearby = " ".join(lines[max(0, index - 1):index + 2]).lower()
                if "affiliate link" not in nearby:
                    unlabelled.append(f"{path}:{index + 1}")
    assert code == ["src/yap_studio/affiliate.py"]
    assert unlabelled == []
