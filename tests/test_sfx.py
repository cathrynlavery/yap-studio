"""Offline sound kit consent, cache, placement and mix contracts."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import synth

from yap_studio import cards, cli, elevenlabs, sfx
from yap_studio.project.model import ProjectFileError, parse_document
from yap_studio.project.status import ProjectContext, human_items
from yap_studio.sfx.place import candidates, space
from yap_studio.stages import assemble as assembly
from yap_studio.stages.assemble_audio import decode, render_audio
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline
from yap_studio.style import ResolvedStyle, resolve, resolve_style_data


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
         _no_real_elevenlabs: Callable[[Path], None]) -> Path:
    executable = Path(__file__).parent / "fakes/elevenlabs-pp-cli"
    monkeypatch.setenv("PATH", str(executable.parent) + os.pathsep + os.environ["PATH"])
    _no_real_elevenlabs(executable)
    log = tmp_path / "elevenlabs.log"
    monkeypatch.setenv("FAKE_ELEVENLABS_LOG", str(log))
    return log


def _calls(log: Path, command: str = "sound-generation") -> list[list[str]]:
    if not log.exists():
        return []
    return [item["argv"] for line in log.read_text().splitlines()
            if (item := json.loads(line))["argv"][:1] == [command]]


def test_default_sfx_cache_fixture_points_under_test_tmp(tmp_path: Path) -> None:
    assert sfx.sfx_cache_root() == tmp_path / "sfx-cache"
    assert not sfx.sfx_cache_root().exists()


def test_recipe_validation_and_no_tracked_audio() -> None:
    kit = sfx.load_kit()
    assert kit["schema"] == sfx.SCHEMA
    assert {sound["name"] for sound in kit["sounds"]} == {
        "whoosh", "click", "tick-roll", "chime"}
    with pytest.raises(sfx.SfxError, match="duration_s"):
        sfx._check_sound({**kit["sounds"][0], "duration_s": 31}, "sound")
    tracked = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    assert not any(Path(name).suffix.lower() in {".mp3", ".wav", ".ogg", ".flac"}
                   and name.startswith("src/yap_studio/sfx/") for name in tracked)
    assert not any(Path(name).suffix.lower() in {".mp3", ".wav", ".ogg", ".flac"}
                   and not name.startswith("tests/") for name in tracked)


def test_kit_approval_argv_cache_reuse_and_one_stale_sound(
    fake: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["sfx", "kit", "--json"]) == 2
    assert "spends ElevenLabs credits" in capsys.readouterr().err
    assert cli.main(["sfx", "list", "--json"]) == 0
    assert all(item["state"] == "missing" for item in
               json.loads(capsys.readouterr().out)["sounds"])
    assert not fake.exists()
    assert cli.main(["sfx", "kit", "--yes", "--json"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert len(first["generated"]) == 4
    calls = _calls(fake)
    assert len(calls) == 4
    for args, sound in zip(calls, sfx.load_kit()["sounds"], strict=True):
        assert args[:-1] == [
            "sound-generation", "--model-id", "eleven_text_to_sound_v2",
            "--text", sound["prompt"], "--duration-seconds", str(sound["duration_s"]),
            "--prompt-influence", str(sound["prompt_influence"]),
            "--output-format", "mp3_44100_128", "--timeout", "2m", "--deliver",
        ]
        assert args[-1].startswith("file:")
        assert not {"--json", "--agent", "--compact", "--quiet", "--yes"} & set(args)
    assert all(item["state"] == "fresh" and Path(item["cache_path"]).is_file()
               for item in first["sounds"])
    assert cli.main(["sfx", "kit", "--yes", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["generated"] == []
    assert len(_calls(fake)) == 4
    overlay = sfx.sfx_cache_root() / "kits/soft.json"
    changed = {"schema": sfx.SCHEMA, "name": "soft", "description": "Changed",
               "sounds": [{**sfx.load_kit()["sounds"][0], "prompt": "A different soft whoosh"}]}
    overlay.parent.mkdir(parents=True)
    overlay.write_text(json.dumps(changed))
    states = {item["name"]: item["state"] for item in sfx.list_kit()["sounds"]}
    assert states == {"whoosh": "stale", "click": "fresh",
                      "tick-roll": "fresh", "chime": "fresh"}
    assert cli.main(["sfx", "kit", "--yes", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["generated"] == ["whoosh"]
    assert len(_calls(fake)) == 5
    changed["sounds"][0]["gain_db"] = -2
    overlay.write_text(json.dumps(changed))
    assert next(item for item in sfx.list_kit()["sounds"]
                if item["name"] == "whoosh")["state"] == "stale"


def test_failure_blocks_retry_and_403_clears_pending(
    fake: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_ELEVENLABS_GENERATE_EXIT", "5")
    with pytest.raises(sfx.SfxError, match="may have been billed"):
        sfx.generate(yes=True)
    pending = sfx.sfx_cache_root() / "pending.json"
    assert json.loads(pending.read_text())["state"] == "unreconciled"
    count = len(_calls(fake))
    with pytest.raises(sfx.SfxError, match="may have been billed"):
        sfx.generate(yes=True)
    assert len(_calls(fake)) == count
    monkeypatch.setenv("FAKE_ELEVENLABS_GENERATE_EXIT", "4")
    monkeypatch.setenv("FAKE_ELEVENLABS_GENERATE_STDERR", "HTTP 403 subscription required")
    with pytest.raises(sfx.SfxError) as exc:
        sfx.generate(yes=True, confirm_retry=True)
    assert exc.value.code == "sfx_paid_plan_required"
    assert not pending.exists()


def test_failed_generation_redacts_pending_error(
    fake: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_ELEVENLABS_GENERATE_EXIT", "5")
    with pytest.raises(sfx.SfxError, match="may have been billed"):
        sfx.generate(yes=True, only="whoosh")
    pending = (sfx.sfx_cache_root() / "pending.json").read_text()
    assert "api-key" not in pending.lower()
    assert "abcd" not in pending
    assert "synthetic failure" in pending


def test_interleaved_kit_calls_reserve_one_paid_attempt(
    fake: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = threading.Barrier(2, timeout=20)
    original_account = elevenlabs.account_state
    original_generate = elevenlabs.generate_sound
    started = threading.Event()
    release = threading.Event()

    def account() -> elevenlabs.AccountState:
        barrier.wait()
        return original_account()

    def delayed_generate(*args: object, **kwargs: object) -> object:
        pending = json.loads((sfx.sfx_cache_root() / "pending.json").read_text())
        assert pending["pid"] == os.getpid()
        assert pending["host"] == socket.gethostname()
        started.set()
        assert release.wait(20)
        return original_generate(*args, **kwargs)

    def attempt() -> str:
        try:
            return sfx.generate(yes=True, only="whoosh", confirm_retry=True)["status"]
        except sfx.SfxError as exc:
            return exc.code

    monkeypatch.setattr(elevenlabs, "account_state", account)
    monkeypatch.setattr(elevenlabs, "generate_sound", delayed_generate)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(attempt) for _ in range(2)]
        try:
            assert started.wait(10)
            done, _ = wait(futures, timeout=10, return_when=FIRST_COMPLETED)
            assert len(done) == 1
        finally:
            release.set()
        outcomes = [future.result() for future in futures]
    assert sorted(outcomes) == ["sfx_unreconciled", "verified"]
    assert len(_calls(fake)) == 1
    assert not (sfx.sfx_cache_root() / "pending.json").exists()


def test_missing_tier_permission_proceeds(
    fake: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_ELEVENLABS_TIER", "missing_permissions")
    assert sfx.generate(yes=True)["status"] == "verified"
    assert len(_calls(fake)) == 4


def test_deleted_cached_audio_is_missing(fake: Path) -> None:
    listing = sfx.generate(yes=True)
    click = next(item for item in listing["sounds"] if item["name"] == "click")
    Path(click["cache_path"]).unlink()
    assert next(item for item in sfx.list_kit()["sounds"]
                if item["name"] == "click")["state"] == "missing"


def test_identical_sound_requests_reuse_validated_audio(fake: Path) -> None:
    first = sfx.add("tap-a", "One shared soft tap", 0.6, "item", yes=True)
    first_sound = next(item for item in first["sounds"] if item["name"] == "tap-a")
    audio_hash = first_sound["audio_sha256"]
    second = sfx.add("tap-b", "One shared soft tap", 0.6, "item", yes=True)
    sounds = {item["name"]: item for item in second["sounds"]}
    assert len(_calls(fake)) == 1
    assert sounds["tap-a"]["state"] == sounds["tap-b"]["state"] == "fresh"
    assert sounds["tap-a"]["audio_sha256"] == sounds["tap-b"]["audio_sha256"] == audio_hash
    assert sounds["tap-a"]["cache_path"] == sounds["tap-b"]["cache_path"]


def test_interleaved_add_preserves_both_overlay_recipes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay_path = sfx.sfx_cache_root() / "kits/soft.json"
    first_paused = threading.Event()
    release_first = threading.Event()
    second_wrote = threading.Event()
    original_write = sfx._write_json

    def interleaved_write(path: Path, value: dict) -> None:
        if path == overlay_path:
            names = {item["name"] for item in value["sounds"]}
            if names == {"tap-a"}:
                first_paused.set()
                assert release_first.wait(10)
            if "tap-b" in names:
                second_wrote.set()
        original_write(path, value)

    monkeypatch.setattr(sfx, "_write_json", interleaved_write)
    monkeypatch.setattr(sfx, "generate", lambda *args, **kwargs: {"status": "verified"})
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(sfx.add, "tap-a", "First tap", 0.6, "item", yes=True)
        assert first_paused.wait(10)
        second = pool.submit(sfx.add, "tap-b", "Second tap", 0.6, "item", yes=True)
        try:
            second_wrote.wait(1)
        finally:
            release_first.set()
        first.result(timeout=10)
        second.result(timeout=10)
    assert {item["name"] for item in sfx._read_kit(overlay_path)["sounds"]} == {
        "tap-a", "tap-b"}


def test_fresh_kit_reuses_cache_after_sign_out(
    fake: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    sfx.generate(yes=True)
    before = len(_calls(fake, "auth"))
    monkeypatch.setenv("FAKE_ELEVENLABS_AUTH_EXIT", "4")
    assert cli.main(["sfx", "kit", "--yes", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "fresh" and result["generated"] == []
    assert len(_calls(fake)) == 4
    assert len(_calls(fake, "auth")) == before


def test_dry_run_add_and_free_tier(
    fake: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = sfx.sfx_cache_root()
    result = sfx.generate(dry_run=True)
    assert result["status"] == "dry_run" and len(result["requests"]) == 4
    assert not root.exists()
    assert all("--dry-run" in args for args in _calls(fake))
    preview = sfx.add("pop", "A soft UI pop", 0.6, "item", dry_run=True)
    assert preview["status"] == "dry_run" and not root.exists()
    with pytest.raises(sfx.SfxError, match="pass --yes"):
        sfx.add("pop", "A soft UI pop", 0.6, "item")
    assert not root.exists()
    added = sfx.add("pop", "A soft UI pop", 0.6, "item", yes=True)
    assert added["generated"] == ["pop"]
    assert any(item["name"] == "pop" for item in sfx.load_kit()["sounds"])
    monkeypatch.setenv("FAKE_ELEVENLABS_TIER", "free")
    with pytest.raises(sfx.SfxError) as exc:
        sfx.add("tap", "A soft tap", 0.5, "item", yes=True)
    assert "non-commercial" in " ".join(exc.value.notices)


def test_kit_dry_run_shows_redacted_request_preview(
    fake: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["sfx", "kit", "--dry-run", "--json"]) == 0
    output = capsys.readouterr().out
    result = json.loads(output)
    assert result["status"] == "dry_run"
    assert "POST https://api.elevenlabs.io/v1/sound-generation" in output
    assert "A short soft airy" in output
    assert "api-key" not in output.lower()
    assert "abcd" not in output


@pytest.mark.parametrize(("state", "env_name", "env_value", "notice"), [
    ("signed_out", "FAKE_ELEVENLABS_AUTH_EXIT", "4", "isn't set up"),
    ("free", "FAKE_ELEVENLABS_TIER", "free", "non-commercial"),
])
def test_kit_dry_run_without_paid_account_includes_signup(
    fake: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    state: str, env_name: str, env_value: str, notice: str,
) -> None:
    monkeypatch.setenv(env_name, env_value)
    assert cli.main(["sfx", "kit", "--dry-run", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "dry_run"
    assert notice in " ".join(result["notices"])
    assert len(_calls(fake)) == 4
    added = sfx.add("pop", "A soft UI pop", 0.6, "item", dry_run=True)
    assert added["status"] == "dry_run"
    assert notice in " ".join(added["notices"])
    assert cli.main(["sfx", "kit", "--dry-run"]) == 0
    assert notice in capsys.readouterr().err


def test_kit_dry_run_requires_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(elevenlabs, "account_state", lambda: "missing_cli")
    with pytest.raises(sfx.SfxError) as exc:
        sfx.generate(dry_run=True)
    assert exc.value.code == "sfx_account_required"


def _item(kind: str, slots: dict, *, start: int = 30, end: int = 120,
          override: object = None) -> ResolvedElement:
    payload = {"template": kind, "slots": slots}
    if override is not None:
        payload["sfx"] = override
    return ResolvedElement("agent-card", "graphic", "agent", start, end, "front", payload)


@pytest.mark.parametrize(("kind", "slots", "rule", "times"), [
    ("checklist", {"items": ["A", "B"], "cues": []}, "item", [1.3, 1.65]),
    ("checklist", {"items": ["A", "B"], "cues": [0.9, 1.4]}, "item", [1.9, 2.4]),
    ("checklist", {"items": ["A", "B"], "cues": [0.9]}, "item", [1.9, 2.25]),
    ("specs_list", {"rows": [{}, {}]}, "item", [1.2, 1.33]),
    ("steps", {"steps": ["A", "B"], "cues": []}, "item", [1.6, 3.1]),
    ("steps", {"steps": ["A", "B"], "cues": [0.9, 1.4]}, "item", [1.9, 2.4]),
    ("steps", {"steps": ["A", "B"], "cues": [1.0]}, "item", [2.0, 3.1]),
    ("steps", {"steps": ["A", "B", "C"], "cues": [1.5]},
     "item", [2.5, 2.5, 3.1]),
    ("flow", {"nodes": ["A", "B"], "cues": []}, "item", [1.99, 2.6]),
    ("flow", {"nodes": ["A", "B"], "cues": [0.8, 1.3]}, "item", [2.25, 2.75]),
    ("flow", {"nodes": ["A", "B"], "cues": [0.8]}, "item", [2.25, 2.6]),
    ("chart", {"bars": [{}, {}]}, "item", [1.3, 1.42]),
    ("chart", {"bars": [{}, {}]}, "count_start", [1.3]),
    ("progress_stat", {}, "count_start", [1.25]),
    ("success_check", {}, "settle", [1.8]),
    ("edit_timeline", {}, "item", []),
    ("storyboard", {}, "item", []),
    ("jiggle_delete", {"items": ["A", "B", "C"], "remove": [2, 1]}, "item", [1.9, 2.4]),
    ("file_drop", {}, "settle", [1.53]),
    ("file_drop", {"drop_at": 0.4}, "settle", [1.78]),
    ("jiggle_delete", {"items": ["A", "B", "C"], "remove": [2, 1], "cues": [1.1]},
     "item", [2.1, 2.6]),
    ("video_card", {}, "item", []),
    ("playing_grid", {}, "item", []),
    ("reel_feed", {}, "item", []),
    ("gravity_drop", {"drops": [{"text": "x", "at": 0.3}, {"text": "y", "at": 0.8}]},
     "settle", [1.3, 1.8]),
    ("gravity_drop", {"drops": [{"text": "x", "at": {"frame": 39, "time_s": 1.3}}]},
     "settle", [1.3]),
    ("choices", {"options": ["One", "Two"]}, "item", []),
    ("choices", {"options": ["One", "Two"]}, "settle", []),
])
def test_card_event_offsets(kind: str, slots: dict, rule: str, times: list[float]) -> None:
    timeline = ResolvedTimeline((_item(kind, slots),), (), (), 150, 45)
    sound = {"name": "tone", "rule": rule, "priority": 1, "duration_s": 0.5}
    actual = [cue.time_s for cue in candidates(timeline, [sound])
              if cue.element_id == "agent-card"]
    assert actual == pytest.approx(times)


@pytest.mark.parametrize("look", ["on_cue", "on_cue_glow"])
@pytest.mark.parametrize("frames", [90, 127, 210])
@pytest.mark.parametrize("cues", [None, [0.7, 1.4], [], [1.1], [2.099]])
def test_specs_list_clicks_match_planned_row_cues(
    look: str, frames: int, cues: list[float] | None,
) -> None:
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["specs_rows"] = look
    slots = {"title": "Plan", "rows": [{"name": "First", "value": "12"},
                                       {"name": "Second", "value": "25"}]}
    if cues is not None:
        slots["cues"] = cues
    plan = cards.card_layer_plan("specs_list", slots, style, frames=frames)
    item = _item("specs_list", slots, end=30 + frames)
    timeline = ResolvedTimeline((item,), (), (), 300, 45)
    sound = {"name": "click", "rule": "item", "priority": 1, "duration_s": 0.2}
    actual = candidates(timeline, [sound], specs_rows=style.data["graphics"]["specs_rows"])
    assert [cue.event for cue in actual] == ["specs_list:1", "specs_list:2"]
    assert [cue.time_s for cue in actual] == pytest.approx([
        item.start_s + cue for cue in json.loads(plan.variables["cues"])])


@pytest.mark.parametrize("frames", [90, 127, 210])
@pytest.mark.parametrize("cues", [None, [0.7, 1.4]])
def test_specs_list_together_clicks_keep_legacy_timing(
    frames: int, cues: list[float] | None,
) -> None:
    slots = {"rows": [{}, {}]}
    if cues is not None:
        slots["cues"] = cues
    item = _item("specs_list", slots, end=30 + frames)
    timeline = ResolvedTimeline((item,), (), (), 300, 45)
    sound = {"name": "click", "rule": "item", "priority": 1, "duration_s": 0.2}
    assert [cue.time_s for cue in candidates(timeline, [sound])] == [1.2, 1.33]
    assert [cue.time_s for cue in candidates(timeline, [sound], specs_rows="together")] == [
        1.2, 1.33]


@pytest.mark.parametrize("look", ["together", "on_cue", "on_cue_glow"])
@pytest.mark.parametrize("reveal", ["count", "wheel", "slot"])
def test_stat_breakdown_rows_keep_existing_sounds(reveal: str, look: str) -> None:
    item = _item("stat", {"end": 37, "reveal": reveal})
    with_rows = replace(item, payload={**item.payload, "slots": {
        **item.payload["slots"], "rows": ["12 first", "25 second"], "cues": [0.7, 1.4]}})
    sounds = [{"name": rule, "rule": rule, "priority": 1, "duration_s": 0.2}
              for rule in ("count_start", "settle", "item")]
    baseline = candidates(ResolvedTimeline((item,), (), (), 150, 45), sounds)
    actual = candidates(ResolvedTimeline((with_rows,), (), (), 150, 45), sounds, specs_rows=look)
    assert actual == baseline
    assert [cue.event for cue in actual if cue.element_id == item.id] == [
        "stat:start", "stat:final"]


@pytest.mark.parametrize("slots", [
    {"request": "go now"},
    {"request": "go now", "cadence": "uniform", "cues": [0.9]},
    {"request": "i" * 60},
    {"request": "go now", "word_cues": [0.3, 0.49]},
])
@pytest.mark.parametrize("frames", [60, 90])
def test_prompt_bar_settle_matches_planned_confirmation(slots: dict, frames: int) -> None:
    plan = cards.card_layer_plan("prompt_bar", slots, resolve("preset:clean_minimal"),
                                 frames=frames)
    confirm = json.loads(plan.variables["schedule"])["confirm_at"]
    item = _item("prompt_bar", slots, end=30 + frames)
    timeline = ResolvedTimeline((item,), (), (), 150, 45)
    sound = {"name": "tone", "rule": "settle", "priority": 1, "duration_s": 0.5}
    actual = [cue for cue in candidates(timeline, [sound]) if cue.element_id == item.id]
    assert len(actual) == 1
    assert actual[0].event == "prompt_bar"
    assert actual[0].time_s == item.start_s + confirm


@pytest.mark.parametrize("slots", [
    {"keyword": "go now"},
    {"keyword": "go now", "cadence": "uniform", "cues": [0.9]},
    {"keyword": "i" * 32},
    {"keyword": "go now", "word_cues": [0.3, 0.49]},
])
@pytest.mark.parametrize("frames", [60, 90])
def test_comment_cta_settle_matches_planned_send(slots: dict, frames: int) -> None:
    plan = cards.card_layer_plan("comment_cta", slots, resolve("preset:clean_minimal"),
                                 frames=frames)
    send = json.loads(plan.variables["schedule"])["confirm_at"]
    item = _item("comment_cta", slots, end=30 + frames)
    timeline = ResolvedTimeline((item,), (), (), 150, 45)
    sound = {"name": "tone", "rule": "settle", "priority": 1, "duration_s": 0.5}
    actual = [cue for cue in candidates(timeline, [sound]) if cue.element_id == item.id]
    assert len(actual) == 1 and actual[0].event == "comment_cta"
    assert actual[0].time_s == item.start_s + send


@pytest.mark.parametrize("drop_at", [None, 0, 0.4])
def test_comment_drop_settle_matches_landing(drop_at: float | None) -> None:
    slots = {"text": "Send it", "initials": "MC"}
    if drop_at is not None:
        slots["drop_at"] = drop_at
    item = _item("comment_drop", slots)
    timeline = ResolvedTimeline((item,), (), (), 150, 45)
    sound = {"name": "tone", "rule": "settle", "priority": 1, "duration_s": 0.5}
    actual = [cue for cue in candidates(timeline, [sound]) if cue.element_id == item.id]
    assert len(actual) == 1 and actual[0].event == "comment_drop"
    assert actual[0].time_s == item.start_s + (0.1 if drop_at is None else drop_at) + 0.6


@pytest.mark.parametrize("strike_at", [None, 0.6, 1.2])
def test_icon_strike_settle_matches_completed_strike(strike_at: float | None) -> None:
    slots = {} if strike_at is None else {"strike_at": strike_at}
    item = _item("icon_strike", slots)
    timeline = ResolvedTimeline((item,), (), (), 150, 45)
    sound = {"name": "tone", "rule": "settle", "priority": 1, "duration_s": 0.5}
    actual = [cue for cue in candidates(timeline, [sound]) if cue.element_id == item.id]
    assert len(actual) == 1 and actual[0].event == "icon_strike"
    assert actual[0].time_s == item.start_s + (0.8 if strike_at is None else strike_at) + 0.28


def test_jiggle_delete_event_names_follow_removal_order() -> None:
    element = _item("jiggle_delete", {"items": ["A", "B", "C"], "remove": [3, 1]})
    timeline = ResolvedTimeline((element,), (), (), 150, 45)
    sound = {"name": "tone", "rule": "item", "priority": 1, "duration_s": 0.2}
    cues = candidates(timeline, [sound])
    assert [cue.event for cue in cues] == ["jiggle_delete:3", "jiggle_delete:1"]


@pytest.mark.parametrize("at", [0, 0.1, 0.5])
@pytest.mark.parametrize("anchored", [False, True])
def test_gravity_drop_sound_lands_at_authored_time(at: float, anchored: bool) -> None:
    anchor = {"frame": 30 + round(at * 30), "time_s": 1 + at} if anchored else at
    timeline = ResolvedTimeline((_item("gravity_drop", {
        "drops": [{"text": "Weight", "at": anchor}],
    }),), (), (), 150, 45)
    sound = {"name": "tone", "rule": "settle", "priority": 1, "duration_s": 0.5}
    cues = candidates(timeline, [sound])
    assert [cue.time_s for cue in cues if cue.element_id == "agent-card"] == pytest.approx([1 + at])


def test_transition_shift_mute_spacing_and_end() -> None:
    transition = ResolvedElement("agent-join", "transition", "agent", 54, 66,
                                 "front", {"join_frame": 60, "sfx": {"shift_s": 0.1}})
    muted = _item("checklist", {"items": ["A"], "cues": []}, override=False)
    timeline = ResolvedTimeline((transition, muted), (), (), 150, 45)
    sounds = [{"name": "whoosh", "rule": "transition", "priority": 3,
               "duration_s": 0.8},
              {"name": "chime", "rule": "settle", "priority": 4,
               "duration_s": 0.8}]
    cue_list = candidates(timeline, sounds)
    assert [cue.time_s for cue in cue_list] == pytest.approx([1.9, 4.15])
    assert not any(cue.element_id == muted.id for cue in cue_list)
    kept, dropped = space(cue_list, 2.5)
    assert [cue.sound for cue in kept] == ["chime"]
    assert dropped[0]["cue"]["sound"] == "whoosh"
    assert dropped[0]["kept_cue"]["sound"] == "chime"


@pytest.mark.parametrize(("kind", "slots", "start_time", "settle_time"), [
    ("stat", {"end": 100}, 1.0, 2.72),
    ("progress_stat", {}, 1.25, 2.1),
    ("chart", {"bars": [{}, {}]}, 1.3, 1.92),
])
def test_count_up_keeps_tick_and_chime(
    kind: str, slots: dict, start_time: float, settle_time: float,
) -> None:
    timeline = ResolvedTimeline((_item(kind, slots),), (), (), 150, 45)
    sounds = [
        {"name": "tick-roll", "rule": "count_start", "priority": 2, "duration_s": 1},
        {"name": "chime", "rule": "settle", "priority": 4, "duration_s": 0.8},
    ]
    kept, _ = space(candidates(timeline, sounds), 2.5)
    pair = [cue for cue in kept if cue.element_id == "agent-card"]
    assert [cue.sound for cue in pair] == ["tick-roll", "chime"]
    assert [cue.time_s for cue in pair] == pytest.approx([start_time, settle_time])


def test_count_pair_still_drops_another_cards_click() -> None:
    count = _item("progress_stat", {})
    other = replace(_item("checklist", {"items": ["A"], "cues": [0.6]}), id="other-card")
    timeline = ResolvedTimeline((count, other), (), (), 150, 45)
    sounds = [
        {"name": "tick-roll", "rule": "count_start", "priority": 2, "duration_s": 1},
        {"name": "chime", "rule": "settle", "priority": 4, "duration_s": 0.8},
        {"name": "click", "rule": "item", "priority": 1, "duration_s": 0.5},
    ]
    kept, dropped = space(candidates(timeline, sounds), 2.5)
    assert {(cue.sound, cue.element_id) for cue in kept} >= {
        ("tick-roll", count.id), ("chime", count.id)}
    assert any(item["cue"]["sound"] == "click" and
               item["cue"]["element_id"] == other.id and
               item["reason"] == "minimum_gap" for item in dropped)


def test_timeline_sfx_contract_and_style_switch() -> None:
    fragment = {"schema": "yap-studio/timeline@1", "elements": [
        {"id": "agent-a", "kind": "graphic", "lane": "agent",
         "start": {"time": 1}, "hold_s": 3, "template": "checklist", "slots": {},
         "sfx": {"shift_s": -0.5}}]}
    parse_document("timeline-fragment", fragment)
    fragment["elements"][0]["sfx"] = False
    parse_document("timeline-fragment", fragment)
    fragment["elements"][0]["sfx"] = {"shift_s": 2}
    with pytest.raises(ProjectFileError):
        parse_document("timeline-fragment", fragment)
    assert resolve_style_data({"schema_version": 1, "music": {"sound_effects": "none"}})[
        "music"]["sound_effects"] == "none"


def test_status_opt_in_file_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                 capsys: pytest.CaptureFixture[str]) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "project.json").write_text(json.dumps({
        "schema": "yap-studio/project@1", "name": "story",
        "style": "preset:clean_minimal", "takes": [{"id": "t1", "path": "take.mov"}],
    }))
    (project / "take.mov").write_bytes(b"synthetic")
    sfx.remember_account("paid")
    monkeypatch.setattr("yap_studio.elevenlabs.account_state",
                        lambda: pytest.fail("status invoked ElevenLabs"))
    assert any("sfx kit --yes" in item
               for item in human_items(ProjectContext(project)))
    assert cli.main(["status", str(project), "--json"]) == 0
    assert any("sfx kit --yes" in item
               for item in json.loads(capsys.readouterr().out)["needs_human"])


@pytest.mark.parametrize(("state", "extra"), [
    ("missing_cli", "isn't set up"),
    ("signed_out", "isn't set up"),
    ("free", "non-commercial"),
])
def test_assembly_account_notice_once_without_cached_kit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str, extra: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "project.json").write_text(json.dumps({
        "schema": "yap-studio/project@1", "name": "story",
        "style": "preset:clean_minimal", "takes": [{"id": "t1", "path": "take.mov"}],
    }))
    (project / "take.mov").write_bytes(b"synthetic")
    checks = []
    monkeypatch.setattr("yap_studio.elevenlabs.account_state",
                        lambda: checks.append(state) or state)
    monkeypatch.setattr(assembly, "_pilot_phrases", lambda context: (b"", "hash"))
    monkeypatch.setattr(assembly, "_pilot_window", lambda snapshot, frames, phrases: None)
    monkeypatch.setattr(assembly, "assembly_snapshot", lambda context, **kwargs:
                        SimpleNamespace(sfx={"kit": "soft", "ready": False,
                                             "sounds": []}, inputs={}))
    monkeypatch.setattr(assembly, "_pilot_assemble", lambda context, snapshot, *args, **kwargs:
                        {"notices": snapshot.sfx.get("notices", [])})
    first = assembly.assemble(project, frames=(0, 0))
    assert extra in " ".join(first["notices"])
    assert any("affiliate link" in line for line in first["notices"])
    second = assembly.assemble(project, frames=(0, 0))
    assert second["notices"] == []
    assert checks == [state, state]
    assert any("sfx kit --yes" in item for item in human_items(ProjectContext(project)))


def test_mix_sfx_stereo_below_voice_and_silent_hold(
    tmp_path: Path, fake: Path,
) -> None:
    listing = sfx.generate(yes=True)
    rate = 48_000
    duration = 5
    t = np.arange(rate * duration) / rate
    synth.float_audio(tmp_path / "cut/voice.wav", (0.2 * np.sin(2 * np.pi * 220 * t)).astype(
        np.float32))
    transition = ResolvedElement("agent-join", "transition", "agent", 54, 66,
                                 "front", {"join_frame": 60})
    timeline = ResolvedTimeline((transition,), (), (), duration * 30, 45)
    sfx_input = {**listing, "ready": True}
    result = render_audio(tmp_path, tmp_path / "with.m4a", work_dir=tmp_path / "with",
                          timeline=timeline, policy={"mode": "preserve"}, music={},
                          sfx=sfx_input, sfx_settings={"sfx_min_gap_s": 2.5,
                                                       "sfx_peak_below_voice_db": 10})
    assert result.sfx and result.sfx["cues"]
    assert set(result.sfx["recipes"]) == {"whoosh", "click", "tick-roll", "chime"}
    assert all(len(value) == 64 for value in result.sfx["audio"].values())
    assert all(cue["relative_db"] <= -8 + 1e-5 for cue in result.sfx["cues"])
    mixed = decode(tmp_path / "with/mix.wav", [], channels=2)
    assert not mixed[duration * rate:].any()
    render_audio(tmp_path, tmp_path / "without.m4a", work_dir=tmp_path / "without",
                 timeline=timeline, policy={"mode": "preserve"}, music={},
                 sfx={"kit": "none", "sounds": [], "ready": False})
    baseline = decode(tmp_path / "without/mix.wav", [], channels=1)
    render_audio(tmp_path, tmp_path / "no-sfx-argument.m4a",
                 work_dir=tmp_path / "no-sfx-argument", timeline=timeline,
                 policy={"mode": "preserve"}, music={})
    np.testing.assert_array_equal(
        baseline, decode(tmp_path / "no-sfx-argument/mix.wav", [], channels=1))
    voice = decode(tmp_path / "cut/voice.wav", [], channels=1)
    np.testing.assert_array_equal(baseline[:duration * rate], voice)
    render_audio(tmp_path, tmp_path / "missing.m4a", work_dir=tmp_path / "missing",
                 timeline=timeline, policy={"mode": "preserve"}, music={},
                 sfx={"kit": "soft", "sounds": [], "ready": False,
                      "account": "missing_cli"})
    np.testing.assert_array_equal(decode(tmp_path / "missing/mix.wav", [], channels=1),
                                  baseline)


def test_count_up_receipt_records_overlapping_tick_and_chime(
    tmp_path: Path, fake: Path,
) -> None:
    listing = sfx.generate(yes=True)
    rate = 48_000
    t = np.arange(rate * 5) / rate
    synth.float_audio(tmp_path / "cut/voice.wav", (0.2 * np.sin(2 * np.pi * 220 * t)).astype(
        np.float32))
    timeline = ResolvedTimeline((_item("progress_stat", {}),), (), (), 150, 45)
    result = render_audio(tmp_path, tmp_path / "count.m4a", work_dir=tmp_path / "count",
                          timeline=timeline, policy={"mode": "preserve"}, music={},
                          sfx={**listing, "ready": True},
                          sfx_settings={"sfx_min_gap_s": 2.5,
                                        "sfx_peak_below_voice_db": 10})
    assert result.sfx is not None
    pair = [cue for cue in result.sfx["cues"] if cue["element_id"] == "agent-card"]
    assert [cue["sound"] for cue in pair] == ["tick-roll", "chime"]
    assert [cue["time_s"] for cue in pair] == pytest.approx([1.25, 2.1])


@pytest.mark.parametrize(("look", "count", "frames", "cues", "times"), [
    ("tags", 2, 90, [], [1 + 0.1 + 5 / 30, 2.8 - 0.0006]),
    ("tags", 3, 90, [], [1 + 0.1 + 5 / 30, 1 + 31 / 30 - 0.0003, 2.8 - 0.0006]),
    ("tags", 3, 45, [], [1 + 0.1 + 5 / 30, 1 + 19 / 30, 2.0]),
    ("tags", 2, 90, [0.8, 1.3], [1.8 + 5 / 30, 2.3 + 5 / 30]),
    ("tags", 3, 90, [0.8, 1.3, 1.7], [1.8 + 5 / 30, 2.3 + 5 / 30, 2.7 + 5 / 30]),
    ("tags", 3, 90, [1.7], [2.7 + 5 / 30] * 3),
    ("boxes", 2, 90, [], [1.99, 2.6]),
    ("boxes", 2, 90, [0.8, 1.3], [2.25, 2.75]),
    ("boxes", 2, 90, [0.8], [2.25, 2.6]),
])
def test_flow_clicks_follow_plan(
    look: str, count: int, frames: int, cues: list[float], times: list[float],
) -> None:
    data = resolve_style_data({"schema_version": 1, "graphics": {"flow_look": look}})
    style = ResolvedStyle(data["name"], "test", data)
    slots = {"nodes": ["A", "B", "C"][:count], "cues": cues}
    plan = cards.card_layer_plan("flow", slots, style, frames=frames)
    item = _item("flow", slots, end=30 + frames)
    timeline = ResolvedTimeline((item,), (), (), 150, 45)
    sound = {"name": "click", "rule": "item", "priority": 1, "duration_s": 0.5}
    actual = candidates(timeline, [sound], flow_look=style.data["graphics"]["flow_look"])
    assert [cue.event for cue in actual] == [f"flow:{index + 1}" for index in range(count)]
    assert [cue.time_s for cue in actual] == pytest.approx(times)
    if look == "tags":
        # Default cues use the renderer's duration, which is 1 ms short of the frame window.
        assert [cue.time_s for cue in actual] == pytest.approx([
            item.start_s + cue + 5 / 30 for cue in json.loads(plan.variables["cues"])
        ])


def test_flow_tags_audio_receipt_uses_tag_landings(tmp_path: Path, fake: Path) -> None:
    listing = sfx.generate(yes=True)
    t = np.arange(48_000 * 5) / 48_000
    synth.float_audio(tmp_path / "cut/voice.wav", (0.2 * np.sin(2 * np.pi * 220 * t)).astype(
        np.float32))
    timeline = ResolvedTimeline((_item("flow", {"nodes": ["A", "B"]}),), (), (), 150, 45)
    result = render_audio(tmp_path, tmp_path / "tags.m4a", work_dir=tmp_path / "tags",
                          timeline=timeline, policy={"mode": "preserve"}, music={},
                          sfx={**listing, "ready": True}, sfx_settings={"sfx_min_gap_s": 0},
                          flow_look="tags")
    assert result.sfx is not None
    clicks = [cue for cue in result.sfx["cues"] if cue["element_id"] == "agent-card"]
    assert [cue["event"] for cue in clicks] == ["flow:1", "flow:2"]
    expected = [round(at * 48_000) / 48_000 for at in (1 + 0.1 + 5 / 30, 2.8 - 0.0006)]
    assert [cue["time_s"] for cue in clicks] == expected


@pytest.mark.parametrize("look", ["together", "on_cue", "on_cue_glow"])
def test_specs_rows_style_reaches_audio_placement(tmp_path: Path, fake: Path, look: str) -> None:
    listing = sfx.generate(yes=True)
    rate = 48_000
    t = np.arange(rate * 5) / rate
    synth.float_audio(tmp_path / "cut/voice.wav", (0.2 * np.sin(2 * np.pi * 220 * t)).astype(
        np.float32))
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["specs_rows"] = look
    slots = {"title": "Plan", "rows": [{"name": "First", "value": "12"},
                                       {"name": "Second", "value": "25"}]}
    plan = cards.card_layer_plan("specs_list", slots, style, frames=90)
    timeline = ResolvedTimeline((_item("specs_list", slots),), (), (), 150, 45)
    result = render_audio(tmp_path, tmp_path / "rows.m4a", work_dir=tmp_path / "rows",
                          timeline=timeline, policy={"mode": "preserve"}, music={},
                          sfx={**listing, "ready": True},
                          sfx_settings={"sfx_min_gap_s": 0.1,
                                        "sfx_peak_below_voice_db": 10},
                          specs_rows=style.data["graphics"]["specs_rows"])
    assert result.sfx is not None
    clicks = [cue for cue in result.sfx["cues"] if cue["element_id"] == "agent-card"]
    assert [cue["event"] for cue in clicks] == ["specs_list:1", "specs_list:2"]
    expected = ([1.2, 1.33] if look == "together" else
                [1 + cue for cue in json.loads(plan.variables["cues"])])
    assert [cue["time_s"] for cue in clicks] == pytest.approx(expected)


def test_specific_rules_prefer_creator_sounds_and_fall_back() -> None:
    timeline = ResolvedTimeline((
        _item("flow", {"nodes": ["A", "B"], "cues": [1, 2]}),
        _item("gravity_drop", {"drops": [{"at": .5}]}),
    ), (), (), 180, 45)
    soft = sfx.load_kit()["sounds"]
    cues = candidates(timeline, soft, flow_look="tags")
    assert next(cue for cue in cues if cue.event == "flow:1").sound == "click"
    assert next(cue for cue in cues if cue.event == "gravity_drop:land").sound == "chime"
    pop = {**soft[1], "name": "custom-pop", "rule": "pop"}
    cues = candidates(timeline, [*soft, pop], flow_look="tags")
    assert next(cue for cue in cues if cue.event == "flow:1").sound == "custom-pop"
