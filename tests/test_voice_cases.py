"""Regression cases for voice fixes."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import synth
from test_cut import time_segment, write_edit
from test_voice_make import prepared
from voice_helpers import calls, move_word

from yap_studio import cli, elevenlabs
from yap_studio.errors import YapError
from yap_studio.project.receipts import projection_sha256
from yap_studio.stages import assemble as assemble_stage
from yap_studio.stages import cut as cut_stage
from yap_studio.voice import decide, make, propose
from yap_studio.voice.consent import require_account
from yap_studio.voice.make import _request
from yap_studio.voice.splice import _tone, build_patch, decode


def noisy_take():
    rate = 48000
    sound = np.random.default_rng(73).normal(0, 0.001, (8 * rate, 1)).astype(np.float32)
    sound[: round(0.15 * rate)] = 0
    spans = [(0.15, 0.35), (1, 1.8), (3, 4.6), (5.5, 6.5)]
    for a, b in spans:
        first, last = round(a * rate), round(b * rate)
        sound[first:last, 0] += 0.08 * np.sin(2 * np.pi * 220 * np.arange(last - first) / rate)
    return sound, spans


def test_room_tone_uses_nearby_floor_instead_of_silent_gap(tmp_path):
    sound, spans = noisy_take()
    speech = (0.02 * np.sin(2 * np.pi * 260 * np.arange(24000) / 48000))[:, None]
    generated = synth.float_audio(tmp_path / "generated.wav", speech)
    _, _, checks = build_patch(
        sound, generated, 3, 4.6, tmp_path, word_spans=spans, word_range=(2, 2)
    )
    tone = next(row for row in checks["items"] if row["id"] == "splice.room_tone")
    assert tone["source_floor_db"] == pytest.approx(-60, abs=3)
    assert tone["patch_floor_db"] == pytest.approx(-60, abs=3)
    assert tone["status"] == "pass" and tone["measured"]


def test_room_tone_check_detects_generated_noise(tmp_path):
    sound, spans = noisy_take()
    speech = np.random.default_rng(9).normal(0, 0.001, (76800, 1)).astype(np.float32)
    for first, last in [(0, 19200), (57600, 76800)]:
        speech[first:last, 0] += 0.02 * np.sin(2 * np.pi * 260 * np.arange(last - first) / 48000)
    generated = synth.float_audio(tmp_path / "generated.wav", speech)
    _, _, checks = build_patch(
        sound, generated, 3, 4.6, tmp_path, word_spans=spans, word_range=(2, 2)
    )
    tone = next(row for row in checks["items"] if row["id"] == "splice.room_tone")
    assert tone["measured"] and tone["difference_db"] > 3
    assert tone["status"] in ("warn", "fail")


def test_room_tone_refuses_gaps_shorter_than_100ms():
    sound = np.random.default_rng(5).normal(0, 0.001, (48000, 1)).astype(np.float32)
    with pytest.raises(YapError) as caught:
        _tone(sound, 24000, 33600, [(0, 0.4), (0.49, 0.5), (0.7, 1)], 9600)
    assert caught.value.code == "voice_room_tone_missing"


def test_room_tone_joins_several_quiet_gaps():
    samples = np.arange(6 * 48000)
    sound = (0.08 * np.sin(2 * np.pi * 220 * samples / 48000))[:, None].astype(np.float32)
    for first, last, frequency in [(0.5, 1.1, 130), (2, 2.6, 370)]:
        a, b = round(first * 48000), round(last * 48000)
        sound[a:b, 0] = (
            np.sqrt(2) * 0.001 * np.sin(2 * np.pi * frequency * np.arange(b - a) / 48000)
        )
    tone, _ = _tone(sound, 4 * 48000, 5 * 48000, [(0, 0.5), (1.1, 2), (2.6, 6)], 48000)
    spectrum = np.abs(np.fft.rfft(tone[:, 0]))
    # Both distinct recorded gaps must contribute, rather than looping just one.
    assert min(spectrum[130], spectrum[370]) > max(spectrum[130], spectrum[370]) * 0.1


def test_room_tone_accepts_a_full_100ms_gap():
    sound = np.random.default_rng(17).normal(0, 0.001, (48000, 1)).astype(np.float32)
    tone, floor = _tone(sound, 28800, 38400, [(0, 0.4), (0.5, 1)], 9600)
    assert len(tone) == 9600
    assert floor == pytest.approx(-60, abs=3)


def test_unknown_account_is_allowed(monkeypatch):
    monkeypatch.setattr(elevenlabs, "account_state", lambda: "unknown")
    require_account()


def test_budget_permission_fallbacks_keep_cost(voice_project, fake_voice, monkeypatch):
    prepared(voice_project, monkeypatch)
    monkeypatch.setenv(
        "FAKE_ELEVENLABS_MISSING_PERMISSIONS",
        json.dumps({"user get-subscription-info": "user_read", "models": "models_read"}),
    )
    fix = propose(voice_project, "t1:6", "t1:6", "water", clone=True)
    assert fix["estimate"]["credits"] == 5
    assert fix["estimate"]["unavailable"]["models"] == "models_read"
    result = make(voice_project, ["f1"], dry_run=True)
    assert result["cost"]["credits"] == 11
    assert result["cost"]["unavailable"]["plan"]["permission"] == "user_read"
    assert result["cost"]["lines"] == [
        "Re-saying 'cold plunge' costs about 11 ElevenLabs credits (11 characters). "
        "Your monthly credits could not be read because your ElevenLabs key cannot read your plan."
    ]
    assert not any(argv[0] == "text-to-speech" for argv in calls(fake_voice))
    # The missing free-read permissions do not prevent a paid request with the other permissions.
    assert make(voice_project, ["f1"], yes=True, max_credits=20)["status"] == "ready"


@pytest.mark.parametrize(
    "command,permission,plain",
    [
        ("voices get-user-v2", "voices_read", "Voices read"),
        ("voices get-by-id", "voices_read", "Voices read"),
        ("voices add", "voices_write", "Voices write"),
        ("voices delete", "voices_write", "Voices write"),
        ("text-to-speech", "text_to_speech", "Text to Speech"),
        ("voices get-user-v2", "custom_permission", "custom_permission"),
    ],
)
def test_voice_calls_report_missing_permission(
    fake_voice, monkeypatch, tmp_path, command, permission, plain
):
    monkeypatch.setenv("FAKE_ELEVENLABS_MISSING_PERMISSIONS", json.dumps({command: permission}))
    identifier = "fake-personal-voice-secret"
    request = {"text": "water", "previous_text": "", "next_text": "", "seed": 1}
    operations = {
        "voices get-user-v2": lambda: elevenlabs.personal_voices("My voice"),
        "voices get-by-id": lambda: elevenlabs.voice_details(identifier),
        "voices add": lambda: elevenlabs.add_voice("My voice", [tmp_path / "sample.wav"]),
        "voices delete": lambda: elevenlabs.remove_voice(identifier),
        "text-to-speech": lambda: elevenlabs.generate_voice(
            identifier, request, tmp_path / "out.mp3"
        ),
    }
    with pytest.raises(YapError) as caught:
        operations[command]()
    assert caught.value.code == "elevenlabs_permission_missing"
    assert str(caught.value) == (
        f"Your ElevenLabs key is missing the {permission} permission. "
        f"In ElevenLabs, open Developers, then API keys, edit your key, "
        f"turn on {plain}, then try again."
    )
    assert identifier not in json.dumps(caught.value.to_dict())


@pytest.mark.parametrize(
    "error",
    [YapError("History could not be read."), OSError("History could not be read.")],
)
def test_history_read_errors_use_estimate(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(elevenlabs, "_voice_json", fail)
    assert elevenlabs.voice_charge("test-voice", "water", 0, 5) == {
        "credits": 5, "source": "estimate"
    }


@pytest.mark.parametrize("word_id,field", [("t1:2", "start"), ("t1:3", "end")])
@pytest.mark.parametrize("phase", ["before_make", "during_generation", "cut"])
def test_moved_words_refuse_stale_fix(
    voice_project, fake_voice, monkeypatch, word_id, field, phase
):
    prepared(voice_project, monkeypatch)

    def move():
        move_word(voice_project, word_id, field, 0.02)

    if phase == "cut":
        make(voice_project, ["f1"], yes=True, max_credits=20)
        decide(voice_project, "f1", "approve", via="chat")
        move()
    elif phase == "during_generation":
        generate = elevenlabs.generate_voice

        def generate_and_move(*args):
            generate(*args)
            move()

        monkeypatch.setattr(elevenlabs, "generate_voice", generate_and_move)
    else:
        move()
    with pytest.raises(
        YapError, match=r"^The words moved since this fix was made\. Propose it again\.$"
    ) as caught:
        if phase == "cut":
            cut_stage.cut(voice_project)
        else:
            make(voice_project, ["f1"], yes=True, max_credits=20)
    assert caught.value.code == "voice_words_changed"
    assert sum(argv[0] == "text-to-speech" for argv in calls(fake_voice)) == (
        0 if phase == "before_make" else 1
    )
    assert not (voice_project / "cut/voice.wav").exists()


@pytest.mark.parametrize(
    "command,args,blocked,permission",
    [
        ("link", ["--name", "My voice", "--yes"], "voices get-user-v2", "voices_read"),
        (
            "make",
            ["--fix", "f1", "--yes", "--max-credits", "20"],
            "text-to-speech",
            "text_to_speech",
        ),
    ],
)
def test_permission_errors_reach_cli(
    voice_project, fake_voice, monkeypatch, capsys, command, args, blocked, permission
):
    prepared(voice_project, monkeypatch)
    monkeypatch.setenv("FAKE_ELEVENLABS_MISSING_PERMISSIONS", json.dumps({blocked: permission}))
    assert cli.main(["voice", command, str(voice_project), *args, "--json"]) == 2
    captured = capsys.readouterr()
    error = json.loads(captured.err)["error"]
    assert error["code"] == "elevenlabs_permission_missing"
    assert permission in error["message"]
    assert "fake-personal-voice-secret" not in captured.err
    assert not (voice_project / ".yap/voice/pending.json").exists()


@pytest.mark.parametrize(
    "code,stderr,refused",
    [
        (5, "HTTP 429 too many requests", True),
        (5, "HTTP 402 payment required", True),
        (5, "HTTP 499 request cancelled", True),
        (5, "HTTP 403 refused", True),
        (4, "request refused", True),
        (5, "HTTP 500 request 4012 failed", False),
        (5, "request 40001 failed", False),
        (5, "timeout after 4000 ms", False),
    ],
)
def test_http_status_controls_pending(
    voice_project, fake_voice, monkeypatch, code, stderr, refused
):
    prepared(voice_project, monkeypatch)
    monkeypatch.setenv("FAKE_ELEVENLABS_GENERATE_EXIT", str(code))
    monkeypatch.setenv("FAKE_ELEVENLABS_GENERATE_STDERR", stderr)
    with pytest.raises(YapError) as caught:
        make(voice_project, ["f1"], yes=True, max_credits=20)
    assert caught.value.code == (
        "voice_generation_refused" if refused else "voice_generation_unreconciled"
    )
    assert (voice_project / ".yap/voice/pending.json").exists() is not refused


def test_two_words_to_one_in_spans_and_cut_words(voice_project, fake_voice, monkeypatch):
    prepared(voice_project, monkeypatch)
    # Start again with the same two source ids and an anchored story segment.
    from test_cut import write_json

    write_json(
        voice_project / "voice/fixes.json", {"schema": "yap-studio/voice-fixes@1", "fixes": []}
    )
    write_edit(
        voice_project,
        [
            {
                "id": "s1",
                "take": "t1",
                "from": {"word": "t1:2"},
                "to": {"word": "t1:3", "edge": "end"},
            }
        ],
    )
    propose(voice_project, "t1:2", "t1:3", "overcharged")
    make(voice_project, ["f1"], yes=True, max_credits=20)
    decide(voice_project, "f1", "approve", via="chat")
    cut_stage.cut(voice_project)
    spans = json.loads((voice_project / "cut/spans.json").read_text())["segments"]
    words = json.loads((voice_project / "cut/cut-words.json").read_text())["words"]
    assert [row["text"] for row in words] == ["overcharged", ""]
    assert spans[0]["first_word"] == "overcharged"
    assert not spans[0].get("last_word")


def test_generation_context_uses_kept_words_in_cut_order(voice_project):
    write_edit(
        voice_project,
        [
            time_segment("water", "t1", 7, 8),
            time_segment("fix", "t1", 2, 5),
            time_segment("bright", "t1", 0, 1),
        ],
    )
    fix = propose(voice_project, "t1:2", "t1:3", "cold plunge")
    request, _ = _request(
        cut_stage.resolve_cut(voice_project), fix, {"voice": {"id_sha256": "1" * 64}}, new=False
    )
    assert request["previous_text"] == "water"
    assert request["next_text"] == "project bright"


def test_review_clips_have_two_seconds_each_side(voice_project, fake_voice, monkeypatch):
    prepared(voice_project, monkeypatch)
    fix = make(voice_project, ["f1"], yes=True, max_credits=20)["fixes"][0]
    for clip in fix["clips"].values():
        assert len(decode(voice_project / clip["path"], 1)) == fix["patch"]["samples"] + 4 * 48000


def test_make_cost_is_printed_once(voice_project, fake_voice, monkeypatch, capsys):
    prepared(voice_project, monkeypatch)
    assert cli.main(["voice", "make", str(voice_project), "--fix", "f1"]) == 2
    captured = capsys.readouterr()
    assert (captured.out + captured.err).count("Re-saying 'cold plunge'") == 1
    assert "error: Approve this cost first." in captured.err
    assert "fix:" in captured.err
    assert not any(argv[0] == "text-to-speech" for argv in calls(fake_voice))


def test_voice_subcommands_have_help(capsys):
    assert cli.main(["voice", "--help"]) == 0
    text = capsys.readouterr().out
    for description in (
        "export clean speech samples",
        "record or withdraw voice consent",
        "create your voice",
        "link your own voice",
        "propose a word fix",
        "generate proposed fixes",
        "approve a ready fix",
        "keep the original words",
        "list voice fixes",
        "delete your linked voice",
    ):
        assert description in text


@pytest.mark.parametrize("talking_head_fix", [False, True])
def test_voice_assembly_checks_alpha_only_for_talking_head(tmp_path, monkeypatch, talking_head_fix):
    takes = [
        SimpleNamespace(id="t1", voiceover=not talking_head_fix),
        SimpleNamespace(id="t2", voiceover=False),
    ]
    fix = {"id": "f1", "take": "t1"}
    projection = {"consent": None, "approved": [fix]}
    context = SimpleNamespace(
        project_dir=tmp_path,
        project=SimpleNamespace(takes=takes, audio={}),
        json=lambda _: {"inputs": {"voice:fixes": projection_sha256(projection)}},
        render_style=lambda: SimpleNamespace(data={
            "voice": {}, "hook": {"style": "none", "answered": True},
        }),
    )
    snapshot = SimpleNamespace(
        sfx={"kit": "none"}, timeline=SimpleNamespace(total_frames=1, elements=[])
    )
    monkeypatch.setattr(assemble_stage, "ProjectContext", lambda _: context)
    monkeypatch.setattr(assemble_stage, "validate_voice_chain", lambda *_: None)
    monkeypatch.setattr(assemble_stage, "assembly_snapshot", lambda *_, **__: snapshot)
    monkeypatch.setattr(assemble_stage, "_pilot_phrases", lambda _: (b"words", "hash"))
    monkeypatch.setattr(assemble_stage, "_pilot_window", lambda *_: None)
    monkeypatch.setattr(assemble_stage, "_pilot_assemble", lambda *_, **__: {"status": "verified"})
    monkeypatch.setattr(cut_stage, "resolve_cut", lambda _: object())
    from yap_studio.voice import store

    monkeypatch.setattr(store, "approved_fixes", lambda _: [fix])
    monkeypatch.setattr(store, "projection", lambda _, *, fixes: projection)
    monkeypatch.setattr(store, "patches_for_plan", lambda _, *, fixes: [(fix, np.zeros((10, 1)))])
    monkeypatch.setattr(store, "require_assembly_picture", lambda *_: None)
    calls = []
    monkeypatch.setattr(
        assemble_stage, "_graphic_picture_frames", lambda *_, **__: calls.append("alpha") or {}
    )
    assert assemble_stage._assemble_impl(tmp_path, frames=(0, 0))["status"] == "verified"
    assert len(calls) == int(talking_head_fix)


def test_token_setup_line_includes_all_voice_permissions():
    assert elevenlabs.TOKEN_LINE == (
        "Then create your own ElevenLabs API key with Music, Sound Effects, Text to Speech, "
        "Voices, History, Models and User read access, and save it: "
        "elevenlabs-pp-cli auth set-token <your-key>"
    )
