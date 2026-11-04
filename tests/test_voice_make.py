"""Paid requests need approval, a ceiling and a reconciled previous attempt."""

import json
from contextlib import contextmanager
from importlib import import_module

import pytest
from test_cut import write_json
from voice_helpers import calls, linked

from yap_studio import receipt
from yap_studio.errors import YapError
from yap_studio.voice import decide, list_fixes, make, propose


def test_generation_rechecks_cache_before_reserving(tmp_path, monkeypatch):
    module = import_module("yap_studio.voice.make")
    generated = []
    request = {"text": "cold plunge"}
    request_hash = "a" * 64

    def generate(identifier, request, target):
        generated.append(request)
        assert len(generated) == 1, "The same request was paid twice."
        target.write_bytes(b"cached speech")

    def other_caller_finishes():
        monkeypatch.setattr(module, "require_account", lambda: None)
        module._generated(tmp_path, "voice", request, request_hash, 11, {}, confirm_retry=False)

    monkeypatch.setattr(module, "require_account", other_caller_finishes)
    monkeypatch.setattr(module, "require_consent", lambda _: None)
    monkeypatch.setattr(module, "decode", lambda *_: None)
    monkeypatch.setattr(module.elevenlabs, "generate_voice", generate)
    monkeypatch.setattr(module.elevenlabs, "voice_charge",
                        lambda *_: {"credits": 9, "source": "history"})
    cache, _ = module._generated(
        tmp_path, "voice", request, request_hash, 11, {}, confirm_retry=False,
    )
    assert cache.read_bytes() == b"cached speech"
    assert len(generated) == 1
    assert not (tmp_path / ".yap/voice/pending.json").exists()


@pytest.mark.parametrize("content", [b"", b"changed speech"])
def test_generation_validates_cache_that_appeared_before_reserving(tmp_path, monkeypatch, content):
    module = import_module("yap_studio.voice.make")
    request_hash = "a" * 64

    def cache_appears():
        cache = tmp_path / f".yap/cache/voice/{request_hash}.mp3"
        cache.parent.mkdir(parents=True)
        cache.write_bytes(content)

    def generate(*_):
        pytest.fail("An invalid cache must not start a paid call.")

    monkeypatch.setattr(module, "require_account", cache_appears)
    monkeypatch.setattr(module, "require_consent", lambda _: None)
    monkeypatch.setattr(module.elevenlabs, "generate_voice", generate)
    with pytest.raises(YapError) as caught:
        module._generated(tmp_path, "voice", {"text": "cold plunge"}, request_hash, 11,
                          {"request_sha256": request_hash, "output_sha256": "b" * 64},
                          confirm_retry=False)
    assert caught.value.code == "voice_cache_invalid"
    assert not (tmp_path / ".yap/voice/pending.json").exists()


def prepared(root, monkeypatch):
    linked(root, monkeypatch)
    return propose(root, "t1:2", "t1:3", "cold plunge", clone=True)


def test_cost_and_no_spend(voice_project, fake_voice, monkeypatch):
    prepared(voice_project, monkeypatch)
    with pytest.raises(YapError) as caught:
        make(voice_project, ["f1"])
    assert caught.value.code == "voice_approval_required"
    assert "11 ElevenLabs credits (11 characters)" in caught.value.cost["lines"][0]
    assert "81,234 credits left" in caught.value.cost["lines"][0]
    result = make(voice_project, ["f1"], dry_run=True)
    assert result["cost"]["credits"] == 11
    assert not any(a[0] == "text-to-speech" for a in calls(fake_voice))
    with pytest.raises(YapError) as caught:
        make(voice_project, ["f1"], yes=True, max_credits=10)
    assert caught.value.code == "voice_over_ceiling"
    with pytest.raises(YapError) as caught:
        make(voice_project, ["f1", "f2", "f3", "f4"], yes=True)
    assert caught.value.code == "voice_too_many_fixes"


def test_history_cache_and_hygiene(voice_project, fake_voice, monkeypatch):
    prepared(voice_project, monkeypatch)
    monkeypatch.setenv("FAKE_ELEVENLABS_CHARGED", "9")
    result = make(voice_project, ["f1"], yes=True, max_credits=20)
    fix = result["fixes"][0]
    assert fix["status"] == "ready"
    assert fix["generation"]["cost"] == {"credits": 9, "source": "history"}
    assert {x["id"] for x in fix["checks"]["items"]} == {
        "splice.click",
        "splice.level",
        "splice.room_tone",
        "splice.pace",
        "splice.heard",
    }
    heard = next(item for item in fix["checks"]["items"] if item["id"] == "splice.heard")
    assert heard["status"] == "skipped"
    assert heard["reason"] == "No local speech engine is available."
    make(voice_project, ["f1"], yes=True, max_credits=20)
    assert sum(a[0] == "text-to-speech" for a in calls(fake_voice)) == 1
    forbidden = b"fake-personal-voice-secret"
    assert forbidden not in json.dumps(result).encode()
    for target in voice_project.rglob("*"):
        if target.is_file():
            assert forbidden not in target.read_bytes(), target


def test_history_permission_keeps_generated_fix_ready(voice_project, fake_voice, monkeypatch):
    prepared(voice_project, monkeypatch)
    monkeypatch.setenv(
        "FAKE_ELEVENLABS_MISSING_PERMISSIONS",
        json.dumps({"history get-speech": "speech_history_read"}),
    )
    result = make(voice_project, ["f1"], yes=True, max_credits=20)
    fix = result["fixes"][0]
    assert result["status"] == fix["status"] == "ready"
    assert fix["generation"]["cost"] == {"credits": 11, "source": "estimate"}
    assert list_fixes(voice_project)["fixes"] == [fix]
    assert set(fix["clips"]) == {"before", "after"}
    for row in (fix["patch"], *fix["clips"].values()):
        assert (voice_project / row["path"]).stat().st_size > 0
    assert not (voice_project / ".yap/voice/pending.json").exists()
    make(voice_project, ["f1"], yes=True, max_credits=20)
    assert sum(a[0] == "text-to-speech" for a in calls(fake_voice)) == 1


def test_pending_retry(voice_project, fake_voice, monkeypatch):
    prepared(voice_project, monkeypatch)
    write_json(
        voice_project / ".yap/voice/pending.json",
        {
            "schema": "yap-studio/voice-pending@1",
            "state": "unreconciled",
            "pid": 0,
            "host": "old",
            "request_sha256": "1" * 64,
        },
    )
    with pytest.raises(YapError) as caught:
        make(voice_project, ["f1"], yes=True, max_credits=20)
    assert caught.value.code == "voice_generation_unreconciled"
    make(voice_project, ["f1"], yes=True, max_credits=20, confirm_retry=True)
    assert sum(a[0] == "text-to-speech" for a in calls(fake_voice)) == 1


def test_new_generation_and_multiple_costs(voice_project, fake_voice, monkeypatch):
    prepared(voice_project, monkeypatch)
    propose(voice_project, "t1:6", "t1:6", "water", clone=True)
    preview = make(voice_project, ["f1", "f2"], dry_run=True)
    assert preview["cost"]["credits"] == 16 and len(preview["cost"]["lines"]) == 3
    first = make(voice_project, ["f1"], yes=True, max_credits=20)["fixes"][0]
    second = make(voice_project, ["f1"], yes=True, max_credits=20, new=True)["fixes"][0]
    assert first["generation"]["request_sha256"] != second["generation"]["request_sha256"]
    assert sum(a[0] == "text-to-speech" for a in calls(fake_voice)) == 2
    assert first["patch"]["path"] != second["patch"]["path"]
    for row in (first["patch"], *first["clips"].values()):
        assert not (voice_project / row["path"]).exists()


@pytest.mark.parametrize("failure", ["before_owner", "before_write", "record"])
def test_regeneration_publication_failure_keeps_previous_generation(
    voice_project, fake_voice, monkeypatch, failure,
):
    module = import_module("yap_studio.voice.make")
    prepared(voice_project, monkeypatch)
    first = make(voice_project, ["f1"], yes=True, max_credits=20)["fixes"][0]
    records = (first["patch"], *first["clips"].values())
    original = {row["path"]: (voice_project / row["path"]).read_bytes() for row in records}
    before = voice_project / first["clips"]["before"]["path"]
    monkeypatch.setenv("FAKE_ELEVENLABS_SPEECH_SECONDS", "1.3")
    if failure == "before_owner":
        before.write_bytes(original[first["clips"]["before"]["path"]] + b"hand edit")
    with monkeypatch.context() as patch:
        if failure == "before_write":
            atomic_output = module.paths.atomic_output

            @contextmanager
            def fail_before(target, *args, **kwargs):
                if target.name == "before.wav":
                    raise OSError("before publication failed")
                with atomic_output(target, *args, **kwargs) as staged:
                    yield staged

            patch.setattr(module.paths, "atomic_output", fail_before)
        elif failure == "record":
            write = module.write

            def fail_record(root, kind, data):
                if kind == "fixes" and data["fixes"][0]["generation"] != first["generation"]:
                    raise OSError("record publication failed")
                return write(root, kind, data)

            patch.setattr(module, "write", fail_record)
        with pytest.raises((YapError, OSError)):
            make(voice_project, ["f1"], yes=True, max_credits=20, new=True)
    assert list_fixes(voice_project)["fixes"] == [first]
    assert {str(path.relative_to(voice_project))
            for path in (voice_project / "voice/fixes").rglob("*.wav")} == set(original)
    assert sum(a[0] == "text-to-speech" for a in calls(fake_voice)) == (
        1 if failure == "before_owner" else 2
    )
    for row in records:
        target = voice_project / row["path"]
        if failure == "before_owner" and target == before:
            assert target.read_bytes() == original[row["path"]] + b"hand edit"
        else:
            assert receipt.sha256_file(target) == row["sha256"]
            assert target.read_bytes() == original[row["path"]]
    if failure == "before_owner":
        before.write_bytes(original[first["clips"]["before"]["path"]])
    latest = make(voice_project, ["f1"], yes=True, max_credits=20, new=True)["fixes"][0]
    assert latest["generation"] != first["generation"]
    for row in (latest["patch"], *latest["clips"].values()):
        assert receipt.sha256_file(voice_project / row["path"]) == row["sha256"]
    assert decide(voice_project, "f1", "approve", via="review")["status"] == "approved"


@pytest.mark.parametrize("stage", ["make", "cut"])
@pytest.mark.parametrize("spoken", [False, True])
def test_heard_checks_only_replacement_audio(
    voice_project, fake_voice, monkeypatch, tmp_path, stage, spoken,
):
    import numpy as np

    from yap_studio.stages import cut as cut_stage
    from yap_studio.transcript import ScriptedBackend
    from yap_studio.transcript.model import Transcript, Word
    from yap_studio.voice.splice import decode
    from yap_studio.voice.store import add_cut_checks

    module = import_module("yap_studio.voice.make")
    prepared(voice_project, monkeypatch)
    build_patch = module.build_patch

    def build(*args, **kwargs):
        signal, meta, checks = build_patch(*args, **kwargs)
        return signal if spoken else np.zeros_like(signal), meta, checks

    heard_audio = []

    def transcribe(path):
        signal = decode(path, 1)
        heard_audio.append(signal)
        words = [Word("cold", 0.2, 0.4), Word("plunge", 0.4, 0.6)]
        return Transcript(words=words if np.any(np.abs(signal) > 0.001) else [])

    monkeypatch.setattr(module, "build_patch", build)
    backend = ScriptedBackend(transcribe)
    monkeypatch.setattr(cut_stage, "_edge_backend", lambda _: backend)
    fix = make(voice_project, ["f1"], yes=True, max_credits=20)["fixes"][0]
    assert np.any(np.abs(decode(voice_project / fix["clips"]["before"]["path"], 1)) > 0.001)
    if stage == "cut":
        meta = fix["patch"]
        first, last = meta["start_sample"], meta["start_sample"] + meta["samples"]
        sound = np.full((12 * 48000, 1), 0.1, dtype=np.float32)
        sound[first:last] = decode(voice_project / meta["path"], 1)
        report = add_cut_checks(
            {"findings": []}, sound,
            [{"id": "f1", "start_sample": first, "end_sample": last}], [fix],
            plan=cut_stage.resolve_cut(voice_project), work=tmp_path, levels=None,
        )
        checks = report["voice_fixes"][0]["checks"]
    else:
        checks = fix["checks"]
    heard = next(item for item in checks["items"] if item["id"] == "splice.heard")
    assert heard["status"] == ("pass" if spoken else "warn")
    assert len(heard_audio[-1]) == fix["patch"]["samples"]


@pytest.mark.parametrize("refused", [False, True])
def test_failure_redaction_and_retry_state(voice_project, fake_voice, monkeypatch, refused):
    prepared(voice_project, monkeypatch)
    monkeypatch.setenv("FAKE_ELEVENLABS_GENERATE_EXIT", "4" if refused else "5")
    monkeypatch.setenv(
        "FAKE_ELEVENLABS_GENERATE_STDERR", "HTTP 403 refused" if refused else "timeout"
    )
    with pytest.raises(YapError) as caught:
        make(voice_project, ["f1"], yes=True, max_credits=20)
    assert "fake-personal-voice-secret" not in str(caught.value)
    assert "fake-personal-voice-secret" not in json.dumps(caught.value.to_dict())
    pending = voice_project / ".yap/voice/pending.json"
    assert pending.exists() is not refused
    for target in voice_project.rglob("*"):
        if target.is_file():
            assert b"fake-personal-voice-secret" not in target.read_bytes()
    if not refused:
        monkeypatch.delenv("FAKE_ELEVENLABS_GENERATE_EXIT")
        with pytest.raises(YapError) as blocked:
            make(voice_project, ["f1"], yes=True, max_credits=20)
        assert blocked.value.code == "voice_generation_unreconciled"


def test_model_rate_estimate_and_history_fallback(voice_project, fake_voice, monkeypatch):
    prepared(voice_project, monkeypatch)
    monkeypatch.setenv("FAKE_ELEVENLABS_MULTIPLIER", "0.5")
    assert make(voice_project, ["f1"], dry_run=True)["cost"]["credits"] == 5.5
    monkeypatch.setenv("FAKE_ELEVENLABS_MODELS_FAIL", "1")
    result = make(voice_project, ["f1"], dry_run=True)
    assert result["cost"]["credits"] == 11 and result["cost"]["source"] == "estimate"
    monkeypatch.setenv("FAKE_ELEVENLABS_HISTORY_EMPTY", "1")
    result = make(voice_project, ["f1"], yes=True, max_credits=20)
    assert result["fixes"][0]["generation"]["cost"] == {"credits": 11, "source": "estimate"}


def test_mouth_rechecked_before_paying(voice_project, fake_voice, monkeypatch):
    prepared(voice_project, monkeypatch)
    data = json.loads((voice_project / "project.json").read_text())
    data["takes"][0]["voiceover"] = False
    write_json(voice_project / "project.json", data)
    with pytest.raises(YapError) as caught:
        make(voice_project, ["f1"], yes=True, max_credits=20)
    assert caught.value.code == "voice_mouth_on_screen"
    assert not any(a[0] == "text-to-speech" for a in calls(fake_voice))


def test_voice_changed_and_local_hearing(voice_project, fake_voice, monkeypatch):
    prepared(voice_project, monkeypatch)
    monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_ID", "different-secret-voice")
    with pytest.raises(YapError) as caught:
        make(voice_project, ["f1"], yes=True, max_credits=20)
    assert caught.value.code == "voice_changed"
    assert "different-secret-voice" not in str(caught.value)
    monkeypatch.delenv("FAKE_ELEVENLABS_VOICE_ID")
    from yap_studio.stages import cut as cut_stage
    from yap_studio.transcript import ScriptedBackend
    from yap_studio.transcript.model import Transcript, Word

    backend = ScriptedBackend(
        lambda _: Transcript(words=[Word("cold", 0, 0.4), Word("plunge", 0.4, 0.8)])
    )
    monkeypatch.setattr(cut_stage, "_edge_backend", lambda _: backend)
    result = make(voice_project, ["f1"], yes=True, max_credits=20)
    heard = next(
        row for row in result["fixes"][0]["checks"]["items"] if row["id"] == "splice.heard"
    )
    assert heard["status"] == "pass"
