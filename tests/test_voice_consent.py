"""Consent, sample proof and private voice identity."""

import json
import os
import shlex

import pytest
from test_cut import write_json
from voice_helpers import calls, linked

from yap_studio.errors import YapError
from yap_studio.voice import consent, create, delete_voice, link


def test_consent_required(voice_project, fake_voice):
    for operation in (
        lambda: consent(voice_project),
        lambda: create(voice_project, yes=True),
        lambda: link(voice_project, name="My voice", yes=True),
    ):
        with pytest.raises(YapError):
            operation()


@pytest.mark.parametrize("during", ["account_state", "personal_voices"])
def test_create_rechecks_consent_before_upload(voice_project, fake_voice, monkeypatch, during):
    from yap_studio import elevenlabs

    linked(voice_project, monkeypatch)
    monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_DUPLICATES", "0")
    lookup = getattr(elevenlabs, during)

    def withdraw(*args):
        result = lookup(*args)
        consent(voice_project, withdraw=True)
        return result

    monkeypatch.setattr(elevenlabs, during, withdraw)
    with pytest.raises(YapError):
        create(voice_project, yes=True)
    assert not any(argv[:2] == ["voices", "add"] for argv in calls(fake_voice))
    assert json.loads((voice_project / "voice/consent.json").read_text())["status"] == "withdrawn"


@pytest.mark.parametrize("during", ["account_state", "personal_voices"])
@pytest.mark.parametrize("replacement", ["same_size", "larger"])
def test_create_rechecks_sample_bytes_before_upload(
    voice_project, fake_voice, monkeypatch, during, replacement,
):
    from yap_studio import elevenlabs

    record = linked(voice_project, monkeypatch)
    before = (voice_project / "voice/consent.json").read_bytes()
    sample = voice_project / record["samples"][0]["path"]
    monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_DUPLICATES", "0")
    lookup = getattr(elevenlabs, during)

    def replace(*args):
        result = lookup(*args)
        content = sample.read_bytes()
        sample.write_bytes(content[:-1] + b"\x01" if replacement == "same_size"
                           else content + b"\x01\x00")
        return result

    monkeypatch.setattr(elevenlabs, during, replace)
    with pytest.raises(YapError) as caught:
        create(voice_project, yes=True)
    assert not any(argv[:2] == ["voices", "add"] for argv in calls(fake_voice))
    assert "Voice samples changed" in str(caught.value)
    assert (voice_project / "voice/consent.json").read_bytes() == before


def test_withdraw_without_original_take(tmp_path, fake_voice, monkeypatch):
    import synth
    from test_cut import make_project

    from yap_studio.voice.model import read

    source = synth.voiced_source(tmp_path / "take.mov", [(0.2, 0.8)], seconds=60)
    root = make_project(tmp_path / "project", [source], word_rows=[[("morning", 0.2, 0.8)]])
    linked(root, monkeypatch)
    target = root / "voice/consent.json"
    record = json.loads(target.read_text())
    for row in record["samples"]:
        row.update(ranges=[[row["start_s"], row["end_s"]]], duration_s=15)
    write_json(target, record)
    assert read(root, "consent")["status"] == "given"
    (root / "takes/t1.mov").rename(root / "offline.mov")
    assert consent(root, withdraw=True)["status"] == "withdrawn"
    assert read(root, "consent")["status"] == "withdrawn"
    assert all((root / row["path"]).is_file() for row in record["samples"])


def test_local_upload_recovery_command_is_shell_quoted(tmp_path, fake_voice, monkeypatch):
    root = tmp_path / "project with spaces"
    root.mkdir()
    linked(root, monkeypatch)
    name = 'My "voice" isn\'t $simple'
    consent(root, yes=True, voice_name=name)
    monkeypatch.setenv("FAKE_ELEVENLABS_ADD_FILES_ERROR", "1")
    monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_DUPLICATES", "0")
    with pytest.raises(YapError) as caught:
        create(root, yes=True)
    assert caught.value.code == "voice_cli_cannot_upload"
    command = caught.value.suggested_action.split("Then run ")[1].removesuffix(".")
    assert shlex.split(command) == [
        "yap-studio", "voice", "link", str(root), "--name", name, "--yes",
    ]


@pytest.mark.parametrize(
    "setting,value",
    [
        ("FAKE_ELEVENLABS_VOICE_DUPLICATES", "2"),
        ("FAKE_ELEVENLABS_VOICE_CATEGORY", "premade"),
        ("FAKE_ELEVENLABS_VOICE_SAMPLES", "[]"),
    ],
)
def test_link_proof(voice_project, fake_voice, monkeypatch, setting, value):
    linked(voice_project, monkeypatch)
    monkeypatch.setenv(setting, value)
    with pytest.raises(YapError):
        link(voice_project, name="My voice", yes=True)


def test_real_cli_upload_failure(voice_project, fake_voice, monkeypatch):
    linked(voice_project, monkeypatch)
    monkeypatch.setenv("FAKE_ELEVENLABS_ADD_FILES_ERROR", "1")
    monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_DUPLICATES", "0")
    with pytest.raises(YapError) as caught:
        create(voice_project, yes=True)
    assert caught.value.code == "voice_cli_cannot_upload"
    assert "Instant Voice Clone" in caught.value.suggested_action
    assert "voice link" in caught.value.suggested_action
    assert "fake-personal-voice-secret" not in json.dumps(caught.value.to_dict())


@pytest.mark.parametrize("folder", [False, True])
@pytest.mark.parametrize("command", ["create", "link"])
def test_project_upload_steps_name_current_files(
    voice_project, fake_voice, monkeypatch, folder, command,
):
    linked(voice_project, monkeypatch)
    record = json.loads((voice_project / "voice/samples.json").read_text())
    if folder:
        generation = voice_project / "voice/samples/g0123456789abcdef"
        generation.mkdir()
        for row in record["samples"]:
            old = voice_project / row["path"]
            target = generation / old.name
            old.rename(target)
            row["path"] = str(target.relative_to(voice_project))
        write_json(voice_project / "voice/samples.json", record)
        consent(voice_project, yes=True)
    if command == "create":
        monkeypatch.setenv("FAKE_ELEVENLABS_ADD_FILES_ERROR", "1")
        monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_DUPLICATES", "0")
    else:
        monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_SAMPLES", "[]")
    with pytest.raises(YapError) as caught:
        if command == "create":
            create(voice_project, yes=True)
        else:
            link(voice_project, name="My voice", yes=True)
    assert caught.value.code == (
        "voice_cli_cannot_upload" if command == "create" else "voice_samples_differ"
    )
    for row in record["samples"]:
        assert str(voice_project / row["path"]) in caught.value.suggested_action
    assert "DIR/voice/samples/" not in caught.value.suggested_action


def test_withdraw_delete(voice_project, fake_voice, monkeypatch):
    linked(voice_project, monkeypatch)
    assert consent(voice_project, withdraw=True)["status"] == "withdrawn"
    assert delete_voice(voice_project, yes=True)["status"] == "deleted"


@pytest.mark.parametrize("field,value", [("file_name", "wrong.wav"), ("size_bytes", 1)])
def test_link_sample_details(voice_project, fake_voice, monkeypatch, field, value):
    linked(voice_project, monkeypatch)
    data = json.loads(os.environ["FAKE_ELEVENLABS_VOICE_SAMPLES"])
    data[0][field] = value
    monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_SAMPLES", json.dumps(data))
    with pytest.raises(YapError) as caught:
        link(voice_project, name="My voice", yes=True)
    assert caught.value.code == "voice_samples_differ"


@pytest.mark.parametrize("state", ["free", "signed_out", "missing_cli"])
def test_account_disclosure(voice_project, fake_voice, monkeypatch, state):
    from yap_studio import elevenlabs
    from yap_studio.affiliate import ELEVENLABS_AFFILIATE_URL

    linked(voice_project, monkeypatch)
    monkeypatch.setattr(elevenlabs, "account_state", lambda: state)
    with pytest.raises(YapError) as caught:
        create(voice_project, yes=True)
    assert ELEVENLABS_AFFILIATE_URL in "\n".join(caught.value.notices)
    assert "affiliate link" in "\n".join(caught.value.notices)
    assert "voice fixes" in caught.value.notices[0].lower()


def test_samples_from_kept_speech(tmp_path):
    import synth
    from test_cut import make_project, time_segment, write_edit, write_json

    from yap_studio import ffmpeg
    from yap_studio.voice import samples

    source, spans = synth.speech_source(
        tmp_path / "speech.mov",
        [
            "These are my own recordings of a quiet morning in the garden "
            "with bright light and water."
        ]
        * 12,
        size="160x90",
        noise_dbfs=-60,
    )
    root = make_project(
        tmp_path / "long",
        [source],
        word_rows=[[(word.text, word.start, word.end) for word in spans]],
    )
    project = json.loads((root / "project.json").read_text())
    project["takes"][0]["voiceover"] = True
    write_json(root / "project.json", project)
    end = ffmpeg.probe(source).video_duration_s
    write_edit(root, [time_segment("s1", "t1", 5, end)])
    data = samples(root)
    assert 4 <= len(data["samples"]) <= 6
    assert 45 <= sum(row["end_s"] - row["start_s"] for row in data["samples"]) <= 150
    for row in data["samples"]:
        assert row["start_s"] >= 5 and row["end_s"] <= end
        assert 15 <= row["end_s"] - row["start_s"] <= 30
        info = ffmpeg.probe(root / row["path"])
        assert info.sample_rate == 44100 and info.channels == 1 and info.audio_codec == "pcm_s16le"
