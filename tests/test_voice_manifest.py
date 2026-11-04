"""Shared voice samples are checked against an external manifest."""

import copy
import hashlib
import json
import shlex
import shutil
import wave
from pathlib import Path

import pytest
from voice_helpers import calls, linked

from yap_studio import receipt
from yap_studio.cli import main
from yap_studio.errors import YapError
from yap_studio.voice import consent, create, delete_voice, link
from yap_studio.voice.model import STATEMENT, validate, validate_samples_manifest

AGREEMENT = (
    "This is my own voice. I agree to Yap Studio making an ElevenLabs voice from my "
    "recordings and using it to re-say words I misspoke in my videos."
)


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    folder = tmp_path / "shared recordings"
    rows = []
    for index, relative in enumerate(("audio/one.wav", "two.wav")):
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(target), "wb") as sound:
            sound.setnchannels(1)
            sound.setsampwidth(2)
            sound.setframerate(8000)
            sound.writeframes(b"\0\0" * 8000)
        rows.append({"path": relative, "take": f"Other project take {index}",
                     "start_s": 1000, "end_s": 1001.5, "duration_s": 1,
                     "ranges": [[1000, 1000.5], [1001, 1001.5]],
                     "sha256": receipt.sha256_file(target), "size_bytes": target.stat().st_size})
    target = folder / "samples.json"
    target.write_text(json.dumps({"schema": "yap-studio/voice-samples@1", "samples": rows}) + "\n")
    monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_SAMPLES", json.dumps([
        {"file_name": Path(row["path"]).name, "size_bytes": row["size_bytes"]} for row in rows
    ]))
    return target


@pytest.fixture
def local_voice(tmp_path, fake_voice, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    linked(root, monkeypatch)
    return root


def shared_manifest(folder, monkeypatch, *, count, seconds=1, extension=".wav"):
    """Write the shared recordings lane's manifest shape and provider listing."""
    (folder / "samples").mkdir(parents=True)
    rows = []
    for index in range(1, count + 1):
        target = folder / f"samples/s{index:02}.wav"
        with wave.open(str(target), "wb") as sound:
            sound.setnchannels(1)
            sound.setsampwidth(2)
            sound.setframerate(8000)
            sound.writeframes(b"\0\0" * 8000 * seconds)
        rows.append({"path": str(target.relative_to(folder)), "take": f"a{index:02}",
                     "start_s": 100, "end_s": 100 + seconds, "duration_s": seconds,
                     "ranges": [[100, 100 + seconds]], "sha256": receipt.sha256_file(target),
                     "size_bytes": target.stat().st_size})
    target = folder / "samples.json"
    target.write_text(json.dumps({"schema": "yap-studio/voice-samples@1", "samples": rows}) + "\n")
    monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_SAMPLES", json.dumps([
        {"file_name": Path(row["path"]).with_suffix(extension).name,
         "size_bytes": row["size_bytes"]} for row in rows
    ]))
    return target


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("difference", [None, "stem", "size", "duplicate"])
def test_link_compares_unique_stems_and_wav_sizes(
    local_voice, tmp_path, fake_voice, monkeypatch, external, difference,
):
    manifest_path = None
    if external:
        manifest_path = shared_manifest(tmp_path / "shared", monkeypatch, count=4)
        rows = json.loads(manifest_path.read_text())["samples"]
    else:
        rows = json.loads((local_voice / "voice/samples.json").read_text())["samples"]
    provider = [{"file_name": Path(row["path"]).with_suffix(".mp3").name,
                 "size_bytes": row["size_bytes"]} for row in rows]
    if difference == "stem":
        provider[0]["file_name"] = "different.mp3"
    elif difference == "size":
        provider[0]["size_bytes"] += 1
    elif difference == "duplicate":
        provider[-1] = {**provider[0], "file_name": Path(rows[0]["path"]).name}
    monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_SAMPLES", json.dumps(provider))
    before = (local_voice / "voice/consent.json").read_bytes()
    if difference is None:
        result = link(local_voice, name="My voice", yes=True, samples=manifest_path)
        assert result["voice"]["made"] == "linked"
    else:
        with pytest.raises(YapError) as caught:
            link(local_voice, name="My voice", yes=True, samples=manifest_path)
        assert caught.value.code == "voice_samples_differ"
        assert (local_voice / "voice/consent.json").read_bytes() == before


def test_manifest_duplicate_stems_are_refused(local_voice, manifest, monkeypatch):
    data = json.loads(manifest.read_text())
    row = data["samples"][0]
    target = manifest.parent / "other/one.WAV"
    target.parent.mkdir()
    target.write_bytes((manifest.parent / row["path"]).read_bytes())
    data["samples"][1] = {**row, "path": "other/one.WAV"}
    manifest.write_text(json.dumps(data))
    monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_SAMPLES", json.dumps([
        {"file_name": Path(row["path"]).name, "size_bytes": row["size_bytes"]}
        for row in data["samples"]
    ]))
    with pytest.raises(YapError) as caught:
        link(local_voice, name="My voice", yes=True, samples=manifest)
    assert caught.value.code == "voice_samples_invalid"


@pytest.mark.parametrize("count", [25, 26])
def test_link_manifest_count(local_voice, tmp_path, monkeypatch, count):
    manifest_path = shared_manifest(tmp_path / "shared", monkeypatch, count=count)
    if count > 25:
        with pytest.raises(YapError) as caught:
            link(local_voice, name="My voice", yes=True, samples=manifest_path)
        assert caught.value.code == "voice_samples_invalid"
        return
    result = link(local_voice, name="My voice", yes=True, samples=manifest_path)
    assert result["voice"]["samples_manifest"] == {
        "sha256": receipt.sha256_file(manifest_path), "files": count,
    }
    saved = json.loads((local_voice / "voice/consent.json").read_text())
    assert validate(local_voice, "consent", saved) == result


def test_manifest_refuses_more_than_25_files(manifest):
    data = json.loads(manifest.read_text())
    data["samples"] = [{**data["samples"][0], "path": f"samples/s{index}.wav"}
                       for index in range(26)]
    with pytest.raises(ValueError):
        validate_samples_manifest(data)


@pytest.mark.parametrize("count", [25, 26])
@pytest.mark.parametrize("dry_run", [False, True])
def test_create_manifest_keeps_instant_clone_count_limit(
    local_voice, tmp_path, fake_voice, monkeypatch, count, dry_run,
):
    manifest_path = shared_manifest(tmp_path / "instant", monkeypatch, count=count)
    monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_APPEARS_AFTER_ADD", "1")
    before = calls(fake_voice)
    if count > 25:
        with pytest.raises(YapError) as caught:
            create(local_voice, yes=True, dry_run=dry_run, samples=manifest_path)
        assert caught.value.code == "voice_samples_invalid"
        assert calls(fake_voice) == before
    else:
        result = create(local_voice, yes=True, dry_run=dry_run, samples=manifest_path)
        if dry_run:
            assert len(result["samples"]) == 25
            assert not any(argv[:2] == ["voices", "add"] for argv in calls(fake_voice))
        else:
            assert result["voice"]["samples_manifest"]["files"] == 25
            upload, = [argv for argv in calls(fake_voice) if argv[:2] == ["voices", "add"]]
            assert len(json.loads(upload[upload.index("--files") + 1])) == 25


def test_link_nine_file_shared_lane_manifest(tmp_path, fake_voice, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    consent(root, yes=True, voice_name="My voice")
    manifest_path = shared_manifest(tmp_path / "shared", monkeypatch, count=9, seconds=22,
                                    extension=".mp3")
    data = json.loads(manifest_path.read_text())
    assert [row["path"] for row in data["samples"]] == [
        f"samples/s{index:02}.wav" for index in range(1, 10)
    ]
    assert data["samples"][6]["take"] == "a07"
    assert all(row["duration_s"] == 22 and row["ranges"] == [[100, 122]]
               for row in data["samples"])
    assert validate_samples_manifest(data) == data
    result = link(root, name="My voice", yes=True, samples=manifest_path)
    assert result["samples"] == []
    assert result["voice"]["samples_manifest"] == {
        "sha256": receipt.sha256_file(manifest_path), "files": 9,
    }


def test_consent_statement_is_the_approved_text():
    assert STATEMENT == AGREEMENT


@pytest.mark.parametrize("ranges", [True, False])
def test_link_manifest_records_byte_hash(local_voice, manifest, ranges):
    if not ranges:
        data = json.loads(manifest.read_text())
        for row in data["samples"]:
            row.pop("ranges")
            row["end_s"] = row["start_s"] + row["duration_s"]
        manifest.write_text(json.dumps(data))
    data = link(local_voice, name="My voice", yes=True, samples=manifest)
    assert data["voice"]["samples_manifest"] == {
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "files": 2,
    }
    saved = json.loads((local_voice / "voice/consent.json").read_text())
    assert saved == data
    assert str(manifest) not in json.dumps(saved)
    assert "fake-personal-voice-secret" not in json.dumps(saved)
    assert consent(local_voice, yes=True)["voice"] == data["voice"]


@pytest.mark.parametrize("field,value", [("file_name", "wrong.wav"), ("size_bytes", 1)])
def test_link_manifest_refuses_provider_differences(
    local_voice, manifest, fake_voice, monkeypatch, field, value,
):
    rows = json.loads(manifest.read_text())["samples"]
    provider = [{"file_name": Path(row["path"]).name, "size_bytes": row["size_bytes"]}
                for row in rows]
    provider[0][field] = value
    monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_SAMPLES", json.dumps(provider))
    before = (local_voice / "voice/consent.json").read_bytes()
    with pytest.raises(YapError) as caught:
        link(local_voice, name="My voice", yes=True, samples=manifest)
    assert caught.value.code == "voice_samples_differ"
    assert str(manifest) in caught.value.suggested_action
    assert "DIR/voice/samples/" not in caught.value.suggested_action
    assert (local_voice / "voice/consent.json").read_bytes() == before


@pytest.mark.parametrize("problem", [
    "hash", "size", "outside", "absolute", "duplicate", "missing", "not_wav", "not_audio",
    "empty", "too_many", "duration", "take", "ranges", "range_duration", "start", "schema",
])
def test_bad_manifest_is_refused_before_upload(
    local_voice, manifest, fake_voice, monkeypatch, problem,
):
    data = json.loads(manifest.read_text())
    row = data["samples"][0]
    if problem == "hash":
        row["sha256"] = "0" * 64
    elif problem == "size":
        row["size_bytes"] += 1
    elif problem == "outside":
        row["path"] = "../outside.wav"
    elif problem == "absolute":
        row["path"] = str((manifest.parent / row["path"]).resolve())
    elif problem == "duplicate":
        row["path"] = "other/two.wav"
        target = manifest.parent / row["path"]
        target.parent.mkdir()
        target.write_bytes((manifest.parent / "two.wav").read_bytes())
    elif problem == "missing":
        row["path"] = "missing.wav"
    elif problem in {"not_wav", "not_audio"}:
        row["path"] = "sample.mp3" if problem == "not_wav" else "sample.wav"
        target = manifest.parent / row["path"]
        target.write_bytes(b"not a WAV file")
        row.update(sha256=receipt.sha256_file(target), size_bytes=target.stat().st_size)
    elif problem == "empty":
        data["samples"] = []
    elif problem == "too_many":
        data["samples"] = data["samples"] * 13
    elif problem == "duration":
        row["duration_s"] = 0
    elif problem == "take":
        row["take"] = " "
    elif problem == "ranges":
        row["ranges"] = [[1000, 1001], [1000.5, 1001.5]]
    elif problem == "range_duration":
        row["duration_s"] = 3
    elif problem == "start":
        row["start_s"] = -1
    else:
        data["schema"] = "wrong"
    manifest.write_text(json.dumps(data))
    monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_DUPLICATES", "0")
    with pytest.raises(YapError) as caught:
        create(local_voice, yes=True, samples=manifest)
    assert caught.value.code == "voice_samples_invalid"
    assert not any(argv[:2] == ["voices", "add"] for argv in calls(fake_voice))


def test_manifest_symlink_cannot_leave_its_folder(local_voice, manifest, tmp_path):
    data = json.loads(manifest.read_text())
    outside = tmp_path / "outside.wav"
    outside.write_bytes((manifest.parent / "audio/one.wav").read_bytes())
    (manifest.parent / "escape.wav").symlink_to(outside)
    data["samples"][0]["path"] = "escape.wav"
    manifest.write_text(json.dumps(data))
    with pytest.raises(YapError) as caught:
        link(local_voice, name="My voice", yes=True, samples=manifest)
    assert caught.value.code == "voice_samples_invalid"


@pytest.mark.parametrize("count", [1, 25])
def test_manifest_uses_its_own_count_and_duration_limits(
    local_voice, manifest, monkeypatch, count,
):
    data = json.loads(manifest.read_text())
    template = data["samples"][0]
    if count == 1:
        # A shared voice is not limited to the project's 150 second export budget.
        target = manifest.parent / template["path"]
        with wave.open(str(target), "wb") as sound:
            sound.setnchannels(1)
            sound.setsampwidth(2)
            sound.setframerate(8000)
            sound.writeframes(b"\0\0" * 8000 * 200)
        template.update(duration_s=200, end_s=1200, ranges=[[1000, 1200]],
                        sha256=receipt.sha256_file(target), size_bytes=target.stat().st_size)
    data["samples"] = []
    for index in range(count):
        row = {**template, "path": f"sample{index}.wav"}
        (manifest.parent / row["path"]).write_bytes(
            (manifest.parent / template["path"]).read_bytes(),
        )
        data["samples"].append(row)
    manifest.write_text(json.dumps(data))
    monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_SAMPLES", json.dumps([
        {"file_name": row["path"], "size_bytes": row["size_bytes"]} for row in data["samples"]
    ]))
    assert link(local_voice, name="My voice", yes=True, samples=manifest)["voice"][
        "samples_manifest"]["files"] == count


def test_create_manifest_uploads_only_its_files(local_voice, manifest, fake_voice, monkeypatch):
    preview = create(local_voice, dry_run=True, samples=manifest)
    files = [str((manifest.parent / row["path"]).resolve())
             for row in json.loads(manifest.read_text())["samples"]]
    assert preview["samples"] == files
    assert not any(argv[:2] == ["voices", "add"] for argv in calls(fake_voice))
    monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_APPEARS_AFTER_ADD", "1")
    result = create(local_voice, yes=True, samples=manifest)
    uploaded, = [argv for argv in calls(fake_voice) if argv[:2] == ["voices", "add"]]
    assert json.loads(uploaded[uploaded.index("--files") + 1]) == files
    assert result["voice"]["samples_manifest"]["sha256"] == receipt.sha256_file(manifest)


def test_create_manifest_manual_steps_name_its_files(
    local_voice, manifest, fake_voice, monkeypatch,
):
    monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_DUPLICATES", "0")
    monkeypatch.setenv("FAKE_ELEVENLABS_ADD_FILES_ERROR", "1")
    with pytest.raises(YapError) as caught:
        create(local_voice, yes=True, samples=manifest)
    assert caught.value.code == "voice_cli_cannot_upload"
    for row in json.loads(manifest.read_text())["samples"]:
        assert str(manifest.parent / row["path"]) in caught.value.suggested_action
    command = caught.value.suggested_action.rsplit("Then run ", 1)[1].removesuffix(".")
    assert shlex.split(command) == [
        "yap-studio", "voice", "link", str(local_voice), "--name", "My voice", "--yes",
        "--samples", str(manifest),
    ]
    assert "DIR/voice/samples/" not in caught.value.suggested_action


@pytest.mark.parametrize("command", ["link", "create"])
def test_consent_without_project_samples_needs_a_manifest(tmp_path, fake_voice, command):
    root = tmp_path / "empty project"
    root.mkdir()
    data = consent(root, yes=True)
    assert data["statement"] == AGREEMENT and data["samples"] == []
    with pytest.raises(YapError) as caught:
        if command == "link":
            link(root, name="My voice", yes=True)
        else:
            create(root, yes=True)
    assert str(caught.value) == (
        "Export this project's samples first with yap-studio voice samples DIR, or pass --samples."
    )


def test_link_manifest_after_consent_without_project_samples(tmp_path, manifest, fake_voice):
    root = tmp_path / "empty project"
    root.mkdir()
    consent(root, yes=True)
    result = link(root, name="My voice", yes=True, samples=manifest)
    assert result["samples"] == []
    assert result["voice"]["samples_manifest"] == {
        "sha256": receipt.sha256_file(manifest), "files": 2,
    }


def test_project_samples_can_be_exported_after_consent(local_voice, tmp_path):
    root = tmp_path / "empty project"
    root.mkdir()
    consent(root, yes=True)
    shutil.copytree(local_voice / "voice/samples", root / "voice/samples")
    shutil.copyfile(local_voice / "voice/samples.json", root / "voice/samples.json")
    result = link(root, name="My voice", yes=True)
    assert len(result["samples"]) == 4
    assert result["voice"].get("samples_manifest") is None


@pytest.mark.parametrize("withdrawn", [False, True])
@pytest.mark.parametrize("yes", [False, True])
def test_delete_manifest_voice_refuses_without_a_remote_call(
    local_voice, manifest, fake_voice, withdrawn, yes,
):
    link(local_voice, name="My voice", yes=True, samples=manifest)
    if withdrawn:
        consent(local_voice, withdraw=True)
    before = calls(fake_voice)
    with pytest.raises(YapError) as caught:
        delete_voice(local_voice, yes=yes)
    assert caught.value.code == "voice_shared"
    assert str(caught.value) == (
        "This voice was made from recordings across your videos, so Yap Studio will not delete it. "
        "Delete it on the ElevenLabs site if you mean to."
    )
    assert calls(fake_voice) == before


@pytest.mark.parametrize("value", [
    {}, {"sha256": "bad", "files": 1}, {"sha256": "a" * 64, "files": 0},
    {"sha256": "a" * 64, "files": 26}, {"sha256": "a" * 64, "files": True},
    {"sha256": "a" * 64, "files": 1, "path": "samples.json"},
])
def test_consent_rejects_bad_manifest_metadata(local_voice, value):
    data = json.loads((local_voice / "voice/consent.json").read_text())
    data["voice"]["samples_manifest"] = copy.deepcopy(value)
    with pytest.raises(YapError) as caught:
        validate(local_voice, "consent", data)
    assert caught.value.code == "voice_record_invalid"


@pytest.mark.parametrize("command", ["link", "create"])
def test_cli_accepts_samples_option(local_voice, manifest, capsys, command):
    arguments = ["--name", "My voice", "--yes"] if command == "link" else ["--dry-run"]
    assert main(["voice", command, str(local_voice), *arguments,
                 "--samples", str(manifest), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    if command == "link":
        assert result["voice"]["samples_manifest"]["files"] == 2
    else:
        assert len(result["samples"]) == 2
