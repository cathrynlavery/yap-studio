"""Immutable render snapshots and strict version chain validation."""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_assemble import project as assembled_project  # noqa: F401
from test_assemble import refuse_unexpected_matte_work  # noqa: F401
from test_cli_captions import fake_layers  # noqa: F401
from test_taste_render import _body

from yap_studio import cli
from yap_studio.captions import CaptionWord, build_phrases
from yap_studio.errors import InputError, ValidationError
from yap_studio.feedback import versions as versions_module
from yap_studio.feedback.versions import (
    LAYERS_SCHEMA,
    VERSION_SCHEMA,
    LayerWindow,
    VersionError,
    find_version_by_media,
    load_version,
    load_versions,
    parse_layers,
    save_version,
    version_cut_words,
    version_layers,
    version_pieces,
    version_report,
)
from yap_studio.project.receipts import projection_sha256
from yap_studio.receipt import sha256_file
from yap_studio.stages import assemble as assemble_stage
from yap_studio.stages.run import layer_windows
from yap_studio.taste import (
    Taste,
    TasteApplication,
    read_taste_record,
    render_style,
    rule_id,
    write_taste_record,
)
from yap_studio.taste.render import TASTE_RECORD_SCHEMA

NOW = datetime(2026, 10, 5, 22, 0, tzinfo=UTC)


def test_pilot_media_and_renamed_copy_never_become_versions(
    assembled_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,  # noqa: F811
) -> None:
    result = assemble_stage.assemble(assembled_project, frames=(30, 89))
    pilot = assembled_project / result["out"]
    renamed = tmp_path / "review-copy.mp4"
    shutil.copyfile(pilot, renamed)
    for media in (pilot, renamed):
        with pytest.raises(VersionError, match="pilots are never saved as versions") as caught:
            save_version(assembled_project, media=media)
        assert caught.value.code == "version_refused"
        assert "frames 30-89" in str(caught.value)
    assert cli.main(["versions", "save", str(assembled_project), "--media", str(pilot)]) == 3
    monkeypatch.setattr(assemble_stage, "pilot_record", lambda media: None)
    with pytest.raises(VersionError, match="pilots are never saved as versions"):
        save_version(assembled_project, media=renamed)
    folder_copy = pilot.parent / "other.mp4"
    shutil.copyfile(assembled_project / "cut/cut.mp4", folder_copy)
    with pytest.raises(VersionError, match="pilots are never saved as versions"):
        save_version(assembled_project, media=folder_copy)
    assert not (assembled_project / "versions").exists()


def _json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _receipt_inputs(root: Path) -> dict:
    edit = json.loads((root / "edit.json").read_text())
    referenced = {item["take"] for item in edit["segments"]}
    removals = json.loads((root / "removals.json").read_text())["removals"] if (
        (root / "removals.json").exists()
    ) else []
    return {
        "edit.json": sha256_file(root / "edit.json"),
        "removals:applied": projection_sha256([
            {key: item[key] for key in ("id", "take", "start", "end")}
            for item in removals if item["status"] == "applied" and item["take"] in referenced
        ]), "take:t1": "b" * 64, "words:t1": "a" * 64,
    }


def _voice(root: Path, *, gain: float = 0.0) -> Path:
    wav_path = root / "cut" / "voice.wav"
    samples = (0.1 * 10 ** (gain / 20) * 32768 * np.sin(
        2 * np.pi * 1000 * np.arange(4 * 48000) / 48000
    )).astype("<i2")
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(samples.tobytes())
    return wav_path


def _render_media(
    media: Path, frames: int, *, amplitude: float | None = None, label: str = "render",
) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg")
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
        "color=c=black:size=32x32:rate=30",
    ]
    if amplitude is not None:
        command.extend([
            "-f", "lavfi", "-i", f"sine=frequency=1000:sample_rate=48000:duration={frames / 30}",
            "-af", f"volume={amplitude / 0.125}", "-c:a", "aac", "-b:a", "128k",
        ])
    command.extend([
        "-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-metadata", f"comment={label}", str(media),
    ])
    subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, timeout=60, check=True)


def _update_cut(
    root: Path, content: bytes | None = None, *, amplitude: float | None = None,
    label: str = "second render",
) -> None:
    media = root / "cut" / "cut.mp4"
    if content is None:
        _render_media(media, 60, amplitude=amplitude, label=label)
    else:
        media.write_bytes(content)
    _json(root / "cut" / "receipt.json", {
        "schema": "yap-studio/stage-receipt@1", "status": "verified",
        "inputs": _receipt_inputs(root),
        "outputs": [{"path": name, "sha256": sha256_file(root / name)} for name in (
            "cut/cut.mp4", "cut/spans.json", "cut/cut-words.json", "cut/voice.wav"
        ) if (root / name).exists()],
        "plan": {"fps": "30/1", "total_frames": 60, "pieces": [
            {"segment_id": "s1", "take": "t1", "out_f0": 0, "n_frames": 30,
             "src_t0": "0", "src_t1": "1", "words": ["t1:0"]},
            {"segment_id": "s2", "take": "t1", "out_f0": 30, "n_frames": 30,
             "src_t0": "2", "src_t1": "3", "words": ["t1:1"]},
        ]},
    })


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    _json(root / "project.json", {
        "schema": "yap-studio/project@1", "name": "sample-video", "style": "styles/local.json",
        "takes": [{"id": "t1", "path": "takes/t1.mp4"}],
    })
    _json(root / "styles" / "local.json", {"schema": "yap-studio/style@1", "color": "blue"})
    _json(root / "edit.json", {
        "schema": "yap-studio/edit@1", "words_sha256": {"t1": "a" * 64}, "segments": [],
        "end_hold_s": 0,
    })
    _json(root / "cut" / "cut-words.json", {
        "schema": "yap-studio/cut-words@1", "time_base": "output",
        "words_sha256": {"t1": "a" * 64}, "words": [
            {"id": "t1:0", "segment": "s1", "text": "First", "start": 0.1, "end": 0.5,
             "conf": 0.9},
            {"id": "t1:1", "segment": "s2", "text": "Second", "start": 1.1, "end": 1.5,
             "conf": None},
        ],
    })
    _json(root / "cut" / "spans.json", {
        "sources": {"t1": "../takes/t1.mp4"}, "fps": "30/1",
        "segments": [{"source": "t1", "start": "0", "end": "1"},
                     {"source": "t1", "start": "2", "end": "3"}],
    })
    _update_cut(root, label="first render")
    return root


def _layer_document() -> dict:
    return {"schema": LAYERS_SCHEMA, "time_base": "output", "layers": [
        {"id": "card1", "kind": "title", "start": 0.1, "end": 1.0, "text": "The card"},
    ]}


def test_saved_layers_accept_overlay_window(project: Path) -> None:
    layer_path = project / "layers.json"
    _json(layer_path, {"schema": LAYERS_SCHEMA, "time_base": "output", "layers": [
        {"id": "broll-photo", "kind": "overlay", "start": 0.2, "end": 1.2, "text": None},
    ]})
    expected = (LayerWindow("broll-photo", "overlay", 0.2, 1.2, None),)
    assert parse_layers(json.loads(layer_path.read_text()), source=str(layer_path)) == expected
    assert version_layers(save_version(project, layers=layer_path)) == expected


def test_save_chain_hashes_snapshots_and_immutability(project: Path) -> None:
    _json(project / "removals.json", {"schema": "yap-studio/removals@1", "removals": []})
    _json(project / "timeline" / "agent.json", {"schema": "yap-studio/timeline@1", "elements": []})
    _json(project / "timeline" / "broll.json", {"schema": "yap-studio/timeline@1", "elements": []})
    _json(project / "qa" / "report.json", {"status": "pass"})
    layer_path = project / "windows.json"
    _json(layer_path, _layer_document())
    inputs = {path.relative_to(project): path.read_bytes()
              for path in project.rglob("*") if path.is_file()}
    first = save_version(project, layers=layer_path, note="Initial review", now=lambda: NOW)
    assert first.path == project.resolve() / "versions" / "v1"
    assert first.number == 1 and first.id == "v1" and first.parent is None
    assert first.project == "sample-video" and first.created_at == "2026-10-05T22:00:00Z"
    assert first.note == "Initial review"
    assert first.media == {
        "file": "media.mp4", "source": "cut/cut.mp4",
        "sha256": sha256_file(project / "cut" / "cut.mp4"), "fps": "30/1",
        "frame_count": 60, "duration_s": 2.0,
    }
    document = json.loads((first.path / "manifest.json").read_text())
    assert document == first.to_dict() and document["schema"] == VERSION_SCHEMA
    assert "path" not in document
    assert first.media_path.read_bytes() == inputs[Path("cut/cut.mp4")]
    entries = [value for key, value in first.snapshots.items()
               if key not in ("timeline", "taste") and value is not None]
    entries.extend(first.snapshots["timeline"]["lanes"].values())
    entries.append(first.qa)
    for entry in entries:
        assert entry["sha256"] == sha256_file(first.path / entry["file"])
    for relative, original in inputs.items():
        assert (project / relative).read_bytes() == original
    assert first.snapshots["style"]["ref"] == "styles/local.json"
    assert first.snapshots["taste"] is None
    assert first.qa["status"] == "pass"
    assert version_cut_words(first).words[0].id == "t1:0"
    assert version_pieces(first) == {
        "schema": "yap-studio/version-pieces@1", "fps": "30/1", "total_frames": 60,
        "inputs": _receipt_inputs(project),
        "pieces": [
            {"segment": "s1", "take": "t1", "out_f0": 0, "n_frames": 30,
             "src_t0": "0", "src_t1": "1"},
            {"segment": "s2", "take": "t1", "out_f0": 30, "n_frames": 30,
             "src_t0": "2", "src_t1": "3"},
        ],
    }
    assert version_layers(first) == (LayerWindow("card1", "title", 0.1, 1.0, "The card"),)
    saved_bytes = {path.relative_to(first.path): path.read_bytes()
                   for path in first.path.rglob("*") if path.is_file()}
    _json(project / "styles" / "local.json", {"color": "green"})
    _json(project / "edit.json", {"schema": "yap-studio/edit@1", "words_sha256": {},
                                 "segments": [], "end_hold_s": 0.5})
    _update_cut(project)
    second = save_version(project, now=lambda: NOW + timedelta(hours=1))
    assert (second.id, second.number, second.parent, second.note) == ("v2", 2, "v1", None)
    assert second.media_path.read_bytes() == (project / "cut" / "cut.mp4").read_bytes()
    assert second.media["sha256"] != first.media["sha256"]
    assert second.snapshots["edit"]["sha256"] != first.snapshots["edit"]["sha256"]
    assert load_versions(project) == (first, second)
    assert load_version(project, "v1") == first
    for relative, original in saved_bytes.items():
        assert (first.path / relative).read_bytes() == original
    assert not list((project / "versions").glob(".*.tmp-*"))


def test_optional_snapshots_preset_style_and_hash_lookup(project: Path) -> None:
    data = json.loads((project / "project.json").read_text())
    data["style"] = "preset:basic"
    _json(project / "project.json", data)
    manifest = save_version(project, now=lambda: NOW)
    assert manifest.qa is None
    assert manifest.snapshots["style"] == {"ref": "preset:basic", "file": None, "sha256": None}
    assert all(manifest.snapshots[key] is None
               for key in ("removals", "timeline", "layers", "taste", "voice_envelope"))
    assert version_layers(manifest) is None
    assert find_version_by_media(project, manifest.media["sha256"]) == manifest
    assert find_version_by_media(project, "f" * 64) is None
    assert load_versions(project / "absent") == ()


def test_save_verify_snapshot_report_and_legacy_manifest(project: Path) -> None:
    _json(project / "cut" / "verify.json", {"status": "WARN", "findings": []})
    manifest = save_version(project)
    assert manifest.snapshots["verify"] == {
        "file": "verify.json", "sha256": sha256_file(project / "cut" / "verify.json"),
    }
    assert version_report(manifest) == {"status": "warn", "statuses": {"verify": "warn"},
                                        "subject": "render", "qa": None,
                                        "verify": {"status": "WARN", "findings": []}}
    document = manifest.to_dict()
    del document["snapshots"]["verify"]
    _json(manifest.path / "manifest.json", document)
    assert version_report(load_version(project, "v1")) is None


@pytest.mark.parametrize("status", ["PASS", None, 1])
def test_report_uses_worst_status_with_both_snapshots(project: Path, status) -> None:
    _json(project / "cut" / "verify.json", {"status": "warn"})
    _json(project / "qa" / "report.json", {"status": status})
    manifest = save_version(project)
    assert version_report(manifest) == {
        "status": "warn" if status == "PASS" else None, "subject": "render",
        "statuses": {"qa": "pass" if status == "PASS" else None, "verify": "warn"},
        "qa": {"status": status}, "verify": {"status": "warn"},
    }
    (manifest.path / "qa-report.json").write_text('{}\n')
    with pytest.raises(InputError, match="snapshot changed"):
        version_report(manifest)


@pytest.mark.parametrize("changed", ["before", "during", "missing"])
def test_verify_snapshot_checks_recorded_receipt_hash(
    project: Path, monkeypatch, changed: str,
) -> None:
    path = project / "cut" / "verify.json"
    _json(path, {"status": "pass"})
    receipt_path = project / "cut" / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["outputs"].append({"path": "cut/verify.json", "sha256": sha256_file(path)})
    _json(receipt_path, receipt)
    if changed == "before":
        _json(path, {"status": "fail"})
    elif changed == "missing":
        path.unlink()
    else:
        original = versions_module.shutil.copyfile

        def changed_copy(source, target):
            if source == path:
                _json(source, {"status": "fail"})
            return original(source, target)

        monkeypatch.setattr(versions_module.shutil, "copyfile", changed_copy)
    with pytest.raises(VersionError, match="cut/verify.json changed"):
        save_version(project)
    assert load_versions(project) == ()


def test_explicit_media_report_subject_is_cut(project: Path) -> None:
    _json(project / "cut" / "verify.json", {"status": "pass"})
    explicit = project / "render.mp4"
    shutil.copyfile(project / "cut" / "cut.mp4", explicit)
    manifest = save_version(project, media=explicit)
    assert version_report(manifest)["subject"] == "cut"


def test_refuse_unchanged_and_stale_cut(project: Path) -> None:
    save_version(project, now=lambda: NOW)
    with pytest.raises(VersionError, match="unchanged since v1"):
        save_version(project)
    (project / "cut" / "cut.mp4").write_bytes(b"replaced without receipt")
    with pytest.raises(VersionError, match="cut/cut.mp4") as error:
        save_version(project)
    assert error.value.suggested_action == "Run yap-studio cut again, then save the version."
    assert [item.id for item in load_versions(project)] == ["v1"]


@pytest.mark.parametrize("missing", [
    "receipt", "failed", "edit.json", "cut/cut-words.json", "cut/spans.json"
])
def test_requires_verified_cut_and_inputs(project: Path, missing: str) -> None:
    if missing == "receipt":
        (project / "cut" / "receipt.json").unlink()
    elif missing == "failed":
        data = json.loads((project / "cut" / "receipt.json").read_text())
        data["status"] = "failed"
        _json(project / "cut" / "receipt.json", data)
    else:
        (project / missing).unlink()
    with pytest.raises(VersionError):
        save_version(project)
    assert not (project / "versions").exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="needs ffmpeg and ffprobe")
@pytest.mark.parametrize(("frames", "rate", "accepted"), [
    (15, 30, False), (60, 24, False), (60, 30, True)
])
def test_explicit_media_matches_plan(project: Path, frames: int, rate: int, accepted: bool) -> None:
    media = project / "final.mov"
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        f"color=c=blue:size=32x32:rate={rate}", "-frames:v", str(frames),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(media),
    ], stdin=subprocess.DEVNULL, capture_output=True, timeout=30, check=True)
    if accepted:
        manifest = save_version(project, media=media, now=lambda: NOW)
        assert manifest.media["file"] == "media.mov" and manifest.media["source"] == "final.mov"
        assert manifest.media["frame_count"] == frames
        assert manifest.media_path.read_bytes() == media.read_bytes()
        assert load_version(project, "v1") == manifest
    else:
        with pytest.raises(VersionError, match="render does not match the current cut"):
            save_version(project, media=media)
        assert not (project / "versions").exists()


@pytest.mark.parametrize(("hold_s", "hold_frames"), [(0.5, 15), (None, 45)])
@pytest.mark.parametrize("frame_delta", [-1, 0, 1])
def test_explicit_media_includes_end_hold(
    project: Path, hold_s: float | None, hold_frames: int, frame_delta: int
) -> None:
    _json(project / "styles" / "local.json", {"schema_version": 1, "name": "sample"})
    edit = json.loads((project / "edit.json").read_text())
    if hold_s is None:
        del edit["end_hold_s"]
    else:
        edit["end_hold_s"] = hold_s
    _json(project / "edit.json", edit)
    _update_cut(project)
    media = project / "final.mov"
    frames = 60 + hold_frames + frame_delta
    _render_media(media, frames)

    if frame_delta:
        with pytest.raises(VersionError, match="render does not match the current cut"):
            save_version(project, media=media)
        assert not (project / "versions").exists()
    else:
        manifest = save_version(project, media=media)
        assert manifest.media["frame_count"] == frames
        assert manifest.media["duration_s"] == frames / 30
        assert manifest.media_path.read_bytes() == media.read_bytes()


def test_explicit_cut_keeps_receipt_frame_count_with_end_hold(project: Path) -> None:
    edit = json.loads((project / "edit.json").read_text())
    edit["end_hold_s"] = 0.5
    _json(project / "edit.json", edit)
    _update_cut(project)
    cut = project / "cut" / "cut.mp4"

    manifest = save_version(project, media=cut)

    assert manifest.media["source"] == "cut/cut.mp4"
    assert manifest.media["frame_count"] == 60
    assert manifest.media["duration_s"] == 2.0
    assert manifest.media_path.read_bytes() == cut.read_bytes()



def test_final_with_repeated_speech_saves_unique_caption_layers(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Segment s2 says "Ship it." again from the same source words as s1.
    words = [("t1:0", "s1", "Ship", 0.1, 0.4), ("t1:1", "s1", "it.", 0.4, 0.8),
             ("t1:0", "s2", "Ship", 1.1, 1.4), ("t1:1", "s2", "it.", 1.4, 1.8)]
    _json(project / "cut" / "cut-words.json", {
        "schema": "yap-studio/cut-words@1", "time_base": "output",
        "words_sha256": {"t1": "a" * 64}, "words": [
            {"id": word, "segment": segment, "text": text, "start": start, "end": end,
             "conf": 0.9} for word, segment, text, start, end in words
        ],
    })
    edit = json.loads((project / "edit.json").read_text())
    edit["end_hold_s"] = 0.5
    _json(project / "edit.json", edit)
    _update_cut(project)
    media = project / "final.mov"
    _render_media(media, 75)

    def phrases() -> list[dict]:
        built = build_phrases([CaptionWord(word, text, start, end)
                               for word, _, text, start, end in words],
                              {"max_words_per_phrase": 3, "max_characters_per_phrase": 30})
        return [phrase.to_dict() for phrase in built]

    built = phrases()
    assert [phrase["ids"][0] for phrase in built] == ["t1:0", "t1:0"]
    _json(project / "captions" / "phrases.json", {"phrases": built})
    # This fixture has no timeline lanes; only caption windows matter here.
    monkeypatch.setattr("yap_studio.stages.timeline.resolve_timeline",
                        lambda _root: SimpleNamespace(elements=()))
    document = layer_windows(project)
    layers = document["layers"]
    assert [layer["id"] for layer in layers] == ["caption:t1:0-t1:1",
                                                  "caption:t1:0-t1:1#2"]
    assert len({layer["id"] for layer in layers}) == 2
    assert layer_windows(project) == document
    _json(project / "layers.json", document)

    manifest = save_version(project, media=media, layers=project / "layers.json")

    assert manifest.media["frame_count"] == 75
    assert version_layers(manifest) == tuple(
        LayerWindow(layer["id"], "caption", layer["start"], layer["end"], "Ship it.")
        for layer in layers
    )

@pytest.mark.parametrize("failure", [
    "copy", "changed_media", "collision", "late_collision", "rename"
])
def test_failure_removes_temporary_folder(
    project: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    original_copy = versions_module.shutil.copyfile

    def copying(source: Path, target: Path) -> None:
        original_copy(source, target)
        if target.name == "media.mp4":
            if failure == "copy":
                raise OSError("copy interrupted")
            if failure == "changed_media":
                target.write_bytes(b"corrupt copy")
            if failure == "collision":
                collision = project / "versions" / "v1"
                collision.mkdir()
                (collision / "marker").write_bytes(b"another writer")

    monkeypatch.setattr(versions_module.shutil, "copyfile", copying)
    if failure in ("rename", "late_collision"):
        def fail_rename(source: Path, destination: Path) -> None:
            if failure == "late_collision":
                destination.mkdir()
                (destination / "marker").write_bytes(b"another writer")
            raise OSError("rename interrupted")
        monkeypatch.setattr(versions_module.os, "rename", fail_rename)
    with pytest.raises((VersionError, OSError)):
        save_version(project)
    assert not list((project / "versions").glob(".*.tmp-*"))
    if failure in ("collision", "late_collision"):
        assert (project / "versions" / "v1" / "marker").read_bytes() == b"another writer"
    else:
        assert not (project / "versions" / "v1").exists()


@pytest.mark.parametrize("bad", ["gap", "parent", "schema", "id", "keys", "snapshot", "frames"])
def test_load_rejects_invalid_manifests(project: Path, bad: str) -> None:
    save_version(project)
    _update_cut(project)
    second = save_version(project)
    path = second.path / "manifest.json"
    data = json.loads(path.read_text())
    if bad == "gap":
        second.path.rename(project / "versions" / "v3")
        path = project / "versions" / "v3" / "manifest.json"
        data.update(id="v3", parent="v2")
    elif bad == "parent":
        data["parent"] = None
    elif bad == "schema":
        data["schema"] = "unknown"
    elif bad == "id":
        data["id"] = "v8"
    elif bad == "keys":
        data["extra"] = True
    elif bad == "snapshot":
        data["snapshots"]["edit"]["file"] = "../edit.json"
    else:
        data["media"]["frame_count"] = True
    _json(path, data)
    original = path.read_bytes()
    with pytest.raises(InputError) as error:
        load_versions(project)
    assert error.value.path == str(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("source", ["../out.mp4", "renders/../out.mp4", "/out.mp4"])
def test_load_rejects_escaping_media_source(project: Path, source: str) -> None:
    manifest = save_version(project)
    path = manifest.path / "manifest.json"
    data = manifest.to_dict()
    data["media"]["source"] = source
    _json(path, data)
    original = path.read_bytes()
    with pytest.raises(InputError, match="media.source must be relative") as error:
        load_versions(project)
    assert error.value.path == str(path)
    assert path.read_bytes() == original


def test_loading_ignores_dot_folders_and_does_not_hash_media(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = save_version(project)
    (project / "versions" / ".v2.tmp-incomplete").mkdir()

    def no_hash(path: Path) -> str:
        raise AssertionError("loading must not hash files")

    monkeypatch.setattr(versions_module, "sha256_file", no_hash)
    assert load_versions(project) == (manifest,)
    assert find_version_by_media(project, manifest.media["sha256"]) == manifest
    with pytest.raises(ValidationError):
        load_version(project, "../v1")
    with pytest.raises(InputError):
        load_version(project, "v2")


@pytest.mark.parametrize("bad", ["duplicate", "end", "kind", "nan", "bool", "text", "keys"])
def test_layers_validate_before_creating_version(project: Path, bad: str) -> None:
    data = _layer_document()
    item = data["layers"][0]
    if bad == "duplicate":
        data["layers"].append(dict(item))
    elif bad == "end":
        item["end"] = item["start"]
    elif bad == "kind":
        item["kind"] = "unknown"
    elif bad == "nan":
        item["start"] = float("nan")
    elif bad == "bool":
        item["start"] = True
    elif bad == "text":
        item["text"] = 7
    else:
        item["extra"] = True
    layer_path = project / "windows.json"
    _json(layer_path, data)
    with pytest.raises(ValidationError):
        parse_layers(data, source="window-input")
    with pytest.raises(ValidationError):
        save_version(project, layers=layer_path)
    assert not (project / "versions").exists()


def test_timestamp_converts_to_utc(project: Path) -> None:
    local = NOW.astimezone(timezone(timedelta(hours=-5)))
    assert save_version(project, now=lambda: local).created_at == "2026-10-05T22:00:00Z"


def test_regression_save_refuses_changed_edit(project: Path) -> None:
    edit = json.loads((project / "edit.json").read_text())
    edit["end_hold_s"] = 0.5
    _json(project / "edit.json", edit)
    with pytest.raises(VersionError, match="edit.json changed after the verified cut") as error:
        save_version(project)
    assert error.value.suggested_action == "Run yap-studio cut again, then save the version."
    assert not (project / "versions").exists()


def test_regression_save_refuses_added_applied_removal(project: Path) -> None:
    edit = json.loads((project / "edit.json").read_text())
    edit["segments"] = [{"id": "s1", "take": "t1", "from": {"time": 0}, "to": {"time": 1}}]
    _json(project / "edit.json", edit)
    _update_cut(project)
    _json(project / "removals.json", {"schema": "yap-studio/removals@1", "removals": [
        {"id": "r1", "take": "t1", "start": 0.2, "end": 0.3,
         "kind": "manual", "status": "applied"},
    ]})
    with pytest.raises(VersionError, match="removals.json changed after the verified cut"):
        save_version(project)
    assert not (project / "versions").exists()


@pytest.mark.parametrize("filename", ["cut/cut-words.json", "cut/spans.json", "cut/voice.wav"])
def test_regression_save_refuses_changed_cut_artifact(project: Path, filename: str) -> None:
    if filename == "cut/voice.wav":
        _voice(project)
        _update_cut(project)
    (project / filename).write_bytes(b"changed output")
    with pytest.raises(VersionError, match=f"{filename} changed after the verified cut"):
        save_version(project)
    assert not (project / "versions").exists()


def test_save_names_every_changed_file_and_refuses_missing_receipt_inputs(project: Path) -> None:
    receipt_path = project / "cut" / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    del receipt["inputs"]
    _json(receipt_path, receipt)
    (project / "cut" / "spans.json").write_bytes(b"changed")
    (project / "cut" / "cut-words.json").write_bytes(b"changed")
    with pytest.raises(VersionError) as error:
        save_version(project)
    assert all(name in str(error.value) for name in (
        "inputs", "edit.json", "removals.json", "cut/spans.json", "cut/cut-words.json"
    ))
    assert not (project / "versions").exists()


@pytest.mark.parametrize("filename", ["edit.json", "spans.json", "cut-words.json", "removals.json"])
def test_regression_save_refuses_snapshot_changed_during_copy(
    project: Path, monkeypatch: pytest.MonkeyPatch, filename: str,
) -> None:
    _json(project / "removals.json", {"schema": "yap-studio/removals@1", "removals": []})
    original = versions_module.shutil.copyfile

    def copying(source: Path, target: Path) -> None:
        original(source, target)
        if target.name == filename:
            target.write_bytes(b"changed during copy")

    monkeypatch.setattr(versions_module.shutil, "copyfile", copying)
    with pytest.raises(VersionError, match="changed while it was being copied"):
        save_version(project)
    assert list((project / "versions").iterdir()) == []


def test_applied_removals_projection_matches_cut_inputs(project: Path) -> None:
    from yap_studio.feedback.versions import applied_removals_sha256
    from yap_studio.project.status import ProjectContext, cut_inputs

    (project / "takes").mkdir()
    (project / "takes" / "t1.mp4").write_bytes(b"source")
    _json(project / "ingest" / "t1" / "words.json", {
        "schema": "yap-studio/words@1", "take": "t1",
        "take_sha256": sha256_file(project / "takes" / "t1.mp4"), "time_base": "source",
        "audio_offset_s": 0, "engine": "none", "model": None, "transcript_sha256": "1" * 64,
        "words": [{"id": "t1:0", "text": "First", "start": 0.1, "end": 0.5, "conf": 1}],
    })
    _json(project / "edit.json", {
        "schema": "yap-studio/edit@1", "words_sha256": {
            "t1": sha256_file(project / "ingest" / "t1" / "words.json"),
        }, "segments": [{"id": "s1", "take": "t1", "from": {"word": "t1:0"},
                         "to": {"word": "t1:0"}}],
    })
    rows = [
        {"id": "r2", "take": "t1", "start": 0.2, "end": 0.3, "kind": "manual",
         "status": "applied", "reason": "metadata is not projected"},
        {"id": "r1", "take": "t1", "start": 0.3, "end": 0.4, "kind": "manual",
         "status": "applied"},
        {"id": "r3", "take": "t1", "start": 0.4, "end": 0.5, "kind": "manual",
         "status": "proposed"},
        {"id": "r4", "take": "t2", "start": 0.2, "end": 0.3, "kind": "manual",
         "status": "applied"},
    ]
    _json(project / "removals.json", {"schema": "yap-studio/removals@1", "removals": rows})
    expected = projection_sha256([{key: row[key] for key in ("id", "take", "start", "end")}
                                  for row in rows[:2]])
    assert applied_removals_sha256(project) == expected
    assert applied_removals_sha256(project) == (
        cut_inputs(ProjectContext(project))["removals:applied"]
    )
    (project / "removals.json").unlink()
    assert applied_removals_sha256(project) == (
        cut_inputs(ProjectContext(project))["removals:applied"]
    )
    assert applied_removals_sha256(project) == projection_sha256([])


def test_save_voice_envelope_and_cut_inputs(project: Path) -> None:
    from yap_studio.feedback.versions import version_envelope

    _update_cut(project, amplitude=0.1)
    manifest = save_version(project)
    envelope = version_envelope(manifest)
    assert 200 <= len(envelope["db"]) <= 203
    assert envelope["db"][10:190] == pytest.approx([-23.0] * 180, abs=0.3)
    assert not (project / "cut" / "voice.wav").exists()
    assert manifest.snapshots["voice_envelope"] == {
        "file": "voice-envelope.json", "sha256": sha256_file(manifest.path / "voice-envelope.json"),
    }
    assert version_pieces(manifest)["inputs"] == _receipt_inputs(project)


@pytest.mark.parametrize("key,filename", [
    ("cut_words", "cut-words.json"), ("cut_pieces", "cut-pieces.json"),
    ("layers", "layers.json"), ("voice_envelope", "voice-envelope.json"),
])
def test_snapshot_readers_refuse_tampering(project: Path, key: str, filename: str) -> None:
    from yap_studio.feedback.versions import version_envelope

    _update_cut(project, amplitude=0.1)
    layer_path = project / "windows.json"
    _json(layer_path, _layer_document())
    manifest = save_version(project, layers=layer_path)
    path = manifest.path / filename
    path.write_text(path.read_text() + " ")
    readers = {"cut_words": version_cut_words, "cut_pieces": version_pieces,
               "layers": version_layers, "voice_envelope": version_envelope}
    with pytest.raises(InputError, match="snapshot changed after it was saved"):
        readers[key](manifest)


@pytest.mark.parametrize("bad", [{"take:t1": "x"}, {"take:t1": None}, []])
def test_version_pieces_validates_source_inputs(project: Path, bad: object) -> None:
    manifest = save_version(project)
    path = manifest.path / "cut-pieces.json"
    data = json.loads(path.read_text())
    data["inputs"] = bad
    _json(path, data)
    manifest.snapshots["cut_pieces"]["sha256"] = sha256_file(path)
    with pytest.raises(InputError, match="inputs must be an object"):
        version_pieces(manifest)


def _render_with_taste(project: Path) -> tuple[Path, TasteApplication]:
    body = _body()
    snapshot = Taste(revision=7, rules=({
        "id": rule_id(body), "status": "approved", **body,
    },)).snapshot()
    application = render_style("preset:clean_minimal", snapshot=snapshot)
    media = project / "render.mp4"
    shutil.copyfile(project / "cut" / "cut.mp4", media)
    return media, application


def test_save_render_record_pins_only_revision_hash_and_applied_ids(project: Path) -> None:
    media, application = _render_with_taste(project)
    write_taste_record(media, application)
    manifest = save_version(project, media=media)
    expected = {
        "revision": 7, "sha256": application.taste_sha256,
        "applied": [item["id"] for item in application.applied],
    }
    assert manifest.snapshots["taste"] == expected
    assert load_version(project, manifest.id).snapshots["taste"] == expected
    assert "#123456" not in json.dumps(manifest.to_dict())
    assert "sentence" not in json.dumps(manifest.to_dict())


@pytest.mark.parametrize("record_source", ["adjacent", "supplied"])
def test_save_version_preserves_taste_record(
    project: Path, monkeypatch: pytest.MonkeyPatch, record_source: str,
) -> None:
    media, application = _render_with_taste(project)
    target = write_taste_record(media, application)
    record = read_taste_record(media)
    if record_source == "supplied":
        target.unlink()
    rename = versions_module.os.rename
    publications = []
    checked_media = []

    def check_record(media_path: Path) -> dict | None:
        checked_media.append(media_path)
        return read_taste_record(media_path)

    def publish(temporary: Path, destination: Path) -> None:
        assert temporary.name.startswith(".v1.tmp-")
        assert temporary / "media.mp4" in checked_media
        assert read_taste_record(temporary / "media.mp4") == record
        publications.append(destination)
        rename(temporary, destination)

    monkeypatch.setattr("yap_studio.taste.render.read_taste_record", check_record)
    monkeypatch.setattr(versions_module.os, "rename", publish)
    manifest = save_version(project, media=media,
                            taste=record if record_source == "supplied" else None)
    assert publications == [manifest.path]
    assert (manifest.path / "media.taste.json").is_file()
    assert read_taste_record(manifest.media_path) == record
    assert sha256_file(manifest.media_path) == record["media_sha256"]


@pytest.mark.parametrize("bad", ["style_sha256", "extra"])
def test_save_refuses_invalid_full_taste_record_before_publication(project: Path, bad: str) -> None:
    media, application = _render_with_taste(project)
    write_taste_record(media, application)
    record = read_taste_record(media)
    record[bad] = "invalid"
    with pytest.raises(VersionError, match="corrupt taste record"):
        save_version(project, media=media, taste=record)
    assert list((project / "versions").iterdir()) == []


@pytest.mark.parametrize("schema", ["yap-studio/taste-record@999", 7, None])
def test_save_refuses_invalid_supplied_taste_schema(project: Path, schema: object) -> None:
    media, application = _render_with_taste(project)
    write_taste_record(media, application)
    record = read_taste_record(media)
    record["schema"] = schema
    with pytest.raises(VersionError, match="corrupt taste record"):
        save_version(project, media=media, taste=record)
    assert list((project / "versions").iterdir()) == []


@pytest.mark.parametrize("with_media_hash", [False, True])
def test_save_refuses_short_pin_claiming_full_record_schema(
    project: Path, with_media_hash: bool,
) -> None:
    media, application = _render_with_taste(project)
    pin = {
        "schema": TASTE_RECORD_SCHEMA, "revision": application.taste_revision,
        "sha256": application.taste_sha256,
        "applied": [item["id"] for item in application.applied],
    }
    if with_media_hash:
        pin["media_sha256"] = sha256_file(media)
    with pytest.raises(VersionError, match="corrupt taste record"):
        save_version(project, media=media, taste=pin)
    assert list((project / "versions").iterdir()) == []


def test_explicit_media_without_record_saves_null_taste(project: Path) -> None:
    media, _ = _render_with_taste(project)
    manifest = save_version(project, media=media)
    assert manifest.snapshots["taste"] is None
    assert load_version(project, manifest.id).snapshots["taste"] is None


def test_default_cut_ignores_taste_record_and_legacy_null_loads(project: Path) -> None:
    (project / "cut" / "cut.taste.json").write_text("{invalid record", encoding="utf-8")
    manifest = save_version(project)
    assert manifest.snapshots["taste"] is None
    data = manifest.to_dict()
    del data["snapshots"]["voice_envelope"]
    del data["snapshots"]["verify"]
    _json(manifest.path / "manifest.json", data)
    assert load_version(project, manifest.id).snapshots["taste"] is None


def test_save_refuses_mismatched_taste_record(project: Path) -> None:
    media, application = _render_with_taste(project)
    write_taste_record(media, application)
    media.write_bytes(b"different render")
    with pytest.raises(VersionError, match="taste record media hash does not match"):
        save_version(project, media=media)
    assert not (project / "versions").exists()


def test_save_refuses_corrupt_taste_record(project: Path) -> None:
    media, _ = _render_with_taste(project)
    media.with_suffix(".taste.json").write_text("{}", encoding="utf-8")
    with pytest.raises(VersionError, match="corrupt taste record"):
        save_version(project, media=media)
    assert not (project / "versions").exists()


@pytest.mark.parametrize("pin_kind", [
    "application", "application_hash", "record", "summary", "summary_hash",
])
def test_save_accepts_supplied_render_pin(project: Path, pin_kind: str) -> None:
    media, application = _render_with_taste(project)
    pin = application.pin()
    if pin_kind == "record":
        pin.update(schema=TASTE_RECORD_SCHEMA, media_sha256=sha256_file(media))
    elif pin_kind in {"summary", "summary_hash"}:
        pin = {"revision": pin["revision"], "sha256": pin["sha256"],
               "applied": [item["id"] for item in pin["applied"]]}
    if pin_kind.endswith("_hash"):
        pin["media_sha256"] = sha256_file(media)
        assert "schema" not in pin
    media.with_suffix(".taste.json").write_text("{unreadable", encoding="utf-8")
    manifest = save_version(project, media=media, taste=pin)
    assert manifest.snapshots["taste"] == {
        "revision": 7, "sha256": application.taste_sha256,
        "applied": [item["id"] for item in application.applied],
    }
    if pin_kind == "record":
        assert read_taste_record(manifest.media_path) == pin
    else:
        assert not (manifest.path / "media.taste.json").exists()


def test_supplied_record_hash_must_match_saved_media(project: Path) -> None:
    media, application = _render_with_taste(project)
    with pytest.raises(VersionError, match="media hash does not match the saved media"):
        save_version(project, media=media, taste={**application.pin(), "media_sha256": "a" * 64})
    assert not (project / "versions").exists()


@pytest.mark.parametrize("taste", [
    {}, {"revision": True, "sha256": "a" * 64, "applied": []},
    {"revision": -1, "sha256": "a" * 64, "applied": []},
    {"revision": 0, "sha256": "bad", "applied": []},
    {"revision": 0, "sha256": "a" * 64, "applied": {}},
    {"revision": 0, "sha256": "a" * 64, "applied": [1]},
    {"revision": 0, "sha256": "a" * 64, "applied": [{"id": "r-123456789abc"}]},
])
def test_save_refuses_invalid_supplied_taste(project: Path, taste: dict) -> None:
    with pytest.raises(VersionError, match="Invalid render taste pin"):
        save_version(project, taste=taste)
    assert not (project / "versions").exists()


@pytest.mark.parametrize("bad", [
    [], {}, {"revision": 0, "sha256": "a" * 64, "applied": [], "extra": True},
    {"revision": True, "sha256": "a" * 64, "applied": []},
    {"revision": -1, "sha256": "a" * 64, "applied": []},
    {"revision": "1", "sha256": "a" * 64, "applied": []},
    {"revision": 0, "sha256": "z" * 64, "applied": []},
    {"revision": 0, "sha256": "a" * 64, "applied": {}},
    {"revision": 0, "sha256": "a" * 64, "applied": [True]},
    {"revision": 0, "sha256": "a" * 64, "applied": ["sentence"]},
    {"revision": 0, "sha256": "a" * 64, "applied": [{"id": "r-123456789abc"}]},
    {"revision": 0, "sha256": "a" * 64, "applied": ["r-123456789abc", "r-123456789abc"]},
])
def test_load_refuses_bad_taste_shape(project: Path, bad: object) -> None:
    manifest = save_version(project)
    data = manifest.to_dict()
    data["snapshots"]["taste"] = bad
    _json(manifest.path / "manifest.json", data)
    with pytest.raises(InputError):
        load_versions(project)


@pytest.mark.parametrize("json_output", [False, True])
def test_versions_cli_reports_saved_taste(
    project: Path, capsys: pytest.CaptureFixture[str], json_output: bool,
) -> None:
    from yap_studio.cli import main

    media, application = _render_with_taste(project)
    write_taste_record(media, application)
    arguments = ["versions", "save", str(project), "--media", str(media)]
    if json_output:
        arguments.append("--json")
    assert main(arguments) == 0
    output = capsys.readouterr().out
    if json_output:
        assert json.loads(output)["snapshots"]["taste"]["revision"] == 7
    else:
        assert output.endswith(" in versions/v1 with taste revision 7\n")
