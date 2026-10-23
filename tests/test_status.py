"""Input snapshots, stage journals, crash recovery, and agent-facing status."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_cli_captions import fake_layers as fake_layers
from test_feedback_batch import _cut as _feedback_cut
from test_feedback_batch import _export as _feedback_export
from test_feedback_batch import feedback_project as feedback_project
from test_feedback_replies import COMMENTS as _reply_comments
from test_feedback_replies import _round as _reply_round

from yap_studio import ffmpeg, receipt
from yap_studio.errors import InputError, OutputPathError, ProcessingError, YapError
from yap_studio.feedback.batch import build_batch, set_resulting_version
from yap_studio.feedback.release import rollback
from yap_studio.feedback.replies import write_reply
from yap_studio.feedback.versions import save_version
from yap_studio.project import receipts
from yap_studio.project import status as project_status
from yap_studio.project.lock import ConflictError
from yap_studio.project.receipts import IdentityCache, StageRun, projection_sha256, read_receipt
from yap_studio.project.status import (
    STAGES,
    ProjectContext,
    StageSpec,
    TakeChangedError,
    _edge_engine_identity,
    cut_inputs,
    fillers_inputs,
    ingest_inputs,
    register,
    status,
)
from yap_studio.review.export import series_export
from yap_studio.review.launch import version_video
from yap_studio.review.model import MediaFacts
from yap_studio.review.store import ReviewStore
from yap_studio.transcript import backends

# A batch from an unsaved cut cannot be applied, so the cut is reviewed as a saved version.
CUT_REVIEW = ("review the cut: yap-studio versions save DIR, "
              "then yap-studio review --versions DIR "
              "(reviews of loose files cannot take replies)")

REPLY_NEED = ('reply to 3 review notes in b1 before the next review: '
              'yap-studio feedback replies DIR --batch bN, then yap-studio feedback reply DIR '
              '--batch bN --comment vK:cN --version vM --status changed|kept|declined '
              '--text "..."')
UNSENT_NEED = ('send 1 review note from earlier versions that are in no feedback batch: '
               'open the current version with yap-studio review --versions DIR, '
               'save Export JSON, then yap-studio feedback build DIR EXPORT '
               '(open earlier notes are carried into the new batch)')


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _stage(result: dict[str, Any], name: str) -> dict[str, Any]:
    """Look up a stage independently of registration order."""
    return next(stage for stage in result["stages"] if stage["name"] == name)


@pytest.mark.parametrize("memo", ["shared", "project"])
@pytest.mark.parametrize("engine", ["parakeet-mlx", "faster-whisper"])
def test_status_lists_cached_transcript_without_hashing(project, monkeypatch, capsys, memo, engine):
    from test_transcript_shared import CountingBackend

    from yap_studio.cli import main
    from yap_studio.transcript import Transcript, shared

    backend = CountingBackend()
    backend.name = engine
    monkeypatch.setattr(project_status, "resolve_backend", lambda *_: backend)
    assert "transcripts" not in status(project)
    take = project / "takes/first.mov"
    digest = (shared.take_digest(take) if memo == "shared"
              else IdentityCache(project).sha256(take))
    shared.store(backend, shared.Request(digest, 0, take), Transcript([], language="en"))
    monkeypatch.setattr(receipt, "sha256_file", lambda _: pytest.fail("status hashed a take"))
    assert status(project)["transcripts"] == [
        {"take": "t1", "engine": engine, "model": backend.model},
    ]
    assert main(["status", str(project)]) == 0
    assert f"transcript: t1 cached by {engine}; ingest reuses it" in capsys.readouterr().out
    take.write_bytes(b"changed")
    assert "transcripts" not in status(project)
    assert main(["status", str(project)]) == 0
    assert "transcript:" not in capsys.readouterr().out


def test_known_identity_is_read_only_and_ignores_stale_or_corrupt_entries(project, monkeypatch):
    take = project / "takes/first.mov"
    identity = IdentityCache(project)
    assert identity.known_sha256(take) is None
    digest = identity.sha256(take)
    before = identity.path.read_bytes()
    monkeypatch.setattr(receipt, "sha256_file", lambda _: pytest.fail("unexpected hash"))
    monkeypatch.setattr(receipts, "project_lock", lambda _: pytest.fail("unexpected lock"))
    monkeypatch.setattr(receipts, "_write_json", lambda *_: pytest.fail("unexpected write"))
    assert identity.known_entry(take) == json.loads(before)[str(take.resolve())]
    assert identity.known_sha256(take) == digest
    assert identity.path.read_bytes() == before
    take.write_bytes(b"changed")
    assert identity.known_entry(take) is None
    assert identity.known_sha256(take) is None
    identity.path.write_text("{")
    assert identity.known_entry(take) is None
    assert identity.known_sha256(take) is None


@pytest.mark.parametrize("error_type", [YapError, OSError, ValueError, TypeError,
                                       KeyError, AttributeError])
def test_status_ignores_transcript_cache_errors(project, monkeypatch, error_type):
    from test_transcript_shared import CountingBackend

    from yap_studio.transcript import shared

    shared.take_digest(project / "takes/first.mov")
    monkeypatch.setattr(project_status, "resolve_backend", lambda *_: CountingBackend())

    def fail(*_args, **_kwargs):
        raise error_type("synthetic cache failure")

    monkeypatch.setattr(shared, "lookup", fail)
    assert "transcripts" not in status(project)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    take = tmp_path / "takes" / "first.mov"
    take.parent.mkdir()
    take.write_bytes(b"read-only source bytes")
    _write(tmp_path / "project.json", {
        "schema": "yap-studio/project@1", "name": "story", "style": "preset:clean_minimal",
        "takes": [{"id": "t1", "path": "takes/first.mov", "crop_x": 0.5}],
    })
    return tmp_path


def _finish_ingest(project: Path, *, audio_fingerprint: bool = False) -> dict:
    context = ProjectContext(project)
    inputs = ingest_inputs(context)
    staged = project / "staged-words.json"
    _write(staged, {
        "schema": "yap-studio/words@1", "take": "t1", "take_sha256": inputs["take:t1"],
        "time_base": "source", "audio_offset_s": 0, "engine": "none", "model": None,
        "transcript_sha256": "1" * 64, "words": [
            {"id": "t1:0", "text": "First", "start": 0, "end": 0.2, "conf": 1},
            {"id": "t1:1", "text": "word", "start": 0.3, "end": 0.5, "conf": 1},
        ],
    })
    with StageRun(project, "ingest") as run:
        for name, digest in inputs.items():
            run.record_inputs(name, digest)
        run.replace_output(project / "ingest/t1/words.json", staged)
        if audio_fingerprint:
            take = context.project.takes[0]
            staged = project / "staged-ingest.json"
            _write(staged, {"schema": "yap-studio/ingest@1", "takes": [{
                "id": take.id, "path": take.path, "sha256": inputs[f"take:{take.id}"],
                "requested_audio_stream": None, "audio_stream": 0, "audio_offset_s": 0,
                "audio_sha256": receipts.audio_stream_sha256(context.take_path(take.id), 0),
            }]})
            run.replace_output(project / "ingest/ingest.json", staged)
        result = run.finish(asserts=[{"id": "source", "pass": True}])
    assert not staged.exists()
    return result


def _edit(project: Path) -> None:
    _write(project / "edit.json", {
        "schema": "yap-studio/edit@1", "words_sha256": {
            "t1": receipt.sha256_file(project / "ingest/t1/words.json"),
        },
        "segments": [{"id": "s1", "take": "t1", "from": {"word": "t1:0"},
                      "to": {"word": "t1:1"}}], "end_hold_s": None,
    })


def _finish_cut(project: Path) -> dict:
    inputs = cut_inputs(ProjectContext(project))
    staged = project / "staged.bin"
    staged.write_bytes(b"rendered cut")
    with StageRun(project, "cut") as run:
        for name, digest in inputs.items():
            run.record_inputs(name, digest)
        run.replace_output(project / "cut/cut.mp4", staged)
        result = run.finish()
    assert not staged.exists()
    return result


@pytest.mark.parametrize("screen", ["green", "blue", None])
def test_cached_cutout_status_and_cli(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    screen: str | None,
) -> None:
    from yap_studio.cli import main

    _finish_ingest(project)
    _edit(project)
    _finish_cut(project)
    assert "cutout" not in status(project)
    assert main(["status", str(project)]) == 0
    assert "cutout:" not in capsys.readouterr().out
    cut = project / "cut/cut.mp4"
    digest = ProjectContext(project).identity.sha256(cut)
    backend = ({"name": "chroma-key", "screen": screen, "key": "#30b050",
                "border_share": 0.714} if screen else {"name": "vision-person-segmentation"})
    folder = project / ".yap/cache/matte" / digest
    _write(folder / "matte.json", {"status": "measured", "backend": backend,
                                    "matte": {"file": "matte.mkv"}})
    (folder / "matte.mkv").write_bytes(b"synthetic matte")
    # The current cut has an identity cache entry before status is called.
    original = receipts.receipt.sha256_file

    def no_cut_hash(source: Path) -> str:
        assert source.resolve() != cut.resolve(), "status rehashed the cut"
        return original(source)

    monkeypatch.setattr(receipts.receipt, "sha256_file", no_cut_hash)
    expected = f"{screen} screen key (71% of the edges)" if screen else "person detection"
    assert status(project)["cutout"] == expected
    assert main(["status", str(project)]) == 0
    assert f"cutout: {expected}\n" in capsys.readouterr().out
    assert main(["status", str(project), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["cutout"] == expected


@pytest.mark.parametrize("media_state", ["missing", "empty", "directory"])
def test_cached_cutout_requires_nonempty_media(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    media_state: str,
) -> None:
    from yap_studio.cli import main

    _finish_ingest(project)
    _edit(project)
    _finish_cut(project)
    cut = project / "cut/cut.mp4"
    folder = project / ".yap/cache/matte" / ProjectContext(project).identity.sha256(cut)
    _write(folder / "matte.json", {"status": "measured", "matte": {"file": "mask.mkv"},
                                    "backend": {"name": "vision-person-segmentation"}})
    media = folder / "mask.mkv"
    media.write_bytes(b"stat only, not decodable")
    original = receipts.receipt.sha256_file

    def no_media_hash(source: Path) -> str:
        assert source.resolve() not in (cut.resolve(), media.resolve()), "status hashed media"
        return original(source)

    monkeypatch.setattr(receipts.receipt, "sha256_file", no_media_hash)
    assert status(project)["cutout"] == "person detection"
    media.unlink()
    if media_state == "empty":
        media.touch()
    elif media_state == "directory":
        media.mkdir()
    assert "cutout" not in status(project)
    assert main(["status", str(project)]) == 0
    assert "cutout:" not in capsys.readouterr().out
    assert main(["status", str(project), "--json"]) == 0
    assert "cutout" not in json.loads(capsys.readouterr().out)


def _finish_fillers(project: Path) -> None:
    """Publish an empty filler stage through its root-output ownership exception."""
    data = {"schema": "yap-studio/removals@1", "removals": []}
    inputs = fillers_inputs(ProjectContext(project))
    staged = project / "staged-removals.json"
    _write(staged, data)
    with StageRun(project, "fillers") as run:
        for key, digest in inputs.items():
            run.record_inputs(key, digest)
        run.replace_output(project / "removals.json", staged)
        run.finish()


def test_fillers_registered_between_ingest_and_cut(project: Path) -> None:
    assert [item.name for item in STAGES] == [
        "ingest", "fillers", "cut", "music", "captions", "graphics", "assemble",
    ]
    _finish_ingest(project)
    _edit(project)
    assert status(project)["next"].startswith("yap-studio fillers ")
    _finish_fillers(project)
    assert status(project)["next"].startswith("yap-studio cut ")


def test_cut_can_finish_without_fillers(project: Path) -> None:
    _finish_ingest(project)
    _edit(project)
    _finish_cut(project)
    stages = {item["name"]: item for item in status(project)["stages"]}
    assert stages["fillers"]["state"] == "missing"
    assert stages["cut"]["state"] == "done"


def test_segment_notes_do_not_stale_fillers(project: Path) -> None:
    _finish_ingest(project)
    _edit(project)
    _finish_fillers(project)
    edit = json.loads((project / "edit.json").read_bytes())
    edit["segments"][0].update(note="Tighten the opening", text="First word")
    _write(project / "edit.json", edit)
    assert _stage(status(project), "fillers")["state"] == "done"
    edit["segments"][0]["to"]["offset_s"] = 0.05
    _write(project / "edit.json", edit)
    fillers = _stage(status(project), "fillers")
    assert fillers["state"] == "stale"
    assert fillers["reason"] == "changed input: edit:segments"


def test_legacy_fillers_receipt_with_notes_stays_done(project: Path) -> None:
    _finish_ingest(project)
    _edit(project)
    edit = json.loads((project / "edit.json").read_bytes())
    edit["segments"][0]["note"] = "Keep this opening"
    _write(project / "edit.json", edit)
    legacy_hash = projection_sha256(edit["segments"])
    inputs = fillers_inputs(ProjectContext(project))
    inputs["edit:segments"] = legacy_hash
    staged = project / "staged-removals.json"
    _write(staged, {"schema": "yap-studio/removals@1", "removals": []})
    with StageRun(project, "fillers") as run:
        for key, digest in inputs.items():
            run.record_inputs(key, digest)
        run.replace_output(project / "removals.json", staged)
        run.finish()

    assert fillers_inputs(ProjectContext(project))["edit:segments"] == legacy_hash
    assert _stage(status(project), "fillers")["state"] == "done"
    edit["segments"][0]["to"]["offset_s"] = 0.05
    _write(project / "edit.json", edit)
    fillers = _stage(status(project), "fillers")
    assert fillers["state"] == "stale"
    assert fillers["reason"] == "changed input: edit:segments"


def test_verified_cut_after_segment_edit_points_next_at_review(project: Path) -> None:
    _finish_ingest(project)
    _edit(project)
    _finish_fillers(project)
    _finish_cut(project)
    edit = json.loads((project / "edit.json").read_bytes())
    edit["segments"][0]["from"]["offset_s"] = 0.05
    _write(project / "edit.json", edit)
    _finish_cut(project)
    result = status(project)
    fillers = _stage(result, "fillers")
    assert fillers["state"] == "stale"
    assert fillers["reason"] == "changed input: edit:segments"
    assert _stage(result, "cut")["state"] == "done"
    assert result["needs_human"] == [CUT_REVIEW]
    assert result["next"] == CUT_REVIEW


def test_fresh_cut_points_next_at_review_until_the_picture_starts(project: Path) -> None:
    _finish_ingest(project)
    _edit(project)
    _finish_fillers(project)
    _finish_cut(project)
    assert status(project)["next"] == CUT_REVIEW
    _write(project / "timeline/agent.json", {
        "schema": "yap-studio/timeline@1",
        "words_sha256": {"t1": receipt.sha256_file(project / "ingest/t1/words.json")},
        "elements": [],
    })
    next_action = status(project)["next"]
    assert next_action != CUT_REVIEW
    assert next_action.startswith("yap-studio music ")


@pytest.mark.parametrize(
    "setting", ["max_pause_s", "beat_breath_s", "language", "asr"],
    ids=["pause-limit", "beat-threshold", "language-setting", "engine-setting"],
)
def test_fillers_projection_tracks_consumed_settings(project: Path, setting: str) -> None:
    _finish_ingest(project)
    _edit(project)
    before = fillers_inputs(ProjectContext(project))
    data = json.loads((project / "project.json").read_bytes())
    if setting in {"max_pause_s", "beat_breath_s"}:
        data.setdefault("rules", {})[setting] = 0.5
        key = "rules"
    else:
        data[setting] = "es" if setting == "language" else {"engine": "none", "model": None}
        key = setting
    _write(project / "project.json", data)
    assert fillers_inputs(ProjectContext(project))[key] != before[key]


def test_fillers_projection_excludes_crop_and_end_hold(project: Path) -> None:
    _finish_ingest(project)
    _edit(project)
    before = fillers_inputs(ProjectContext(project))
    data = json.loads((project / "project.json").read_bytes())
    data["takes"][0]["crop_x"] = 0.7
    _write(project / "project.json", data)
    edit = json.loads((project / "edit.json").read_bytes())
    edit["end_hold_s"] = 2
    _write(project / "edit.json", edit)
    assert fillers_inputs(ProjectContext(project)) == before


def test_canonical_projection_hash() -> None:
    assert projection_sha256({"b": 1, "a": "é"}) == projection_sha256({"a": "é", "b": 1})
    expected = hashlib.sha256('{"a":"é","b":1}'.encode()).hexdigest()
    assert projection_sha256({"b": 1, "a": "é"}) == expected
    with pytest.raises(ValueError):
        projection_sha256({"invalid": float("nan")})


def test_ingest_stale_after_requested_audio_stream_changes(project: Path) -> None:
    stage = _finish_ingest(project)
    assert stage["inputs"] == ingest_inputs(ProjectContext(project))
    assert stage["inputs"]["audio_stream"] == projection_sha256({"t1": None})
    data = json.loads((project / "project.json").read_bytes())
    data["takes"][0]["audio_stream"] = 1
    _write(project / "project.json", data)
    state = _stage(status(project), "ingest")
    assert state["state"] == "stale"
    assert state["reason"] == "changed input: audio_stream"


@pytest.mark.parametrize("escape", ["absolute", "parent", "symlink"])
def test_receipt_outputs_outside_project_fail_without_hashing(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, escape: str,
) -> None:
    data = _finish_ingest(project)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"
    outside.write_bytes(b"outside bytes")
    if escape == "symlink":
        link = project / "ingest/escape.bin"
        link.symlink_to(outside)
        named = "ingest/escape.bin"
    elif escape == "absolute":
        named = str(outside)
    else:
        named = f"../{outside.name}"
    data["outputs"].append({"path": named, "sha256": "0" * 64})
    _write(project / "ingest/receipt.json", data)
    original = receipt.sha256_file

    def guarded(target: str | Path) -> str:
        assert Path(target).resolve() != outside.resolve(), "outside output must not be hashed"
        return original(target)

    monkeypatch.setattr(receipt, "sha256_file", guarded)
    run = StageRun(project, "ingest")
    assert not run.up_to_date(data["inputs"])
    state = _stage(status(project), "ingest")
    assert state["state"] == "failed"
    assert state["reason"] == "receipt names a file outside the project"
    _write(project / "ingest/receipt.pending.json", {"started": data["started"]})
    with pytest.raises(InputError, match="receipt names a file outside the project"):
        with run:
            pytest.fail("unsafe recovery must be refused")


def test_identity_cache_touch_rehashes_once_and_keeps_done(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _finish_ingest(project)
    assert _stage(status(project), "ingest")["state"] == "done"
    take = project / "takes/first.mov"
    calls = []
    original = receipt.sha256_file

    def counting(path: str | Path) -> str:
        if Path(path) == take:
            calls.append(path)
        return original(path)

    monkeypatch.setattr(receipt, "sha256_file", counting)
    cache = IdentityCache(project)
    old_hash = cache.sha256(take)
    assert calls == []
    identity = take.stat()
    os.utime(take, ns=(identity.st_atime_ns, identity.st_mtime_ns + 1_000_000))
    assert _stage(status(project), "ingest")["state"] == "done"
    assert cache.sha256(take) == old_hash
    assert calls == [take]
    metadata = json.loads((project / ".yap/identity.json").read_bytes())[str(take.resolve())]
    assert set(metadata) == {"sha256", "size", "mtime_ns", "ctime_ns", "ino", "dev"}


@pytest.fixture
def audio_take(tmp_path: Path) -> Path:
    """Build a tiny real take with two distinct encoded audio streams."""
    take = tmp_path / "audio.mp4"
    ffmpeg.run(
        [
            "ffmpeg", "-nostdin", "-v", "error",
            "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=10:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=880:duration=1",
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0",
            "-c:v", "libx264", "-c:a", "aac", str(take),
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    return take


@pytest.fixture
def audio_project(audio_take: Path) -> Path:
    """Publish an ingest snapshot for real media with two audio streams."""
    root = audio_take.parent
    _write(root / "project.json", {
        "schema": "yap-studio/project@1", "name": "audio-story",
        "takes": [{"id": "t1", "path": audio_take.name}],
    })
    _finish_ingest(root, audio_fingerprint=True)
    return root


@pytest.mark.parametrize("change", ["proxy", "reencode", "mtime"])
def test_take_changes_keep_audio_stages_or_stale_ingest(audio_project: Path, change: str) -> None:
    _edit(audio_project)
    _finish_fillers(audio_project)
    _finish_cut(audio_project)
    before = status(audio_project)
    for name in ("ingest", "fillers", "cut"):
        assert _stage(before, name)["state"] == "done"
    source = audio_project / "audio.mp4"
    original = ProjectContext(audio_project).ingested_take("t1")
    assert original is not None
    if change == "mtime":
        identity = source.stat()
        os.utime(source, ns=(identity.st_atime_ns, identity.st_mtime_ns + 1_000_000))
    else:
        stand_in = audio_project / "stand-in.mp4"
        ffmpeg.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-map", "0",
                "-vf", "scale=32:32", "-c:v", "libx264",
                "-c:a", "copy" if change == "proxy" else "aac",
                *([] if change == "proxy" else ["-b:a", "64k"]), str(stand_in),
            ],
            timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
        )
        stand_in.replace(source)
        assert receipt.sha256_file(source) != original["sha256"]
        assert (receipts.audio_stream_sha256(source, 0) == original["audio_sha256"]) is (
            change == "proxy"
        )
    after = status(audio_project)
    if change == "proxy":
        assert _stage(after, "ingest")["state"] == "done"
        assert _stage(after, "fillers")["state"] == "done"
        assert _stage(after, "cut")["state"] == "stale"
        assert "take:t1" in _stage(after, "cut")["reason"]
    elif change == "reencode":
        assert _stage(after, "ingest")["state"] == "stale"
        assert "take:t1" in _stage(after, "ingest")["reason"]
    else:
        assert {item["name"]: item["state"] for item in after["stages"]} == {
            item["name"]: item["state"] for item in before["stages"]
        }


def test_check_take_refuses_missing_recorded_audio_stream(audio_project: Path) -> None:
    data = json.loads((audio_project / "project.json").read_bytes())
    data["takes"][0]["audio_stream"] = 1
    _write(audio_project / "project.json", data)
    context = ProjectContext(audio_project)
    recorded = context.take_fingerprint("t1")
    assert recorded["audio_stream"] == 1
    manifest = context.json("ingest/ingest.json")
    manifest["takes"][0].update({**recorded, "requested_audio_stream": 1})
    _write(audio_project / "ingest/ingest.json", manifest)
    source = context.take_path("t1")
    replacement = audio_project / "one-stream.mp4"
    ffmpeg.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", str(replacement),
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    replacement.replace(source)
    changed = ProjectContext(audio_project)
    with pytest.raises(TakeChangedError) as caught:
        changed.check_take("t1", recorded, since="ingest")
    error = caught.value
    assert error.message == "Take t1 (audio.mp4) changed since ingest: its audio stream differs."
    assert isinstance(error.__cause__, receipts.AudioStreamMissingError)
    assert "audio stream 1 is missing" in str(error.__cause__)
    assert "yap-studio ingest DIR --force" in error.suggested_action
    assert changed.take_identity("t1") == receipt.sha256_file(source) != recorded["sha256"]


@pytest.mark.parametrize(
    "failure",
    [InputError("audio.mp4", "cannot probe audio"), ProcessingError("ffmpeg", 1, "probe failed")],
    ids=["input-error", "processing-error"],
)
def test_check_take_propagates_other_audio_failures(
    audio_project: Path, monkeypatch: pytest.MonkeyPatch, failure: InputError | ProcessingError,
) -> None:
    context = ProjectContext(audio_project)
    recorded = context.ingested_take("t1")
    assert recorded is not None
    recorded = {**recorded, "sha256": "0" * 64}

    def failed_audio(path: Path, ordinal: int) -> str:
        assert path == audio_project / "audio.mp4" and ordinal == 0
        raise failure

    monkeypatch.setattr(receipts, "audio_stream_sha256", failed_audio)
    with pytest.raises(type(failure)) as caught:
        context.check_take("t1", recorded, since="ingest")
    assert caught.value is failure


@pytest.mark.parametrize("recorded", [True, False], ids=["ingested", "no-record"])
def test_take_identity_unchanged_does_not_hash_audio(
    audio_project: Path, monkeypatch: pytest.MonkeyPatch, recorded: bool,
) -> None:
    if not recorded:
        (audio_project / "ingest/ingest.json").unlink()
    context = ProjectContext(audio_project)
    calls = []

    def unexpected_audio(path: Path, ordinal: int) -> str:
        calls.append((path, ordinal))
        pytest.fail("unchanged takes and takes without an ingest record need no audio hash")

    monkeypatch.setattr(context.identity, "audio_sha256", unexpected_audio)
    assert context.take_identity("t1") == receipt.sha256_file(audio_project / "audio.mp4")
    assert context.take_identity("t1") == context.take_sha256("t1")
    assert calls == []


def test_take_fingerprint_reuses_ingest_audio_and_hashes_a_new_selection(
    audio_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ProjectContext(audio_project)
    record = context.ingested_take("t1")
    assert record is not None
    calls = []
    original = IdentityCache.audio_sha256

    def counting(cache: IdentityCache, path: Path, ordinal: int) -> str:
        calls.append((path, ordinal))
        return original(cache, path, ordinal)

    monkeypatch.setattr(IdentityCache, "audio_sha256", counting)
    assert context.take_fingerprint("t1") == {
        key: record[key] for key in ("path", "sha256", "audio_stream", "audio_sha256")
    }
    assert calls == []
    data = context.json("project.json")
    data["takes"][0]["audio_stream"] = 1
    _write(audio_project / "project.json", data)
    selected = ProjectContext(audio_project)
    fingerprint = selected.take_fingerprint("t1")
    assert fingerprint["sha256"] == record["sha256"]
    assert fingerprint["audio_stream"] == 1
    assert fingerprint["audio_sha256"] == receipts.audio_stream_sha256(
        audio_project / "audio.mp4", 1,
    )
    assert fingerprint["audio_sha256"] != record["audio_sha256"]
    assert selected.take_fingerprint("t1") == fingerprint
    assert calls == [(audio_project / "audio.mp4", 1)]


@pytest.mark.parametrize("rehash", ["audio", "file"])
def test_identity_cache_audio_reuses_hash_and_invalidates_both_on_change(
    audio_take: Path, monkeypatch: pytest.MonkeyPatch, rehash: str,
) -> None:
    original = receipts.audio_stream_sha256
    calls = []

    def counting(path: Path, ordinal: int) -> str:
        calls.append((path, ordinal))
        return original(path, ordinal)

    monkeypatch.setattr(receipts, "audio_stream_sha256", counting)
    cache = IdentityCache(audio_take.parent)
    audio_hash = cache.audio_sha256(audio_take, 0)
    assert audio_hash == original(audio_take, 0)
    assert cache.audio_sha256(audio_take, 0) == audio_hash
    assert IdentityCache(audio_take.parent).audio_sha256(audio_take, 0) == audio_hash
    assert calls == [(audio_take, 0)]
    key = str(audio_take.resolve())
    entry = json.loads(cache.path.read_bytes())[key]
    assert "sha256" not in entry
    assert entry["audio"] == {"0": audio_hash}

    file_hash = cache.sha256(audio_take)
    assert file_hash == receipt.sha256_file(audio_take)
    assert json.loads(cache.path.read_bytes())[key]["audio"] == {"0": audio_hash}
    assert cache.audio_sha256(audio_take, 0) == audio_hash
    assert calls == [(audio_take, 0)]

    identity = audio_take.stat()
    os.utime(audio_take, ns=(identity.st_atime_ns, identity.st_mtime_ns + 1_000_000))
    if rehash == "audio":
        assert cache.audio_sha256(audio_take, 0) == audio_hash
        assert "sha256" not in json.loads(cache.path.read_bytes())[key]
    else:
        assert cache.sha256(audio_take) == file_hash
        assert "audio" not in json.loads(cache.path.read_bytes())[key]
    assert cache.sha256(audio_take) == file_hash
    assert cache.audio_sha256(audio_take, 0) == audio_hash
    assert cache.audio_sha256(audio_take, 0) == audio_hash
    assert calls == [(audio_take, 0), (audio_take, 0)]
    entry = json.loads(cache.path.read_bytes())[key]
    assert entry["sha256"] == file_hash
    assert entry["audio"] == {"0": audio_hash}


def test_identity_cache_audio_preserves_other_ordinals_and_file_hash(
    audio_take: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = IdentityCache(audio_take.parent)
    file_hash = cache.sha256(audio_take)
    original = receipts.audio_stream_sha256
    calls = []

    def counting(path: Path, ordinal: int) -> str:
        calls.append(ordinal)
        return original(path, ordinal)

    monkeypatch.setattr(receipts, "audio_stream_sha256", counting)
    hashes = {str(ordinal): cache.audio_sha256(audio_take, ordinal) for ordinal in (0, 1)}
    assert hashes["0"] != hashes["1"]
    assert cache.audio_sha256(audio_take, 0) == hashes["0"]
    assert cache.audio_sha256(audio_take, 1) == hashes["1"]
    assert calls == [0, 1]
    entry = json.loads(cache.path.read_bytes())[str(audio_take.resolve())]
    assert entry["sha256"] == file_hash
    assert entry["audio"] == hashes


@pytest.mark.parametrize("keeps_changing", [False, True], ids=["retry", "refuse"])
def test_identity_cache_audio_retries_changed_file(
    audio_take: Path, monkeypatch: pytest.MonkeyPatch, keeps_changing: bool,
) -> None:
    original = receipts.audio_stream_sha256
    calls = []

    def changing(path: Path, ordinal: int) -> str:
        calls.append(path)
        result = original(path, ordinal)
        if keeps_changing or len(calls) == 1:
            identity = path.stat()
            os.utime(path, ns=(identity.st_atime_ns, identity.st_mtime_ns + 1_000_000))
        return result

    monkeypatch.setattr(receipts, "audio_stream_sha256", changing)
    cache = IdentityCache(audio_take.parent)
    if keeps_changing:
        with pytest.raises(InputError, match="kept changing while hashing; wait and retry"):
            cache.audio_sha256(audio_take, 0)
        assert len(calls) == 3
        assert not cache.path.exists()
    else:
        assert cache.audio_sha256(audio_take, 0) == original(audio_take, 0)
        assert cache.audio_sha256(audio_take, 0) == original(audio_take, 0)
        assert len(calls) == 2


def test_identity_cache_retries_changed_file(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    take = project / "takes/first.mov"
    original = receipt.sha256_file
    calls = []

    def changing(path: str | Path) -> str:
        result = original(path)
        if Path(path) == take:
            calls.append(path)
            if len(calls) == 1:
                take.write_bytes(b"updated source bytes")
        return result

    monkeypatch.setattr(receipt, "sha256_file", changing)
    assert IdentityCache(project).sha256(take) == original(take)
    assert len(calls) == 2


def test_identity_cache_refuses_unstable_file(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    take = project / "takes/first.mov"
    original = receipt.sha256_file

    def unstable(path: str | Path) -> str:
        result = original(path)
        if Path(path) == take:
            with take.open("ab") as stream:
                stream.write(b"more")
        return result

    monkeypatch.setattr(receipt, "sha256_file", unstable)
    with pytest.raises(InputError, match="kept changing"):
        IdentityCache(project).sha256(take)


def test_fresh_status_ingest_next_and_edit_write_after_ingest(project: Path) -> None:
    result = status(project)
    assert result["project"] == "story"
    assert _stage(result, "ingest")["state"] == "missing"
    assert _stage(result, "fillers")["reason"] == "waits for ingest"
    assert result["needs_human"] == []
    assert result["next"].startswith("yap-studio ingest ")
    _finish_ingest(project)
    result = status(project)
    assert _stage(result, "ingest")["state"] == "done"
    assert result["needs"] == ["write edit.json segments (read `yap-studio words DIR`)"]
    assert result["next"] == result["needs"][0]
    _edit(project)
    assert status(project)["next"].startswith("yap-studio fillers ")


def test_status_running_progress_and_same_stage_conflict(project: Path) -> None:
    with StageRun(project, "ingest") as run:
        run.progress("transcribing", 0.4)
        running = _stage(status(project), "ingest")
        assert running["state"] == "running"
        assert running["reason"] == "transcribing"
        assert running["progress"]["pid"] == os.getpid()
        assert running["progress"]["fraction"] == 0.4
        with pytest.raises(YapError) as caught:
            with StageRun(project, "ingest"):
                pytest.fail("second stage should not start")
        assert caught.value.code == "stage_running"
        assert _stage(status(project), "ingest")["state"] == "running"
    assert _stage(status(project), "ingest")["reason"] == "interrupted"


@pytest.mark.parametrize("change", ["take", "asr", "language"],
                         ids=["take-content", "asr-settings", "language-setting"])
def test_status_staleness_names_changed_ingest_input(project: Path, change: str) -> None:
    _finish_ingest(project)
    if change == "take":
        (project / "takes/first.mov").write_bytes(b"changed take")
    else:
        data = json.loads((project / "project.json").read_bytes())
        data[change] = "es" if change == "language" else {"engine": "none", "model": None}
        _write(project / "project.json", data)
    result = status(project)
    assert _stage(result, "ingest")["state"] == "stale"
    assert change in _stage(result, "ingest")["reason"]
    assert _stage(result, "fillers")["state"] == "missing"
    assert _stage(result, "fillers")["reason"] == "waits for ingest"


def test_cut_projection_ignores_unread_settings_and_proposed_removals(project: Path) -> None:
    _finish_ingest(project)
    _edit(project)
    _finish_cut(project)
    _write(project / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [
            {"id": "r1", "take": "t1", "start": 0.1, "end": 0.2, "kind": "manual",
             "status": "proposed"},
        ],
    })
    data = json.loads((project / "project.json").read_bytes())
    data["word_fixes"] = {"first": "opening"}
    data["rules"] = {"zoom_max_per_video": 2}
    _write(project / "project.json", data)
    assert {stage["name"]: stage["state"] for stage in status(project)["stages"]} == {
        "ingest": "done", "fillers": "missing", "cut": "done",
        "music": "missing", "captions": "missing", "graphics": "missing",
        "assemble": "missing",
    }
    removals = json.loads((project / "removals.json").read_bytes())
    removals["removals"][0]["status"] = "applied"
    _write(project / "removals.json", removals)
    result = _stage(status(project), "cut")
    assert result["state"] == "stale"
    assert "removals:applied" in result["reason"]


def test_cut_projection_tracks_platform_without_changing_none(project: Path) -> None:
    _finish_ingest(project)
    _edit(project)
    before = _finish_cut(project)["inputs"]
    assert "project:platform" not in before
    data = json.loads((project / "project.json").read_bytes())
    data["platform"] = "none"
    _write(project / "project.json", data)
    assert cut_inputs(ProjectContext(project)) == before
    data["platform"] = "reels"
    _write(project / "project.json", data)
    assert cut_inputs(ProjectContext(project)) == {
        **before, "project:platform": projection_sha256("reels"),
    }
    assert _stage(status(project), "cut")["reason"] == "changed input: project:platform"


def test_cut_projection_tracks_crop_and_words(project: Path) -> None:
    _finish_ingest(project)
    _edit(project)
    _finish_cut(project)
    data = json.loads((project / "project.json").read_bytes())
    data["takes"][0]["crop_x"] = 0.25
    _write(project / "project.json", data)
    assert _stage(status(project), "cut")["reason"] == "changed input: crop_x"
    original = cut_inputs(ProjectContext(project))
    words = project / "ingest/t1/words.json"
    data = json.loads(words.read_bytes())
    data["words"][0]["text"] = "Changed"
    _write(words, data)
    assert cut_inputs(ProjectContext(project))["words:t1"] != original["words:t1"]


def test_cut_projection_makes_legacy_pause_rule_stale(project: Path) -> None:
    _finish_ingest(project)
    _edit(project)
    data = _finish_cut(project)
    assert "edges:pause_threshold" in cut_inputs(ProjectContext(project))
    data["inputs"].pop("edges:pause_threshold")
    _write(project / "cut/receipt.json", data)
    result = _stage(status(project), "cut")
    assert result["state"] == "stale"
    assert result["reason"] == "changed input: edges:pause_threshold"


def test_fillers_and_cut_share_unchanged_pause_rule_projection(project: Path) -> None:
    _finish_ingest(project)
    _edit(project)
    context = ProjectContext(project)
    expected = projection_sha256({
        "PAUSE_RULE_VERSION": 4,
        "PAUSE_SHARE_MAX": 0.02,
        "PAUSE_BASELINE_MAX": 0.2,
        "PAUSE_MIN_WORDS": 10,
        "PAUSE_SCAN_TOP_DBFS": -10.0,
        "PAUSE_SCAN_STEP_DB": 0.5,
        "PAUSE_ABOVE_FLOOR_MIN_DB": 3.0,
        "SILENCE_ABOVE_FLOOR_DB": 15.0,
        "SILENCE_CEILING_DBFS": -45.0,
        "SILENCE_MIN_S": 0.04,
        "HOP_S": 0.01,
        "VOICED_TRIM_MIN_S": 0.25,
        "VOICED_PAD_S": 0.05,
        "VOICED_OWNER_RATIO": 2,
        "VOICED_RULE_VERSION": 5,
    })
    assert cut_inputs(context)["edges:pause_threshold"] == expected
    assert fillers_inputs(context)["edges:pause_threshold"] == expected


def test_pause_rule_version_change_makes_finished_cut_and_fillers_stale(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.stages import edges

    _finish_ingest(project)
    _edit(project)
    _finish_fillers(project)
    _finish_cut(project)
    before = status(project)
    for name in ("cut", "fillers"):
        assert _stage(before, name)["state"] == "done"

    monkeypatch.setattr(edges, "PAUSE_RULE_VERSION", edges.PAUSE_RULE_VERSION + 1)
    after = status(project)
    for name in ("cut", "fillers"):
        result = _stage(after, name)
        assert result["state"] == "stale"
        assert result["reason"] == "changed input: edges:pause_threshold"


def test_pause_rule_version_3_fillers_receipt_is_stale_but_ingest_stays_current(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.stages import edges

    _finish_ingest(project)
    _edit(project)
    with monkeypatch.context() as previous:
        previous.setattr(edges, "PAUSE_RULE_VERSION", 3)
        _finish_fillers(project)
        assert _stage(status(project), "fillers")["state"] == "done"

    after = status(project)
    result = _stage(after, "fillers")
    assert result["state"] == "stale"
    assert result["reason"] == "changed input: edges:pause_threshold"
    assert _stage(after, "ingest")["state"] == "done"


def test_voiced_rule_version_change_makes_finished_cut_and_fillers_stale(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.stages import edges

    _finish_ingest(project)
    _edit(project)
    _finish_fillers(project)
    _finish_cut(project)
    before = status(project)
    for name in ("cut", "fillers"):
        assert _stage(before, name)["state"] == "done"

    monkeypatch.setattr(edges, "VOICED_RULE_VERSION", edges.VOICED_RULE_VERSION + 1)
    after = status(project)
    for name in ("cut", "fillers"):
        result = _stage(after, name)
        assert result["state"] == "stale"
        assert result["reason"] == "changed input: edges:pause_threshold"


def test_fillers_projection_tracks_pause_rule_changes(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.stages import edges

    _finish_ingest(project)
    _edit(project)
    _finish_fillers(project)
    monkeypatch.setattr(edges, "PAUSE_BASELINE_MAX", 0.3)
    result = _stage(status(project), "fillers")
    assert result["state"] == "stale"
    assert result["reason"] == "changed input: edges:pause_threshold"


def test_cut_projection_tracks_pause_baseline_limit(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.stages import edges

    _finish_ingest(project)
    _edit(project)
    _finish_cut(project)
    monkeypatch.setattr(edges, "PAUSE_BASELINE_MAX", 0.3)
    result = _stage(status(project), "cut")
    assert result["state"] == "stale"
    assert result["reason"] == "changed input: edges:pause_threshold"


@pytest.mark.parametrize("change", ["language", "engine", "model"])
@pytest.mark.parametrize("time_only", [False, True])
def test_cut_projection_tracks_speech_settings(
    project: Path, change: str, time_only: bool,
) -> None:
    _finish_ingest(project)
    _edit(project)
    if time_only:
        data = json.loads((project / "edit.json").read_bytes())
        data["segments"][0].update({"from": {"time": 0}, "to": {"time": 0.5}})
        _write(project / "edit.json", data)
    _finish_cut(project)
    context = ProjectContext(project)
    inputs = cut_inputs(context)
    assert inputs["asr"] == projection_sha256(context.project.asr)
    assert inputs["language"] == projection_sha256(context.project.language)
    data = json.loads((project / "project.json").read_bytes())
    if change == "language":
        data["language"] = "es"
    else:
        data["asr"] = {**context.project.asr, change: "faster-whisper" if change == "engine"
                       else "large-v3"}
    _write(project / "project.json", data)
    # Refresh ingest without changing words, isolating the cut's own projection.
    _finish_ingest(project)
    result = _stage(status(project), "cut")
    assert result["state"] == "stale"
    assert result["reason"] == f"changed input: {'language' if change == 'language' else 'asr'}"


@pytest.mark.parametrize("identity", [
    {"engine": "faster-whisper", "model": "small.en"},
    {"engine": "parakeet-onnx", "model": "custom-model"},
    {"engine": "none", "model": None},
])
def test_cut_projection_tracks_auto_engine_identity(
    project: Path, monkeypatch: pytest.MonkeyPatch, identity: dict[str, str | None],
) -> None:
    monkeypatch.setattr(project_status, "_edge_engine_identity",
                        lambda context: {"engine": "parakeet-onnx", "model": "default-model"})
    _finish_ingest(project)
    _edit(project)
    _finish_cut(project)
    assert _stage(status(project), "cut")["state"] == "done"
    monkeypatch.setattr(project_status, "_edge_engine_identity", lambda context: identity)
    result = _stage(status(project), "cut")
    assert result["state"] == "stale"
    assert result["reason"] == "changed input: asr:resolved"


@pytest.mark.parametrize(("engine", "model", "available", "expected"), [
    ("auto", None, list(backends.ENGINES),
     {"engine": "parakeet-mlx", "model": "mlx-community/parakeet-tdt-0.6b-v3"}),
    ("auto", None, ["parakeet-onnx", "faster-whisper"],
     {"engine": "parakeet-onnx", "model": "nemo-parakeet-tdt-0.6b-v3"}),
    ("auto", "large-v3", ["faster-whisper"],
     {"engine": "faster-whisper", "model": "large-v3"}),
    ("auto", None, [], {"engine": "none", "model": None}),
    ("faster-whisper", "large-v3", ["faster-whisper"],
     {"engine": "faster-whisper", "model": "large-v3"}),
    ("faster-whisper", "large-v3", [], {"engine": "none", "model": None}),
    ("none", None, list(backends.ENGINES), {"engine": "none", "model": None}),
])
def test_edge_engine_identity_resolves_availability_without_transcribing(
    project: Path, monkeypatch: pytest.MonkeyPatch, engine: str, model: str | None,
    available: list[str], expected: dict[str, str | None],
) -> None:
    data = json.loads((project / "project.json").read_bytes())
    data["asr"] = {"engine": engine, "model": model}
    _write(project / "project.json", data)
    monkeypatch.setattr(backends, "available_engines",
                        lambda: {name: name in available for name in backends.ENGINES})
    assert _edge_engine_identity(ProjectContext(project)) == expected


def test_cut_projection_tracks_requested_and_ingested_audio_streams(project: Path) -> None:
    _finish_ingest(project)
    _edit(project)
    _finish_cut(project)
    data = json.loads((project / "project.json").read_bytes())
    data["takes"][0]["audio_stream"] = 1
    _write(project / "project.json", data)
    stages = {stage["name"]: stage for stage in status(project)["stages"]}
    assert stages["ingest"]["reason"] == "changed input: audio_stream"
    assert stages["cut"]["reason"] == "waits for ingest"
    requested = cut_inputs(ProjectContext(project))["audio_stream"]
    _write(project / "ingest/ingest.json", {"takes": [{"id": "t1", "audio_stream": 1}]})
    assert cut_inputs(ProjectContext(project))["audio_stream"] != requested


def test_snapshot_mid_render_edit_leaves_cut_stale(project: Path) -> None:
    _finish_ingest(project)
    _edit(project)
    context = ProjectContext(project)
    snapshot = context.read("edit.json")
    inputs = cut_inputs(context)
    staged = project / "staged.mp4"
    staged.write_bytes(b"rendered only from the snapshot")
    with StageRun(project, "cut") as run:
        for name, digest in inputs.items():
            run.record_inputs(name, digest)
        data = json.loads(snapshot)
        data["segments"][0]["to"] = {"time": 0.7}
        _write(project / "edit.json", data)
        assert context.read("edit.json") == snapshot
        run.replace_output(project / "cut/cut.mp4", staged)
        final_receipt = run.finish()
    assert final_receipt["inputs"]["edit.json"] == hashlib.sha256(snapshot).hexdigest()
    result = _stage(status(project), "cut")
    assert result["state"] == "stale"
    assert result["reason"] == "changed input: edit.json"


def test_snapshot_retains_missing_optional_file_and_take_hash(project: Path) -> None:
    _finish_ingest(project)
    _edit(project)
    context = ProjectContext(project)
    inputs = cut_inputs(context)
    _write(project / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [
            {"id": "r1", "take": "t1", "start": 0.1, "end": 0.2,
             "kind": "manual", "status": "applied"},
        ],
    })
    (project / "takes/first.mov").write_bytes(b"changed during a run")
    assert cut_inputs(context) == inputs
    assert cut_inputs(ProjectContext(project)) != inputs


def test_stage_receipt_relative_outputs_and_unchanged_noop(project: Path) -> None:
    finished = _finish_ingest(project)
    assert finished["schema"] == "yap-studio/stage-receipt@1"
    assert finished["status"] == "verified"
    assert finished["asserts"] == [{"id": "source", "pass": True}]
    assert finished["outputs"][0]["path"] == "ingest/t1/words.json"
    assert "yap_studio" in finished["versions"]
    assert not (project / "ingest/receipt.pending.json").exists()
    assert not (project / "ingest/progress.json").exists()
    before = (project / "ingest/receipt.json").read_bytes()
    run = StageRun(project, "ingest")
    assert run.up_to_date(ingest_inputs(ProjectContext(project)))
    assert (project / "ingest/receipt.json").read_bytes() == before
    assert read_receipt(project / "ingest") == finished
    assert read_receipt(project / "not-yet") is None
    _finish_ingest(project)
    assert _stage(status(project), "ingest")["state"] == "done"


def test_stage_receipt_times_progress_phases(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([10.0, 10.25, 12.0, 14.0])
    monkeypatch.setattr(receipts.time, "perf_counter", lambda: next(ticks))
    with StageRun(project, "ingest") as run:
        run.progress("transcribing", 0.2)
        run.progress("transcribing", 0.4)
        run.progress("publishing", 0.9)
        finished = run.finish()
    assert finished["timings"] == {
        "phases": [
            {"name": "starting", "seconds": 0.25},
            {"name": "transcribing", "seconds": 1.75},
            {"name": "publishing", "seconds": 2.0},
        ],
        "total_s": 4.0,
    }
    assert sum(phase["seconds"] for phase in finished["timings"]["phases"]) <= (
        finished["timings"]["total_s"] + 0.003
    )
    assert json.loads((project / "ingest/receipt.json").read_text())["timings"] == (
        finished["timings"]
    )


def test_stage_receipt_timings_include_output_hash(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [10.0]
    monkeypatch.setattr(receipts.time, "perf_counter", lambda: clock[0])
    staged = project / "staged.bin"
    staged.write_bytes(b"rendered output")
    output = project / "ingest/output.bin"
    with StageRun(project, "ingest") as run:
        clock[0] = 12.0
        run.progress("publishing", 0.9)
        run.replace_output(output, staged)
        clock[0] = 14.0
        original_sha256_file = receipt.sha256_file

        def slow_hash(path: Path) -> str:
            clock[0] += 3.0
            return original_sha256_file(path)

        monkeypatch.setattr(receipt, "sha256_file", slow_hash)
        finished = run.finish()

    assert finished["timings"] == {
        "phases": [
            {"name": "starting", "seconds": 2.0},
            {"name": "publishing", "seconds": 5.0},
        ],
        "total_s": 7.0,
    }


def test_failed_stage_receipt_has_timings(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([2.0, 2.1234, 3.1234])
    monkeypatch.setattr(receipts.time, "perf_counter", lambda: next(ticks))
    with StageRun(project, "ingest") as run:
        run.progress("checking", 0.5)
        finished = run.finish("failed")
    assert finished["status"] == "failed"
    assert finished["timings"] == {
        "phases": [
            {"name": "starting", "seconds": 0.123},
            {"name": "checking", "seconds": 1.0},
        ],
        "total_s": 1.123,
    }


def test_stage_receipt_refuses_extra_timings(project: Path) -> None:
    with StageRun(project, "ingest") as run:
        with pytest.raises(ValueError, match="receipt contract fields"):
            run.finish(extra={"timings": {"phases": [], "total_s": 0}})


def test_exception_after_one_output_retries_all_and_refuses_hand_edit(project: Path) -> None:
    staged = project / "staged.bin"
    staged.write_bytes(b"first intended bytes")
    first = project / "ingest/first.bin"
    second = project / "ingest/second.bin"
    with pytest.raises(RuntimeError, match="injected crash"):
        with StageRun(project, "ingest") as run:
            run.replace_output(first, staged)
            raise RuntimeError("injected crash")
    assert first.read_bytes() == b"first intended bytes"
    assert not staged.exists()
    assert not second.exists()
    assert not (project / "ingest/progress.json").exists()
    assert _stage(status(project), "ingest")["reason"] == "interrupted"
    staged.write_bytes(b"first intended bytes")
    with StageRun(project, "ingest") as run:
        run.replace_output(first, staged)
        staged.write_bytes(b"second intended bytes")
        run.replace_output(second, staged)
        final_receipt = run.finish()
    assert len(final_receipt["outputs"]) == 2
    first.write_bytes(b"hand edited")
    staged.write_bytes(b"first intended bytes")
    with pytest.raises(OutputPathError):
        with StageRun(project, "ingest") as run:
            run.replace_output(first, staged)
    assert first.read_bytes() == b"hand edited"


@pytest.mark.parametrize("existing", [False, True])
def test_crash_between_journal_and_replace_recovers(
    project: Path, monkeypatch: pytest.MonkeyPatch, existing: bool,
) -> None:
    target = project / "ingest/result.bin"
    staged = project / "staged.bin"
    staged.write_bytes(b"old bytes")
    if existing:
        with StageRun(project, "ingest") as run:
            run.replace_output(target, staged)
            run.finish()
    staged.write_bytes(b"new bytes")
    original = StageRun._journal_output

    def interrupted(run: StageRun, entry: dict[str, Any]) -> None:
        original(run, entry)
        raise RuntimeError("journal persisted")

    with monkeypatch.context() as patch:
        patch.setattr(StageRun, "_journal_output", interrupted)
        with pytest.raises(RuntimeError, match="journal persisted"):
            with StageRun(project, "ingest") as run:
                run.replace_output(target, staged)
    assert not target.exists() or target.read_bytes() == b"old bytes"
    pending = json.loads((project / "ingest/receipt.pending.json").read_bytes())
    assert pending["outputs"][-1]["new_sha256"] == receipt.sha256_file(staged)
    with StageRun(project, "ingest") as run:
        run.replace_output(target, staged)
        run.finish()
    assert target.read_bytes() == b"new bytes"


def test_interrupted_output_edited_by_hand_is_never_claimed(project: Path) -> None:
    target = project / "ingest/result.json"
    staged = project / "staged.json"
    staged.write_text('{"schema":"yap-studio/example@1","text":"engine"}', encoding="utf-8")
    with pytest.raises(RuntimeError):
        with StageRun(project, "ingest") as run:
            run.replace_output(target, staged)
            raise RuntimeError("interrupted")
    target.write_text('{"schema":"yap-studio/example@1","text":"hand"}', encoding="utf-8")
    before = (project / "ingest/receipt.pending.json").read_bytes()
    staged.write_text('{"schema":"yap-studio/example@1","text":"engine"}', encoding="utf-8")
    with pytest.raises(OutputPathError):
        with StageRun(project, "ingest") as run:
            run.replace_output(target, staged)
    after = json.loads((project / "ingest/receipt.pending.json").read_bytes())
    assert after["outputs"] == json.loads(before)["outputs"]
    assert json.loads(target.read_bytes())["text"] == "hand"


@pytest.mark.parametrize("pending", [False, True])
def test_deleted_receipt_with_outputs_is_interrupted(project: Path, pending: bool) -> None:
    _finish_ingest(project)
    stage_dir = project / "ingest"
    if pending:
        os.replace(stage_dir / "receipt.json", stage_dir / "receipt.pending.json")
    else:
        (stage_dir / "receipt.json").unlink()
    result = _stage(status(project), "ingest")
    assert result["state"] == "failed"
    assert result["reason"] == "interrupted"
    assert result["receipt"] is None


def test_crash_after_final_receipt_before_journal_cleanup_recovers(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = project / "staged.bin"
    staged.write_bytes(b"completed output")
    target = project / "ingest/result.bin"
    unlink = Path.unlink

    def interrupted(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.name == "receipt.pending.json":
            raise RuntimeError("receipt written before cleanup")
        unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", interrupted)
        with pytest.raises(RuntimeError, match="before cleanup"):
            with StageRun(project, "ingest") as run:
                run.replace_output(target, staged)
                run.finish()
    assert (project / "ingest/receipt.json").is_file()
    assert (project / "ingest/receipt.pending.json").is_file()
    staged.write_bytes(b"completed output")
    with StageRun(project, "ingest") as run:
        run.replace_output(target, staged)
        run.finish()
    assert not (project / "ingest/receipt.pending.json").exists()


def test_same_device_output_moves_staged_file_without_copying(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = project / "staged.mov"
    content = b"staged layer bytes"
    staged.write_bytes(content)
    digest = receipt.sha256_file(staged)
    target = project / "ingest/layer.mov"

    def no_copy(*args: Any, **kwargs: Any) -> None:
        pytest.fail("same-device output must move the staged file")

    monkeypatch.setattr(receipts.shutil, "copyfile", no_copy)
    with StageRun(project, "ingest") as run:
        run.replace_output(target, staged)
        assert not staged.exists()
        assert target.read_bytes() == content
        finished = run.finish()
    assert finished["outputs"][0]["sha256"] == digest


@pytest.mark.parametrize("error_number", [errno.EXDEV, errno.EACCES])
def test_output_copy_fallback_only_for_cross_device_moves(
    project: Path, monkeypatch: pytest.MonkeyPatch, error_number: int,
) -> None:
    staged = project / "staged.mov"
    staged.write_bytes(b"staged layer bytes")
    target = project / "ingest/layer.mov"
    replace = os.replace
    copyfile = receipts.shutil.copyfile
    copies = []

    def move(source: Any, destination: Any) -> None:
        if Path(source) == staged:
            raise OSError(error_number, "injected move failure")
        replace(source, destination)

    def copy(source: Any, destination: Any) -> Any:
        copies.append(Path(source))
        return copyfile(source, destination)

    monkeypatch.setattr(receipts.os, "replace", move)
    monkeypatch.setattr(receipts.shutil, "copyfile", copy)
    if error_number == errno.EXDEV:
        with StageRun(project, "ingest") as run:
            run.replace_output(target, staged)
            run.finish()
        assert copies == [staged]
        assert target.read_bytes() == staged.read_bytes()
    else:
        with pytest.raises(OSError) as caught:
            with StageRun(project, "ingest") as run:
                run.replace_output(target, staged)
        assert caught.value.errno == errno.EACCES
        assert copies == []
        assert not target.exists()
    assert staged.exists()


def test_moved_staged_output_is_rehashed_before_publication(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = project / "staged.mov"
    staged.write_bytes(b"intended bytes")
    target = project / "ingest/layer.mov"
    replace = os.replace

    def changed(source: Any, destination: Any) -> None:
        if Path(source) == staged:
            staged.write_bytes(b"changed during transfer")
        replace(source, destination)

    monkeypatch.setattr(receipts.os, "replace", changed)
    with pytest.raises(ConflictError):
        with StageRun(project, "ingest") as run:
            run.replace_output(target, staged)
    assert not target.exists()
    assert not list(target.parent.glob(".yap_tmp_*"))


def test_take_cannot_be_replaced_by_lock_metadata(project: Path) -> None:
    take = project / ".yap/lock"
    take.parent.mkdir()
    original = b"source container bytes"
    take.write_bytes(original)
    data = json.loads((project / "project.json").read_bytes())
    data["takes"][0]["path"] = ".yap/lock"
    _write(project / "project.json", data)
    with pytest.raises(OutputPathError, match="input file"):
        with StageRun(project, "ingest"):
            pytest.fail("input must be protected before any bookkeeping write")
    assert take.read_bytes() == original


def test_missing_original_remains_protected_from_bookkeeping(project: Path) -> None:
    copied = project / "ingest/takes/t1.mov"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(b"verified source bytes")
    data = json.loads((project / "project.json").read_bytes())
    data["takes"][0]["path"] = "cut/progress.json"
    _write(project / "project.json", data)
    _write(project / "ingest/ingest.json", {"takes": [
        {"id": "t1", "path": "cut/progress.json", "copy": "ingest/takes/t1.mov",
         "sha256": receipt.sha256_file(copied)},
    ]})
    with pytest.raises(OutputPathError, match="input file"):
        with StageRun(project, "cut"):
            pytest.fail("missing original must remain a protected input")
    assert not (project / "cut/progress.json").exists()


@pytest.mark.parametrize("copied", [False, True])
def test_stage_project_and_take_validation_holds_project_lock(
    project: Path, monkeypatch: pytest.MonkeyPatch, copied: bool,
) -> None:
    from yap_studio import paths

    if copied:
        source = project / "ingest/takes/t1.mov"
        source.parent.mkdir(parents=True)
        source.write_bytes((project / "takes/first.mov").read_bytes())
        _write(project / "ingest/ingest.json", {"takes": [
            {"id": "t1", "path": "takes/first.mov", "copy": "ingest/takes/t1.mov",
             "sha256": receipt.sha256_file(source)},
        ]})
    active = False
    original_lock = receipts.project_lock
    original_read = ProjectContext.read
    original_validate = paths.validate_input
    validated = []

    @contextmanager
    def checked_lock(root: Path) -> Iterator[None]:
        nonlocal active
        assert not active, "the project lock must not be acquired twice"
        with original_lock(root):
            active = True
            try:
                yield
            finally:
                active = False

    def checked_read(context: ProjectContext, relative: str) -> bytes:
        assert active, "project snapshots must be read under the lock"
        return original_read(context, relative)

    def checked_validate(target: str | Path) -> Path:
        assert active, "stage inputs must be validated under the lock"
        validated.append(Path(target))
        return original_validate(target)

    monkeypatch.setattr(receipts, "project_lock", checked_lock)
    monkeypatch.setattr(ProjectContext, "read", checked_read)
    monkeypatch.setattr(paths, "validate_input", checked_validate)
    with StageRun(project, "ingest") as run:
        run.finish()
    assert project / "project.json" in validated
    assert project / ("ingest/takes/t1.mov" if copied else "takes/first.mov") in validated


def test_stage_failed_receipt_and_dead_progress(project: Path) -> None:
    with StageRun(project, "ingest") as run:
        run.finish(status="failed", asserts=[{"id": "source", "pass": False,
                                              "expected": "present", "actual": "missing"}])
    failed = _stage(status(project), "ingest")
    assert failed["state"] == "failed"
    assert failed["reason"] == "failed: source (expected present, actual missing)"
    (project / "ingest/receipt.json").unlink()
    _write(project / "ingest/progress.json", {"pid": 2147483647, "step": "abandoned"})
    assert _stage(status(project), "ingest")["reason"] == "interrupted"


def test_unowned_output_and_take_are_refused(project: Path) -> None:
    target = project / "ingest/unknown.bin"
    target.parent.mkdir()
    target.write_bytes(b"unowned")
    staged = project / "staged.bin"
    staged.write_bytes(b"replacement")
    with StageRun(project, "ingest") as run:
        with pytest.raises(OutputPathError):
            run.replace_output(target, staged)
        with pytest.raises(OutputPathError, match="input file"):
            run.replace_output(project / "takes/first.mov", staged)
        with pytest.raises(OutputPathError, match="input file"):
            run.replace_output(project / "project.json", staged)
    assert target.read_bytes() == b"unowned"


def test_concurrent_hand_change_before_replace_is_preserved(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = project / "ingest/result.bin"
    staged = project / "staged.bin"
    staged.write_bytes(b"original")
    with StageRun(project, "ingest") as run:
        run.replace_output(target, staged)
        run.finish()
    staged.write_bytes(b"next version")
    original = StageRun._journal_output

    def edited(run: StageRun, entry: dict[str, Any]) -> None:
        original(run, entry)
        target.write_bytes(b"hand changed during write")

    monkeypatch.setattr(StageRun, "_journal_output", edited)
    with pytest.raises((OutputPathError, ConflictError)):
        with StageRun(project, "ingest") as run:
            run.replace_output(target, staged)
    assert target.read_bytes() == b"hand changed during write"


def test_registry_extension_and_human_hook(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("yap_studio.project.status.STAGES", list(STAGES))
    spec = StageSpec("later", "later", ("cut",), lambda context: {}, human=True)
    register(spec)
    register(StageSpec("later", "later", ("cut",), lambda context: {}, human=True))
    from yap_studio.project.status import STAGES as extended

    assert [item.name for item in extended] == [
        "ingest", "fillers", "cut", "music", "captions", "graphics", "assemble", "later",
    ]
    monkeypatch.setattr("yap_studio.project.status.human_items", lambda context: ["approval hook"])
    _finish_ingest(project)
    _edit(project)
    _finish_cut(project)
    _finish_fillers(project)
    for name, folder in (("music", "music"), ("captions", "captions"),
                         ("graphics", "graphics"), ("assemble", "final")):
        register(StageSpec(name, folder, ("cut",), lambda context: {}))
        with StageRun(project, folder) as run:
            run.finish()
    assert status(project)["needs_human"] == ["approval hook", CUT_REVIEW, "review later"]


def test_verified_cut_requests_review_without_an_approval_stage(project: Path) -> None:
    _finish_ingest(project)
    _edit(project)
    _finish_cut(project)
    result = status(project)
    assert _stage(result, "cut")["state"] == "done"
    assert result["needs_human"] == [CUT_REVIEW]


def test_status_flags_unreplied_saved_version_notes(
    feedback_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, first, _ = _reply_round(feedback_project, tmp_path / "review-data")
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(store.root))
    monkeypatch.setattr(project_status, "STAGES", ())
    set_resulting_version(feedback_project, "b1", "v2")
    store.add_comment(first.media["sha256"], store.review_facts(first.media["sha256"]),
                      {"text": "Unsent extra note", "start_frame": 9})
    result = status(feedback_project)
    assert REPLY_NEED in result["needs"]
    assert UNSENT_NEED in result["needs"]
    assert result["next"] == REPLY_NEED
    write_reply(feedback_project, store, "b1", "v1", "c1", "changed", "Fixed the opening",
                version_id="v2")
    write_reply(feedback_project, store, "b1", "v1", "c2", "kept", "The ending is intact")
    write_reply(feedback_project, store, "b1", "v1", "c3", "declined", "The wording fits")
    result = status(feedback_project)
    assert REPLY_NEED not in result["needs"]
    assert result["needs"] == [UNSENT_NEED]
    assert result["next"] == UNSENT_NEED

    monkeypatch.setattr(ReviewStore, "series", lambda *_args: (_ for _ in ()).throw(OSError()))
    assert REPLY_NEED not in status(feedback_project)["needs"]


def _carried_note_batches(root: Path, data_dir: Path) -> ReviewStore:
    store, _, second = _reply_round(root, data_dir, comments=[_reply_comments[0]])
    video = version_video([str(root)], store, lambda _: [])
    export = _feedback_export(root, [], version=second)
    export["series"] = series_export(store, video.series, 2)
    batch = build_batch(root, export)
    assert batch["id"] == "b2"
    assert [(item["comment"]["version"], item["comment"]["id"], item["carried"])
            for item in batch["items"]] == [("v1", "c1", True)]
    _feedback_cut(root, label="third render")
    save_version(root)
    set_resulting_version(root, "b2", "v3")
    return store


def test_status_carried_note_replied_in_later_batch_is_done(
    feedback_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _carried_note_batches(feedback_project, tmp_path / "review-data")
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(store.root))
    write_reply(feedback_project, store, "b2", "v1", "c1", "declined",
                "No further change needed")
    assert not any("reply to" in need for need in status(feedback_project)["needs"])


def test_status_carried_note_without_reply_counts_once_in_newest_batch(
    feedback_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _carried_note_batches(feedback_project, tmp_path / "review-data")
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(store.root))
    needs = status(feedback_project)["needs"]
    assert len(needs) == 1
    assert needs[0].startswith("reply to 1 review note in b2 before the next review:")


def test_status_sends_earlier_notes_outside_batches(
    feedback_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = save_version(feedback_project)
    store = ReviewStore(tmp_path / "review-data")
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(store.root))
    facts = MediaFacts("render-1.mp4", first.media["sha256"], 3.0, None, False)
    store.add_comment(first.media["sha256"], facts,
                      {"text": "Unsent note", "start_s": 0.5})
    _feedback_cut(feedback_project, label="second render")
    save_version(feedback_project)
    monkeypatch.setattr(project_status, "STAGES", ())
    result = status(feedback_project)
    assert result["needs"] == [UNSENT_NEED]
    assert result["next"] == UNSENT_NEED


def test_status_does_not_send_current_version_notes(
    feedback_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_version(feedback_project)
    _feedback_cut(feedback_project, label="second render")
    second = save_version(feedback_project)
    store = ReviewStore(tmp_path / "review-data")
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(store.root))
    facts = MediaFacts("render-2.mp4", second.media["sha256"], 3.0, None, False)
    store.add_comment(second.media["sha256"], facts,
                      {"text": "Current note", "start_s": 0.5})
    assert not any(need.startswith("send ") for need in status(feedback_project)["needs"])


def test_status_building_current_export_carries_unsent_note(
    feedback_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = save_version(feedback_project)
    store = ReviewStore(tmp_path / "review-data")
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(store.root))
    facts = MediaFacts("render-1.mp4", first.media["sha256"], 3.0, None, False)
    note = store.add_comment(first.media["sha256"], facts,
                             {"text": "Unsent note", "start_s": 0.5})
    _feedback_cut(feedback_project, label="second render")
    second = save_version(feedback_project)
    assert UNSENT_NEED in status(feedback_project)["needs"]
    video = version_video([str(feedback_project)], store, lambda _: [])
    export = _feedback_export(feedback_project, [], version=second)
    export["series"] = series_export(store, video.series, 2)
    batch = build_batch(feedback_project, export)
    assert [(item["comment"]["version"], item["comment"]["id"], item["carried"])
            for item in batch["items"]] == [("v1", note.id, True)]
    result = status(feedback_project)
    assert not any(need.startswith("send ") for need in result["needs"])
    assert any(need.startswith("reply to 1 review note in b1") for need in result["needs"])


def test_status_uses_current_version_after_rollback(
    feedback_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_version(feedback_project)
    _feedback_cut(feedback_project, label="second render")
    second = save_version(feedback_project)
    store = ReviewStore(tmp_path / "review-data")
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(store.root))
    facts = MediaFacts("render-2.mp4", second.media["sha256"], 4.0, Fraction(30), True)
    note = store.add_comment(second.media["sha256"], facts,
                             {"text": "Fix v2", "start_frame": 15})
    build_batch(feedback_project, _feedback_export(feedback_project, [note], version=second))
    store.add_comment(second.media["sha256"], facts,
                      {"text": "Unsent v2 note", "start_frame": 30})
    _feedback_cut(feedback_project, label="third render")
    save_version(feedback_project)
    rollback(feedback_project, "v2")
    assert not any("reply to" in need for need in status(feedback_project)["needs"])
    assert not any(need.startswith("send ") for need in status(feedback_project)["needs"])


def test_status_skips_reply_check_without_two_saved_versions(
    feedback_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "review-data"))
    assert REPLY_NEED not in status(feedback_project)["needs"]
    save_version(feedback_project)
    assert REPLY_NEED not in status(feedback_project)["needs"]


def _saved_cut_version(project: Path) -> SimpleNamespace:
    """Stand in for a saved version holding the current spans and cut inputs."""
    folder = project / "versions/v1"
    folder.mkdir(parents=True, exist_ok=True)
    spans = project / "cut/spans.json"
    if not spans.is_file():
        spans.write_bytes(b"current spans")
    pieces = folder / "cut-pieces.json"
    pieces.write_text(json.dumps({
        "schema": "yap-studio/version-pieces@1", "fps": "30", "total_frames": 30,
        "pieces": [], "inputs": read_receipt(project / "cut")["inputs"],
    }), encoding="utf-8")
    return SimpleNamespace(path=folder, snapshots={
        "spans": {"file": "spans.json", "sha256": receipt.sha256_file(spans)},
        "cut_pieces": {"file": pieces.name, "sha256": receipt.sha256_file(pieces)},
    })


def test_saved_version_of_the_current_cut_is_reviewed_without_saving_again(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.feedback import versions

    _finish_ingest(project)
    _edit(project)
    _finish_cut(project)
    saved = _saved_cut_version(project)
    # The current version may be the cut itself or a final assembled from it.
    monkeypatch.setattr(versions, "current_version", lambda _root: saved)
    assert status(project)["needs_human"] == [
        "review the current version: yap-studio review --versions DIR",
    ]
    saved.snapshots["spans"]["sha256"] = "0" * 64
    assert status(project)["needs_human"] == [CUT_REVIEW]


def test_saved_cut_version_points_next_at_current_review(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.feedback import versions

    _finish_ingest(project)
    _edit(project)
    _finish_cut(project)
    saved = _saved_cut_version(project)
    monkeypatch.setattr(versions, "current_version", lambda _root: saved)
    assert status(project)["next"] == (
        "review the current version: yap-studio review --versions DIR"
    )


def test_recut_after_a_crop_change_is_reviewed_as_a_new_cut(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.feedback import versions

    _finish_ingest(project)
    _edit(project)
    _finish_cut(project)
    saved = _saved_cut_version(project)
    monkeypatch.setattr(versions, "current_version", lambda _root: saved)
    spans = (project / "cut/spans.json").read_bytes()
    data = json.loads((project / "project.json").read_text(encoding="utf-8"))
    data["takes"][0]["crop_x"] = 0.7
    _write(project / "project.json", data)
    assert _stage(status(project), "cut")["reason"] == "changed input: crop_x"
    _finish_cut(project)
    # The crop moves the picture without changing the source spans.
    assert (project / "cut/spans.json").read_bytes() == spans
    result = status(project)
    assert _stage(result, "cut")["state"] == "done"
    assert result["needs_human"] == [CUT_REVIEW]


@pytest.mark.parametrize("slot", [None, "steps", "cues"],
                         ids=["element-edge", "graphic-step", "graphic-cue"])
def test_time_anchors_are_flagged_after_recut(project: Path, slot: str | None) -> None:
    _finish_ingest(project)
    _edit(project)
    lane = project / "timeline/agent.json"
    element = {"id": "agent-gap", "kind": "caption_gap", "start": {"time": 0.1},
               "hold_s": 0.2}
    if slot:
        element = {"id": "agent-card", "kind": "graphic", "start": {"word": "t1:0"},
                   "template": "terminal", "slots": {slot: [{"at": {"time": 0.1}}]}}
        if slot == "cues":
            element.update(template="choices", hold_s=3)
            element["slots"]["options"] = ["One", "Two"]
    _write(lane, {"schema": "yap-studio/timeline@1", "words_sha256": {}, "elements": [element]})
    _finish_cut(project)
    cut_time = (project / "cut/receipt.json").stat().st_mtime_ns
    os.utime(lane, ns=(cut_time - 1_000_000, cut_time - 1_000_000))
    result = status(project)
    assert "review output-time anchors in timeline/agent.json after re-cut" in result["needs"]


def test_stage_run_api_boundaries(project: Path) -> None:
    with pytest.raises(YapError):
        StageRun(project, "../escape")
    run = StageRun(project, "ingest")
    with pytest.raises(RuntimeError):
        run.record_inputs("take", "1" * 64)
    with run:
        with pytest.raises(ValueError):
            run.record_inputs("take", "invalid")
        run.record_inputs("take", "1" * 64)
        with pytest.raises(ConflictError):
            run.record_inputs("take", "2" * 64)
        with pytest.raises(ValueError):
            run.progress("bad", 1.1)
        with pytest.raises(ValueError):
            run.finish(extra={"inputs": {}})
        run.finish()
    with pytest.raises(RuntimeError):
        run.progress("finished", 1)


def test_voiceover_status_warns_about_graphic_only_picture_without_blocking(tmp_path: Path) -> None:
    from test_stage_captions import make_project, set_elements

    root = make_project(tmp_path)
    document = json.loads((root / "project.json").read_text())
    document["takes"][0]["voiceover"] = True
    _write(root / "project.json", document)
    set_elements(root, {"id": "agent-card", "kind": "graphic", "start": {"time": 0},
                        "end": {"time": 3}, "template": "text_tag", "slots": {"text": "Card"}})
    warning = ("Voiceover picture at frames 0-89 comes only from graphics; assembly counts a "
               "frame only where a graphic is opaque and fills the frame.")
    result = status(root)
    assert warning in result["warnings"]
    assert not any(need.startswith("cover the voiceover") for need in result["needs"])
    assert result["next"].startswith("yap-studio music ")
    set_elements(root)
    set_elements(root, {"id": "broll-cover", "kind": "insert", "start": {"time": 0},
                        "end": {"time": 3}, "media": {"path": "source.mov",
                            "sha256": receipt.sha256_file(root / "source.mov")}}, lane="broll")
    result = status(root)
    assert warning not in result["warnings"]
    assert not any(need.startswith("cover the voiceover") for need in result["needs"])
    assert result["next"].startswith("yap-studio music ")


def test_caption_and_graphics_status_next_done_stale_missing(
    tmp_path: Path, fake_layers: list[dict],
) -> None:
    from test_stage_captions import make_project, set_elements

    from yap_studio.stages.captions import captions
    from yap_studio.stages.graphics import graphics
    from yap_studio.stages.music import music

    root = make_project(tmp_path)
    set_elements(root, {"id": "agent-title", "kind": "title", "start": {"time": 0},
                        "lines": ["Small steps"]})
    assert status(root)["next"].startswith("yap-studio music ")
    music(root)
    assert _stage(status(root), "music")["state"] == "done"
    assert status(root)["next"].startswith("yap-studio captions ")
    captions(root)
    assert status(root)["next"].startswith("yap-studio graphics ")
    graphics(root)
    assert status(root)["next"].startswith("yap-studio assemble ") and len(fake_layers) == 2
    data = json.loads((root / "style.json").read_bytes())
    data["captions"]["highlight_style"] = "color"
    _write(root / "style.json", data)
    result = status(root)
    assert _stage(result, "captions")["state"] == "stale"
    assert _stage(result, "graphics")["state"] == "done"
    assert result["next"].startswith("yap-studio captions ")
    captions(root)
    (root / "graphics/agent-title.mov").unlink()
    result = status(root)
    assert _stage(result, "graphics")["state"] == "stale"
    assert "missing output" in _stage(result, "graphics")["reason"]
    assert result["next"].startswith("yap-studio graphics ")


def test_video_choices_keep_stage_inputs_and_status(tmp_path, fake_layers, capsys):
    from test_stage_captions import make_project

    from yap_studio import cli
    from yap_studio.stages.captions import captions
    from yap_studio.stages.graphics import graphics
    from yap_studio.stages.music import music

    root = make_project(tmp_path)
    _write(root / "cut/spans.json", {
        "sources": {"t1": "../source.mov"}, "fps": "30/1",
        "ranges": [{"source": "t1", "start": "0", "end": "3"}],
    })
    ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(root / "source.mov"), "-vn",
                "-c:a", "pcm_f32le", str(root / "cut/voice.wav")], timeout_s=30)
    music(root)
    captions(root)
    graphics(root)
    before = status(root)
    inputs = {spec.name: spec.inputs(ProjectContext(root)) for spec in STAGES}
    assert before["video"] == {"type": None, "broll_sources": None}
    assert cli.main(["status", str(root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["video"] == before["video"]
    assert cli.main(["status", str(root)]) == 0
    assert capsys.readouterr().out.splitlines()[0] == (
        "video: not chosen yet (yap-studio video-type show)")
    assert cli.main(["video-type", "set", str(root), "--type", "talking_head",
                     "--broll", "folder", "--broll", "album"]) == 0
    capsys.readouterr()
    after = status(root)
    assert after["video"] == {"type": "talking_head", "broll_sources": ["folder", "album"]}
    assert cli.main(["status", str(root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["video"] == after["video"]
    assert after["stages"] == before["stages"]
    assert after["needs"] == before["needs"] and after["needs_human"] == before["needs_human"]
    assert {spec.name: spec.inputs(ProjectContext(root)) for spec in STAGES} == inputs
    assert cli.main(["status", str(root)]) == 0
    assert capsys.readouterr().out.splitlines()[0] == (
        "video: talking head; B-roll from your inbox or a folder and a Photos album")


def test_timeline_violation_stales_layers_and_prioritizes_agent_lane_fix(
    tmp_path: Path, fake_layers: list[dict], capsys: pytest.CaptureFixture[str],
) -> None:
    from test_stage_captions import make_project, set_elements

    from yap_studio import cli
    from yap_studio.stages.captions import captions
    from yap_studio.stages.graphics import graphics
    from yap_studio.stages.timeline import resolve_timeline

    root = make_project(tmp_path)
    set_elements(root, {"id": "agent-title", "kind": "title", "start": {"time": 0},
                        "lines": ["Small steps"]})
    captions(root)
    graphics(root)
    before = status(root)
    assert all(_stage(before, name)["state"] == "done" for name in ("captions", "graphics"))
    set_elements(root, {
        "id": "broll-one", "kind": "insert", "start": {"time": 0.2}, "hold_s": 1,
        "media": {"path": "source.mov", "sha256": receipt.sha256_file(root / "source.mov")},
    }, lane="broll")
    violation = next(item for item in resolve_timeline(root).findings
                     if item.rule == "visual_overlap")
    result = status(root)
    for name in ("captions", "graphics"):
        assert _stage(result, name)["state"] == "stale"
        assert _stage(result, name)["reason"] == violation.message
    assert f"{violation.message}; {violation.suggested_action}" in result["needs_human"]
    assert result["next"] == violation.suggested_action
    assert "--lane agent" in result["next"] and "yap-studio captions" not in result["next"]
    assert len(fake_layers) == 2
    assert cli.main(["status", str(root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == result
