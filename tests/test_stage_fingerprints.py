"""Downstream stages bind synthetic takes to the audio fingerprints recorded by cut."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from test_cli_captions import fake_layers as fake_layers
from test_timeline import write

from yap_studio import ffmpeg, receipt
from yap_studio.project.receipts import audio_stream_sha256
from yap_studio.project.status import (
    ProjectContext,
    cut_inputs,
    fillers_inputs,
    ingest_inputs,
    status,
)
from yap_studio.stages.assemble import assemble
from yap_studio.stages.captions import captions
from yap_studio.stages.graphics import graphics
from yap_studio.stages.music import music
from yap_studio.stages.timeline import TimelineRefusal, resolve_timeline


def make_fingerprinted_project(root: Path, *, ingest_manifest: bool = False) -> Path:
    """Add cut identities and optional ingest facts to the shared real-media project."""
    from test_stage_captions import make_project, set_elements

    root = make_project(root)
    if ingest_manifest:
        context = ProjectContext(root)
        take = context.take_path("t1")
        write(root / "ingest/ingest.json", {
            "schema": "yap-studio/ingest@1", "takes": [{
                "id": "t1", "path": context.project.takes[0].path,
                "sha256": receipt.sha256_file(take), "requested_audio_stream": None,
                "audio_stream": 0, "audio_sha256": audio_stream_sha256(take, 0),
                "audio_offset_s": ffmpeg.probe(take).audio_offset_s,
            }],
        })
        # Compute the receipt projections after the selected ingest stream is recorded.
        for name, inputs in (("ingest", ingest_inputs), ("fillers", fillers_inputs),
                             ("cut", cut_inputs)):
            record = json.loads((root / name / "receipt.json").read_bytes())
            record["inputs"] = inputs(ProjectContext(root))
            write(root / name / "receipt.json", record)
    record = json.loads((root / "cut/receipt.json").read_bytes())
    record["takes"] = {"t1": ProjectContext(root).take_fingerprint("t1")}
    write(root / "cut/receipt.json", record)
    set_elements(root, {"id": "agent-title", "kind": "title", "start": {"time": 0},
                        "lines": ["Small steps"]})
    before = {item["name"]: item for item in status(root)["stages"]}
    assert all(before[name]["state"] == "done" for name in ("ingest", "fillers", "cut"))
    return root


def replace_take(root: Path, *, proxy: bool = False) -> Path:
    """Replace the same take path with encoded picture and copied or re-encoded audio."""
    context = ProjectContext(root)
    take = context.take_path("t1")
    original = context.take_fingerprint("t1")
    replacement = root / f"replacement{take.suffix}"
    ffmpeg.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(take),
            "-map", "0:v:0", "-map", "0:a:0", "-vf", "scale=80:44",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "copy" if proxy else "aac",
            *([] if proxy else ["-b:a", "64k"]), str(replacement),
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    replacement.replace(take)
    assert receipt.sha256_file(take) != original["sha256"]
    assert (audio_stream_sha256(take, 0) == original["audio_sha256"]) is proxy
    return take


@pytest.mark.parametrize("stage", [resolve_timeline, captions, graphics, music, assemble])
def test_reencoded_take_refuses_every_downstream_stage(
    tmp_path: Path, stage: Callable[[Path], Any],
) -> None:
    root = make_fingerprinted_project(tmp_path)
    take = replace_take(root)
    with pytest.raises(TimelineRefusal) as caught:
        stage(root)
    error = caught.value
    assert error.code == "stale_cut"
    assert "t1" in error.message and take.name in error.message
    assert "since the cut" in error.message and "its audio stream differs" in error.message
    assert "yap-studio ingest DIR --force" in error.suggested_action
    assert "yap-studio cut DIR" in error.suggested_action


def test_missing_recorded_audio_stream_refuses_timeline(tmp_path: Path) -> None:
    root = make_fingerprinted_project(tmp_path)
    take = ProjectContext(root).take_path("t1")
    two_streams = root / "two-streams.mov"
    ffmpeg.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(take),
            "-map", "0:v:0", "-map", "0:a:0", "-map", "0:a:0",
            "-c", "copy", str(two_streams),
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    two_streams.replace(take)
    project = json.loads((root / "project.json").read_bytes())
    project["takes"][0]["audio_stream"] = 1
    write(root / "project.json", project)
    for name, inputs in (("ingest", ingest_inputs), ("fillers", fillers_inputs),
                         ("cut", cut_inputs)):
        record = json.loads((root / name / "receipt.json").read_bytes())
        record["inputs"] = inputs(ProjectContext(root))
        write(root / name / "receipt.json", record)
    record = json.loads((root / "cut/receipt.json").read_bytes())
    record["takes"] = {"t1": ProjectContext(root).take_fingerprint("t1")}
    assert record["takes"]["t1"]["audio_stream"] == 1
    write(root / "cut/receipt.json", record)
    assert resolve_timeline(root).valid

    one_stream = root / "one-stream.mov"
    ffmpeg.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(take),
            "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", str(one_stream),
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    one_stream.replace(take)
    with pytest.raises(TimelineRefusal) as caught:
        resolve_timeline(root)
    error = caught.value
    assert error.code == "stale_cut"
    assert error.message == (
        f"Take t1 ({take.name}) changed since the cut: its audio stream differs."
    )
    assert f"Put the original file back at {take.name}" in error.suggested_action
    assert "yap-studio ingest DIR --force" in error.suggested_action
    assert "yap-studio cut DIR" in error.suggested_action


def test_reencoded_take_stales_ingest_and_blocks_layer_renders(
    tmp_path: Path, fake_layers: list[dict],
) -> None:
    root = make_fingerprinted_project(tmp_path)
    captions(root)
    graphics(root)
    before = {item["name"]: item for item in status(root)["stages"]}
    assert all(before[name]["state"] == "done" for name in ("captions", "graphics"))
    take = replace_take(root)
    result = status(root)
    after = {item["name"]: item for item in result["stages"]}
    assert after["ingest"]["state"] == "stale"
    assert "take:t1" in after["ingest"]["reason"]
    for name in ("captions", "graphics"):
        assert after[name]["state"] != "done"
        assert after[name]["reason"].startswith("waits for ")
    assert result["next"].startswith("yap-studio ingest ")
    for stage in (captions, graphics):
        with pytest.raises(TimelineRefusal) as caught:
            stage(root)
        error = caught.value
        assert error.code == "stale_cut"
        assert "t1" in error.message and take.name in error.message
        assert "its audio stream differs" in error.message
        assert "yap-studio ingest DIR --force" in error.suggested_action
        assert "yap-studio cut DIR" in error.suggested_action
    assert len(fake_layers) == 2


def test_proxy_preserves_ingest_but_requires_a_new_picture_cut(tmp_path: Path) -> None:
    root = make_fingerprinted_project(tmp_path, ingest_manifest=True)
    replace_take(root, proxy=True)
    with pytest.raises(TimelineRefusal) as caught:
        resolve_timeline(root)
    error = caught.value
    assert error.code == "stale_cut"
    assert "The cut is stale" in error.message and "take:t1" in error.message
    assert "run yap-studio cut DIR" in error.message
    assert "audio stream differs" not in error.message
    after = {item["name"]: item for item in status(root)["stages"]}
    assert after["ingest"]["state"] == "done"
    assert after["cut"]["state"] == "stale"


def test_mtime_only_change_keeps_done_stages_and_layer_noops(
    tmp_path: Path, fake_layers: list[dict],
) -> None:
    root = make_fingerprinted_project(tmp_path, ingest_manifest=True)
    music(root)
    captions(root)
    graphics(root)
    before = {item["name"]: item["state"] for item in status(root)["stages"]}
    assert all(before[name] == "done" for name in ("music", "captions", "graphics"))
    take = ProjectContext(root).take_path("t1")
    identity = take.stat()
    os.utime(take, ns=(identity.st_atime_ns, identity.st_mtime_ns + 1_000_000))
    after = {item["name"]: item["state"] for item in status(root)["stages"]}
    assert all(after[name] == "done" for name, state in before.items() if state == "done")
    assert captions(root)["noop"] is True
    assert graphics(root)["noop"] is True
    assert len(fake_layers) == 2


@pytest.mark.parametrize("legacy", ["absent", "not-a-dict"])
def test_legacy_cut_still_refuses_reencoded_take(tmp_path: Path, legacy: str) -> None:
    root = make_fingerprinted_project(tmp_path)
    record = json.loads((root / "cut/receipt.json").read_bytes())
    if legacy == "absent":
        record.pop("takes")
    else:
        record["takes"] = []
    write(root / "cut/receipt.json", record)
    replace_take(root)
    with pytest.raises(TimelineRefusal) as caught:
        resolve_timeline(root)
    assert caught.value.code == "stale_cut"
    assert "The ingest is stale" in caught.value.message
    assert "changed input: take:t1" in caught.value.message


@pytest.mark.parametrize("recorded", [[], {}, {"sha256": 42}],
                         ids=["not-a-dict", "missing-sha256", "non-string-sha256"])
def test_unreadable_cut_take_record_refuses_resolution(tmp_path: Path, recorded: Any) -> None:
    root = make_fingerprinted_project(tmp_path)
    record = json.loads((root / "cut/receipt.json").read_bytes())
    record["takes"]["t1"] = recorded
    write(root / "cut/receipt.json", record)
    with pytest.raises(TimelineRefusal) as caught:
        resolve_timeline(root)
    assert caught.value.code == "stale_cut"
    assert caught.value.message == (
        "The cut receipt has an unreadable record for take t1; run yap-studio cut DIR."
    )


def test_cut_fingerprints_ignore_takes_no_longer_in_project(tmp_path: Path) -> None:
    root = make_fingerprinted_project(tmp_path)
    record = json.loads((root / "cut/receipt.json").read_bytes())
    record["takes"]["retired"] = {**record["takes"]["t1"], "sha256": "0" * 64}
    write(root / "cut/receipt.json", record)
    assert resolve_timeline(root).valid
