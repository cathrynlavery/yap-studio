"""Shared offline voice projects."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import synth
from test_cut import make_project, time_segment, write_edit, write_json


@pytest.fixture
def voice_project(tmp_path):
    source = synth.voiced_source(
        tmp_path / "source.mov", [(i + 0.2, i + 0.8) for i in range(12)], seconds=12
    )
    root = make_project(
        tmp_path / "project",
        [source],
        word_rows=[
            [
                (word, i + 0.2, i + 0.8)
                for i, word in enumerate(
                    (
                        "bright morning coal flunge project quiet garden water "
                        "after before light home"
                    ).split()
                )
            ]
        ],
    )
    data = json.loads((root / "project.json").read_text())
    data["takes"][0]["voiceover"] = True
    write_json(root / "project.json", data)
    write_edit(root, [time_segment("s1", "t1", 0, 12)])
    return root


@pytest.fixture
def fake_voice(tmp_path, monkeypatch, _no_real_elevenlabs):
    executable = Path(__file__).parent / "fakes/elevenlabs-pp-cli"
    _no_real_elevenlabs(executable)
    monkeypatch.setenv("PATH", str(executable.parent) + os.pathsep + os.environ["PATH"])
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_ELEVENLABS_LOG", str(log))
    return log


def calls(log):
    return (
        [json.loads(line)["argv"] for line in log.read_text().splitlines()] if log.exists() else []
    )


def move_word(root, word_id, field, delta):
    """Publish new source timings and bind the edit to the new ingest."""
    from yap_studio.project.receipts import StageRun
    from yap_studio.project.status import ProjectContext, ingest_inputs

    target = root / "ingest/t1/words.json"
    data = json.loads(target.read_text())
    word = next(word for word in data["words"] if word["id"] == word_id)
    word[field] += delta
    staged = root / "staged-words.json"
    write_json(staged, data)
    inputs = ingest_inputs(ProjectContext(root))
    with StageRun(root, "ingest") as run:
        for name, digest in inputs.items():
            run.record_inputs(name, digest)
        run.replace_output(target, staged)
        run.finish()
    edit = json.loads((root / "edit.json").read_text())
    write_edit(root, edit["segments"])


def linked(root, monkeypatch):
    from yap_studio import receipt
    from yap_studio.voice import consent, link

    records = []
    for i in range(4):
        target = root / f"voice/samples/s{i + 1:02}.wav"
        synth.float_audio(target, np.zeros((15 * 48000, 1)))
        records.append(
            {
                "path": str(target.relative_to(root)),
                "sha256": receipt.sha256_file(target),
                "take": "t1",
                "start_s": i * 15,
                "end_s": (i + 1) * 15,
                "size_bytes": target.stat().st_size,
            }
        )
    write_json(
        root / "voice/samples.json", {"schema": "yap-studio/voice-samples@1", "samples": records}
    )
    monkeypatch.setenv(
        "FAKE_ELEVENLABS_VOICE_SAMPLES",
        json.dumps(
            [
                {"file_name": Path(row["path"]).name, "size_bytes": row["size_bytes"]}
                for row in records
            ]
        ),
    )
    consent(root, yes=True, by="Creator", voice_name="My voice")
    return link(root, name="My voice", yes=True)
