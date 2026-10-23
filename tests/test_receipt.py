"""Run receipt tests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from yap_studio.errors import InputError
from yap_studio.project.receipts import failed_assert_lines, retained_outputs
from yap_studio.receipt import build_receipt, sha256_file, write_receipt


def test_sha256_file_and_receipt(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    output = tmp_path / "output.bin"
    missing = tmp_path / "future.bin"
    source.write_bytes(b"source bytes")
    output.write_bytes(b"output bytes")
    started = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
    ended = started + timedelta(seconds=2.5)

    assert sha256_file(source) == hashlib.sha256(b"source bytes").hexdigest()
    receipt = build_receipt(
        [source],
        [output, missing],
        started,
        ended,
        extra={"speech": {"engine": "test", "model": "small"}, "check": "value"},
    )

    assert receipt["schema"] == "yap-studio/receipt@1"
    assert receipt["started_at"] == "2026-09-28T12:00:00Z"
    assert receipt["duration_s"] == 2.5
    assert receipt["inputs"][0] == {
        "path": str(source),
        "sha256": hashlib.sha256(b"source bytes").hexdigest(),
        "bytes": 12,
    }
    assert receipt["outputs"][1] == {"path": str(missing), "sha256": None, "bytes": None}
    assert receipt["speech"] == {"engine": "test", "model": "small"}
    assert receipt["check"] == "value"
    assert receipt["versions"]["ffmpeg"]


def test_write_receipt_is_json_with_trailing_newline(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    receipt = {"schema": "test", "speech": None}
    assert write_receipt(path, receipt) == path
    payload = path.read_text(encoding="utf-8")
    assert payload.endswith("\n")
    assert json.loads(payload) == receipt


def test_pending_ownership_uses_current_hash_and_checks_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    output = project / "final/story.mp4"
    output.parent.mkdir()
    output.write_bytes(b"new")
    digest = sha256_file(output)
    pending = {"outputs": [{"path": "final/story.mp4", "old_sha256": "0" * 64,
                            "new_sha256": digest}]}
    assert retained_outputs(project, pending) == {
        "final/story.mp4": {"path": "final/story.mp4", "sha256": digest, "bytes": 3}}
    pending["outputs"][0]["path"] = "../outside.mp4"
    with pytest.raises(InputError, match="outside the project"):
        retained_outputs(project, pending)


def test_failed_assert_lines_include_available_values() -> None:
    assert failed_assert_lines([
        {"id": "ok", "pass": True},
        {"id": "picture_contract", "pass": False,
         "expected": {"width": 1080}, "actual": {"width": 1}},
        {"id": "missing", "pass": False},
    ]) == ['picture_contract (expected {"width":1080}, actual {"width":1})', "missing"]
