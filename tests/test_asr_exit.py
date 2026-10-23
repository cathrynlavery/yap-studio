"""Opt-in regression for real console verification through native engine shutdown."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import synth

from yap_studio import process
from yap_studio.report import exit_code
from yap_studio.transcript import ParakeetOnnxBackend


@pytest.mark.asr
def test_repeated_verify_exits_with_its_report_status(tmp_path: Path) -> None:
    """Reuse one multi-join render across fresh console processes on every run."""
    if os.environ.get("YAP_STUDIO_ASR_TESTS") != "1":
        pytest.skip("set YAP_STUDIO_ASR_TESTS=1 to run speech engine tests")
    if not ParakeetOnnxBackend.available() or shutil.which("espeak-ng") is None:
        pytest.skip("onnx-asr and espeak-ng are required")
    source, words = synth.speech_source(tmp_path / "speech.mov", [
        "Bright planets circle slowly over quiet hills on calm winter evenings",
        "Seven green boxes wait beside the open door near our garden",
        "Morning light fills the empty kitchen before breakfast begins each day",
    ])
    spans = []
    for sentence in range(3):
        spoken = [word for word in words if word.sentence == sentence]
        spans.append((spoken[0].start - 0.2, spoken[-1].end + 0.2))
    final = synth.render_clean(source, spans, tmp_path / "final.mp4")
    edl = tmp_path / "edit.json"
    edl.write_text(json.dumps({
        "source": str(source),
        "segments": [{"start": start, "end": end} for start, end in spans],
    }), encoding="utf-8")
    command = [
        str(Path(sys.executable).parent / "yap-studio"),
        "verify", str(final), "--edl", str(edl), "--asr", "parakeet-onnx", "--json",
    ]
    runs = int(os.environ.get("YAP_STUDIO_ASR_EXIT_RUNS", "3"))
    assert runs > 0
    for index in range(runs):
        result = process.spawn(command, timeout_s=900, tool="verify")
        assert result.returncode in {0, 1}, (index, result.returncode, result.stderr)
        report = json.loads(result.stdout)
        assert result.returncode == exit_code(report, strict=False), (index, report)
        assert "libc++abi" not in result.stderr
        assert report["checks"]["join.words"]["ran"] is True
        assert len(report["cuts"]) == 2
        if index:
            assert "First run downloads the speech model" not in result.stderr
