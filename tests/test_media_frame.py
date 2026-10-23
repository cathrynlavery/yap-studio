"""Synthetic colour checks for shared frame extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import synth

from yap_studio import ffmpeg
from yap_studio.media.frame import extract_frame


@pytest.mark.parametrize("matrix", ["bt470bg", "bt709"])
def test_full_range_jpeg_preserves_colours(tmp_path: Path, matrix: str) -> None:
    calibration = synth.calib_source(tmp_path / "calibration.mov", seconds=1)
    source = synth.full_range_source(calibration, tmp_path / "source.mov", matrix=matrix)
    output = tmp_path / "frame.jpg"
    extract_frame(source, ffmpeg.probe(source), output, time_s=0, width=320, height=180,
                  conversion=None)
    assert synth._stream(output, "video")["color_range"] == "pc"
    np.testing.assert_allclose(synth.patch_colours(output).astype(int),
                               synth.patch_colours(source).astype(int), atol=3)
