"""Constant-rate validation against real packet presentation timestamps."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from yap_studio import ffmpeg
from yap_studio.errors import InputError


def timestamp_plate(
    path: Path, *, irregular: bool = True, width: int = 64, height: int = 96
) -> Path:
    """Delay one frame by half a frame without changing the advertised rates."""
    filters = "settb=1/60000"
    if irregular:
        filters += ",setpts=PTS+eq(N\\,4)*1000"
    ffmpeg.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         f"testsrc2=size={width}x{height}:rate=30", "-frames:v", "12", "-vf", filters,
         "-enc_time_base", "1:60000", "-fps_mode", "passthrough", "-c:v", "ffv1",
         str(path)], timeout_s=30,
    )
    return path


def test_irregular_packet_timestamps_with_matching_rates_are_refused(tmp_path: Path) -> None:
    from yap_studio.matte.timing import require_constant_rate

    plate = timestamp_plate(tmp_path / "irregular.mkv")
    info = ffmpeg.probe(plate)
    assert info.fps == info.nominal_fps == Fraction(30)
    assert not info.variable_frame_rate
    with pytest.raises(InputError, match="constant-rate"):
        require_constant_rate(plate, info)


def test_rounded_constant_rate_packet_timestamps_pass(tmp_path: Path) -> None:
    from yap_studio.matte.timing import require_constant_rate

    plate = timestamp_plate(tmp_path / "constant.mkv", irregular=False)
    require_constant_rate(plate, ffmpeg.probe(plate))


def test_reported_rate_mismatch_is_still_refused(tmp_path: Path) -> None:
    from yap_studio.matte.timing import require_constant_rate

    plate = timestamp_plate(tmp_path / "constant.mkv", irregular=False)
    with pytest.raises(InputError, match="constant-rate"):
        require_constant_rate(plate, replace(ffmpeg.probe(plate), nominal_fps=Fraction(60)))


@pytest.mark.parametrize("offset,refused", [("0.008333", False), ("0.008334", True)])
def test_quarter_frame_tolerance_and_sorted_pts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offset: str, refused: bool
) -> None:
    from yap_studio.matte.timing import require_constant_rate

    plate = timestamp_plate(tmp_path / "constant.mkv", irregular=False)
    info = ffmpeg.probe(plate)
    pts = [Fraction(2, 30), Fraction(0), Fraction(1, 30) + Fraction(offset)]
    result = ffmpeg.run(
        ["ffprobe", "-v", "error", "-show_entries", "packet=pts_time", "-of", "json",
         str(plate)], timeout_s=30,
    )

    def packets(command: list[str], **kwargs: object):
        assert command[command.index("-select_streams") + 1] == "v:0"
        assert command[command.index("-show_entries") + 1] == "packet=pts_time"
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({
            "packets": [{"pts_time": str(float(value))} for value in pts]
        }), stderr=result.stderr)

    monkeypatch.setattr(ffmpeg, "run", packets)
    if refused:
        with pytest.raises(InputError, match="constant-rate"):
            require_constant_rate(plate, info)
    else:
        require_constant_rate(plate, info)
