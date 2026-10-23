"""HDR conversion selection, hardware capability, and cut receipt evidence."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import synth
from synth_broll import rotated
from test_cut import make_project, time_segment, write_edit

from yap_studio import ffmpeg, hdr
from yap_studio.hdr import HdrUnsupportedError, conversion, hdr_kind, videotoolbox_available
from yap_studio.media import sheet
from yap_studio.stages.cut import cut


@pytest.fixture(scope="module")
def media_info(tmp_path_factory: pytest.TempPathFactory) -> ffmpeg.MediaInfo:
    source = synth.calib_source(tmp_path_factory.mktemp("hdr-info") / "source.mov", seconds=1)
    return ffmpeg.probe(source)


@pytest.mark.parametrize(
    "transfer,kind", [(None, None), ("bt709", None), ("arib-std-b67", "hlg"), ("smpte2084", "pq")]
)
def test_hdr_transfer_detection(
    media_info: ffmpeg.MediaInfo, transfer: str | None, kind: str | None
) -> None:
    assert hdr_kind(replace(media_info, color_transfer=transfer)) == kind


def test_sdr_never_requires_conversion(
    media_info: ffmpeg.MediaInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(names: list[str]) -> dict:
        pytest.fail("SDR should not inspect HDR filter availability")

    monkeypatch.setattr(ffmpeg, "has_filters", unexpected)
    assert conversion(replace(media_info, color_transfer="bt709")) is None


@pytest.mark.parametrize(
    "pixel_format,download", [("yuv420p10le", "p010le"), ("p010le", "p010le"), ("yuv420p", "nv12")]
)
def test_macos_prefix_download_depth(
    media_info: ffmpeg.MediaInfo, monkeypatch: pytest.MonkeyPatch, pixel_format: str, download: str
) -> None:
    monkeypatch.setattr(hdr.sys, "platform", "darwin")
    monkeypatch.setattr(ffmpeg, "has_filters", lambda names: dict.fromkeys(names, True))
    result = conversion(replace(media_info, color_transfer="arib-std-b67", pix_fmt=pixel_format))
    assert result is not None
    assert result.method == "videotoolbox"
    assert result.input_args == [
        "-hwaccel",
        "videotoolbox",
        "-hwaccel_output_format",
        "videotoolbox_vld",
    ]
    assert result.filter_prefix.startswith("scale_vt=")
    assert f"hwdownload,format={download},format=yuv420p" in result.filter_prefix
    assert result.describe()["output_colour"] == "bt709"


@pytest.mark.parametrize(
    "available", [{"zscale": False, "tonemap": True}, {"zscale": True, "tonemap": False}]
)
def test_missing_portable_conversion_refuses_with_fix(
    media_info: ffmpeg.MediaInfo, monkeypatch: pytest.MonkeyPatch, available: dict[str, bool]
) -> None:
    monkeypatch.setattr(hdr.sys, "platform", "linux")
    monkeypatch.setattr(ffmpeg, "has_filters", lambda names: available)
    with pytest.raises(HdrUnsupportedError) as caught:
        conversion(replace(media_info, color_transfer="smpte2084"))
    assert caught.value.code == "hdr_unsupported"
    assert "zscale" in caught.value.suggested_action
    assert "macOS" in caught.value.suggested_action


def test_portable_conversion_prefix(
    media_info: ffmpeg.MediaInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hdr.sys, "platform", "linux")
    monkeypatch.setattr(ffmpeg, "has_filters", lambda names: dict.fromkeys(names, True))
    result = conversion(replace(media_info, color_transfer="smpte2084"))
    assert result is not None
    assert result.method == "zscale" and result.input_args == []
    assert result.filter_prefix == (
        "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
        "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
    )


@pytest.mark.parametrize("rotation,expected", [
    (0, ""), (90, "transpose=cclock"), (180, "hflip,vflip"), (270, "transpose=clock"),
])
def test_rotation_filter(rotation: int, expected: str) -> None:
    assert hdr.rotation_filter(rotation) == expected


@pytest.mark.parametrize("rotation", [-90, 45, 360])
def test_rotation_filter_rejects_unsupported_angles(rotation: int) -> None:
    with pytest.raises(ValueError, match="Unsupported display rotation"):
        hdr.rotation_filter(rotation)


@pytest.mark.parametrize("platform", ["darwin", "linux"])
@pytest.mark.parametrize("transfer", ["arib-std-b67", "smpte2084"])
@pytest.mark.parametrize("rotation,expected", [
    (0, ""), (90, "transpose=cclock"), (180, "hflip,vflip"), (270, "transpose=clock"),
])
def test_conversion_applies_rotation_after_colour(
    media_info: ffmpeg.MediaInfo, monkeypatch: pytest.MonkeyPatch,
    platform: str, transfer: str, rotation: int, expected: str,
) -> None:
    monkeypatch.setattr(hdr.sys, "platform", platform)
    monkeypatch.setattr(ffmpeg, "has_filters", lambda names: dict.fromkeys(names, True))
    result = conversion(replace(
        media_info, color_transfer=transfer, pix_fmt="yuv420p10le", rotation=rotation,
    ))
    assert result is not None
    if platform == "darwin":
        input_args = ["-hwaccel", "videotoolbox", "-hwaccel_output_format", "videotoolbox_vld"]
        colour = (
            "scale_vt=color_matrix=bt709:color_primaries=bt709:color_transfer=bt709,"
            "hwdownload,format=p010le,format=yuv420p"
        )
        assert result.method == "videotoolbox"
    else:
        input_args = []
        colour = (
            "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
            "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
        )
        assert result.method == "zscale"
    if rotation:
        input_args.extend(["-noautorotate", "-display_rotation", "0"])
    assert result.input_args == input_args
    assert result.filter_prefix == colour + (f",{expected}" if expected else "")
    assert result.rotation == rotation
    assert result.describe()["rotation"] == rotation


@pytest.mark.parametrize("rotation", [90, 180, 270])
@pytest.mark.parametrize("fallback", [False, True])
def test_sheet_rotates_once_with_hdr_or_decode_fallback(
    tmp_path: Path, media_info: ffmpeg.MediaInfo, monkeypatch: pytest.MonkeyPatch,
    rotation: int, fallback: bool,
) -> None:
    monkeypatch.setattr(hdr.sys, "platform", "darwin")
    monkeypatch.setattr(ffmpeg, "has_filters", lambda names: dict.fromkeys(names, True))
    width, height = (180, 320) if rotation in (90, 270) else (320, 180)
    info = replace(
        media_info, color_transfer="arib-std-b67", pix_fmt="yuv420p10le", rotation=rotation,
        width=width, height=height,
    )
    monkeypatch.setattr(ffmpeg, "probe", lambda source: info)
    commands = []

    def run(command: list[str], **kwargs: object) -> None:
        if command[-1].endswith(".png"):
            commands.append(command)
            expression = command[command.index("-vf") + 1]
            if fallback and "scale_vt=" in expression:
                raise OSError("hardware decode unavailable")
            Path(command[-1]).write_bytes(b"tile")

    monkeypatch.setattr(ffmpeg, "run", run)
    _, converted = sheet._render_clip(Path(info.path), {
        "hdr": "hlg", "rotation": rotation, "width": width, "height": height,
        "orientation": "vertical" if height > width else "horizontal",
        "duration_s": 1, "fps": "30/1",
    }, 1, tmp_path / "sheet.jpg")
    assert converted is (not fallback)
    assert len(commands) == (2 if fallback else 1)
    for command in commands:
        assert command[:command.index("-i")].count("-noautorotate") == 1
        expression = command[command.index("-vf") + 1]
        assert expression.count(hdr.rotation_filter(rotation)) == 1


def make_hlg(target: Path) -> Path:
    """Generate measured HLG tags on a small ten-bit test pattern."""
    ffmpeg.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=180x320:r=30:d=2",
            "-f",
            "lavfi",
            "-i",
            "anoisesrc=color=pink:amplitude=0.1:seed=13:sample_rate=48000:duration=2",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx265",
            "-preset",
            "ultrafast",
            "-x265-params",
            "pools=1:log-level=error:colorprim=bt2020:transfer=arib-std-b67:"
            "colormatrix=bt2020nc",
            "-pix_fmt",
            "yuv420p10le",
            "-colorspace",
            "bt2020nc",
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "arib-std-b67",
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
        timeout_s=60,
    )
    return target


def test_hdr_cut_refuses_without_conversion_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_project(tmp_path / "project", [make_hlg(tmp_path / "hdr.mov")], word_rows=[[]])
    write_edit(root, [time_segment("s1", "t1", 0.2, 1.6)])
    monkeypatch.setattr(hdr.sys, "platform", "linux")
    monkeypatch.setattr(ffmpeg, "has_filters", lambda names: dict.fromkeys(names, False))
    with pytest.raises(HdrUnsupportedError):
        cut(root)
    assert not (root / "cut").exists()


@pytest.mark.helpers
def test_hlg_cut_converts_on_videotoolbox(tmp_path: Path) -> None:
    if sys.platform != "darwin":
        pytest.skip("VideoToolbox requires macOS")
    if not videotoolbox_available():
        pytest.skip("VideoToolbox session cannot start")
    root = make_project(tmp_path / "project", [make_hlg(tmp_path / "hdr.mov")], word_rows=[[]])
    write_edit(root, [time_segment("s1", "t1", 0.2, 1.6)])
    result = cut(root)
    assert result["status"] == "verified", result
    info = ffmpeg.probe(root / "cut/cut.mp4")
    assert info.video_frames == 42
    assert (info.color_space, info.color_transfer, info.color_primaries) == ("bt709",) * 3
    stage_receipt = json.loads((root / "cut/receipt.json").read_bytes())
    assert stage_receipt["conversions"]["t1"]["method"] == "videotoolbox"
    assert stage_receipt["conversions"]["t1"]["kind"] == "hlg"


@pytest.fixture(scope="module")
def landscape_pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    if sys.platform != "darwin":
        pytest.skip("VideoToolbox requires macOS")
    if not videotoolbox_available():
        pytest.skip("VideoToolbox session cannot start")
    if not synth.display_rotation_supported():
        pytest.skip("display rotation is unsupported by this encoder")
    folder = tmp_path_factory.mktemp("hdr-rotation-source")
    picture = folder / "pattern.png"
    # Unequal boxes stay distinct even in the narrow landscape centre crop.
    pattern = (
        "color=c=black:s=320x180,"
        "drawbox=x=112:y=8:w=20:h=20:color=0x404040:t=fill,"
        "drawbox=x=180:y=50:w=20:h=30:color=0xe0e0e0:t=fill,"
        "drawbox=x=138:y=104:w=18:h=62:color=0xa0a0a0:t=fill,"
        "drawbox=x=173:y=136:w=22:h=20:color=0x686868:t=fill"
    )
    ffmpeg.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", pattern,
        "-frames:v", "1", "-threads", "1", str(picture),
    ], timeout_s=30)
    sources = []
    for kind in ("hlg", "sdr"):
        target = folder / f"{kind}.mov"
        command = [
            "ffmpeg", "-nostdin", "-v", "error", "-loop", "1", "-framerate", "30",
            "-i", str(picture), "-f", "lavfi", "-i",
            "anoisesrc=color=pink:amplitude=0.1:seed=13:sample_rate=48000:duration=1.2",
            "-map", "0:v:0", "-map", "1:a:0", "-t", "1.2",
        ]
        if kind == "hlg":
            command.extend([
                "-c:v", "libx265", "-preset", "ultrafast", "-x265-params",
                "pools=1:frame-threads=1:log-level=error:colorprim=bt2020:"
                "transfer=arib-std-b67:colormatrix=bt2020nc",
                "-pix_fmt", "yuv420p10le", "-colorspace", "bt2020nc",
                "-color_primaries", "bt2020", "-color_trc", "arib-std-b67",
            ])
        else:
            command.extend([
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            ])
        ffmpeg.run([*command, "-c:a", "pcm_s16le", str(target)], timeout_s=30)
        sources.append(target)
    return sources[0], sources[1]


def _gray_frame(path: Path, size: tuple[int, int] = (54, 96)) -> np.ndarray:
    width, height = size
    result = ffmpeg.run([
        "ffmpeg", "-nostdin", "-v", "error", "-noautorotate", "-ss", "0.2",
        "-i", str(path), "-map", "0:v:0", "-frames:v", "1",
        "-vf", f"scale={width}:{height}:flags=area,format=gray",
        "-threads", "1", "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ], timeout_s=30, binary=True)
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width)


@pytest.mark.helpers
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_rotated_hlg_cut_matches_sdr_autorotation(
    tmp_path: Path, landscape_pair: tuple[Path, Path], rotation: int,
) -> None:
    outputs = []
    for kind, original in zip(("hlg", "sdr"), landscape_pair, strict=True):
        source = rotated(original, tmp_path / f"{kind}.mov", rotation=rotation)
        source_info = ffmpeg.probe(source)
        assert source_info.rotation == rotation
        stored = json.loads(ffmpeg.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", str(source),
        ], timeout_s=30).stdout)["streams"][0]
        assert (stored["width"], stored["height"]) == (320, 180)
        assert (source_info.width, source_info.height) == (
            (180, 320) if rotation in (90, 270) else (320, 180)
        )
        root = make_project(tmp_path / f"{kind}-project", [source], word_rows=[[]])
        write_edit(root, [time_segment("s1", "t1", 0.1, 1.0)])
        result = cut(root)
        assert result["status"] == "verified", result
        output = root / "cut/cut.mp4"
        info = ffmpeg.probe(output)
        assert (info.width, info.height) == (1080, 1920)
        assert info.rotation == 0
        assert (info.color_space, info.color_transfer, info.color_primaries) == ("bt709",) * 3
        if kind == "hlg":
            assert source_info.color_transfer == "arib-std-b67"
            stage_receipt = json.loads((root / "cut/receipt.json").read_bytes())
            selected = stage_receipt["conversions"]["t1"]
            assert selected["method"] == "videotoolbox"
            assert selected["rotation"] == rotation
        outputs.append(_gray_frame(output))
    actual, reference = outputs
    correlation = float(np.corrcoef(actual.ravel(), reference.ravel())[0, 1])
    reversed_correlation = float(np.corrcoef(
        actual.ravel(), np.rot90(reference, 2).ravel(),
    )[0, 1])
    # Pattern half-turn correlations are below 0.03; leave room for colour-level changes.
    assert correlation > 0.9, correlation
    assert reversed_correlation < 0.5, reversed_correlation
