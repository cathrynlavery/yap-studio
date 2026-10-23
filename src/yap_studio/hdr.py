"""Shared HDR conversion to the cut's BT.709 picture contract."""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from yap_studio import ffmpeg
from yap_studio.errors import YapError

_VIDEOTOOLBOX_DOWNLOAD_FORMATS = {
    "yuv420p": "nv12", "yuvj420p": "nv12", "nv12": "nv12",
    "yuv420p10le": "p010le", "yuv420p10be": "p010le",
    "p010le": "p010le", "p010be": "p010le",
    "yuv422p": "nv16", "yuvj422p": "nv16", "nv16": "nv16",
    "yuv422p10le": "p210le", "yuv422p10be": "p210le",
    "p210le": "p210le", "p210be": "p210le", "nv20le": "p210le", "nv20be": "p210le",
    "yuv444p": "nv24", "yuvj444p": "nv24", "nv24": "nv24",
    "yuv444p10le": "p410le", "yuv444p10be": "p410le",
    "p410le": "p410le", "p410be": "p410le",
}


class HdrUnsupportedError(YapError):
    """Refuse HDR when the installed media tools cannot convert it."""

    def __init__(self, *, pixel_format: str | None = None) -> None:
        super().__init__(
            (f"HDR pixel format {pixel_format!r} has no supported VideoToolbox download format."
             if pixel_format is not None else "HDR conversion to BT.709 is unavailable."),
            error_type="dependency_error",
            code="hdr_unsupported",
            suggested_action=(
                "Transcode the HDR source to yuv420p10le, or convert it to SDR BT.709, and retry."
                if pixel_format is not None
                else "Install an ffmpeg build with zscale and tonemap, or run on macOS."
            ),
        )


@dataclass(frozen=True)
class HdrConversion:
    """Input options and first filters that produce an upright BT.709 picture."""

    method: str
    input_args: list[str]
    filter_prefix: str
    rotation: int = 0

    def describe(self) -> dict[str, Any]:
        """Return the conversion evidence retained in a stage receipt."""
        return {
            "method": self.method,
            "input_args": list(self.input_args),
            "filter_prefix": self.filter_prefix,
            "rotation": self.rotation,
            "output_colour": "bt709",
        }


def rotation_filter(rotation: int) -> str:
    """Match ffmpeg autorotation for a probed display rotation."""
    filters = {0: "", 90: "transpose=cclock", 180: "hflip,vflip", 270: "transpose=clock"}
    if rotation not in filters:
        raise ValueError(f"Unsupported display rotation: {rotation}")
    return filters[rotation]


def hdr_kind(info: ffmpeg.MediaInfo) -> str | None:
    """Identify HLG and PQ from the source transfer function."""
    return {"arib-std-b67": "hlg", "smpte2084": "pq"}.get(info.color_transfer or "")


def _filters(names: list[str], commands: list[tuple[str, ...]] | None) -> dict[str, bool]:
    """Keep the capability probe when an assembly caller is building a command receipt."""
    available = ffmpeg.has_filters(names)
    if commands is not None:
        commands.append((ffmpeg.find_tool("ffmpeg"), "-nostdin", "-hide_banner", "-filters"))
    return available


def _videotoolbox_download_format(pixel_format: str | None) -> str:
    """Match both chroma subsampling and bit depth to the hardware surface layout."""
    download = _VIDEOTOOLBOX_DOWNLOAD_FORMATS.get(pixel_format or "")
    if download is None:
        raise HdrUnsupportedError(pixel_format=pixel_format or "unknown")
    return download


def conversion(info: ffmpeg.MediaInfo, *, commands: list[tuple[str, ...]] | None = None
               ) -> HdrConversion | None:
    """Choose a conversion to upright BT.709, or refuse an unsupported HDR input."""
    if hdr_kind(info) is None:
        return None
    rotate = rotation_filter(info.rotation)
    input_args = ["-noautorotate", "-display_rotation", "0"] if info.rotation else []
    suffix = f",{rotate}" if rotate else ""
    if sys.platform == "darwin":
        download = _videotoolbox_download_format(info.pix_fmt)
        if not _filters(["scale_vt"], commands).get("scale_vt", False):
            raise HdrUnsupportedError()
        return HdrConversion(
            "videotoolbox",
            ["-hwaccel", "videotoolbox", "-hwaccel_output_format", "videotoolbox_vld",
             *input_args],
            "scale_vt=color_matrix=bt709:color_primaries=bt709:color_transfer=bt709,"
            f"hwdownload,format={download},format=yuv420p{suffix}",
            rotation=info.rotation,
        )
    filters = _filters(["zscale", "tonemap"], commands)
    if not all(filters.get(name, False) for name in ("zscale", "tonemap")):
        raise HdrUnsupportedError()
    return HdrConversion(
        "zscale",
        input_args,
        "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
        f"tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p{suffix}",
        rotation=info.rotation,
    )


@lru_cache(maxsize=1)
def videotoolbox_available() -> bool:
    """Try one hardware-decoded HLG frame once, including a hardware download."""
    if sys.platform != "darwin":
        return False
    try:
        with tempfile.TemporaryDirectory(prefix="yap-hdr-") as temporary:
            source = Path(temporary) / "probe.mp4"
            ffmpeg.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=s=64x64:r=30:d=0.1",
                    "-map",
                    "0:v:0",
                    "-frames:v",
                    "1",
                    "-c:v",
                    "libx265",
                    "-x265-params",
                    "pools=1:log-level=error:colorprim=bt2020:transfer=arib-std-b67:"
                    "colormatrix=bt2020nc",
                    "-pix_fmt",
                    "yuv420p10le",
                    "-color_trc",
                    "arib-std-b67",
                    "-color_primaries",
                    "bt2020",
                    "-colorspace",
                    "bt2020nc",
                    str(source),
                ],
                timeout_s=30,
            )
            selected = conversion(ffmpeg.probe(source))
            assert selected is not None
            ffmpeg.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    *selected.input_args,
                    "-i",
                    str(source),
                    "-map",
                    "0:v:0",
                    "-vf",
                    selected.filter_prefix,
                    "-frames:v",
                    "1",
                    "-f",
                    "null",
                    "-",
                ],
                timeout_s=30,
            )
        return True
    except (YapError, OSError):
        return False
