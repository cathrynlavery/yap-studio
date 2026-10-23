"""Extract an upright frame with the caller's colour conversion."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from yap_studio import ffmpeg, hdr
from yap_studio.errors import InputError


def extract_frame(source: Path, info: ffmpeg.MediaInfo, output: Path, *,
                  time_s: float | None, width: int, height: int,
                  conversion: hdr.HdrConversion | None, rotation: int | None = None) -> None:
    """Retry one frame earlier when a midpoint falls after the last timestamp."""
    prefix = f"{conversion.filter_prefix}," if conversion else ""
    rotate = hdr.rotation_filter(info.rotation if rotation is None else rotation)
    if rotate and not (conversion and conversion.rotation):
        prefix += f"{rotate},"
    input_args = list(conversion.input_args) if conversion else []
    if "-noautorotate" not in input_args:
        input_args.append("-noautorotate")
    step = float(1 / Fraction(info.fps or 30))
    seeks = (None,) if time_s is None else (time_s, max(0.0, time_s - step))
    stream = "0:v:0"
    if time_s is None and source.suffix.lower() in {".heic", ".heif"}:
        from yap_studio.media.catalog import heif_grid

        if heif_grid(source) is not None:
            stream = "0:g:0"
    jpeg = output.suffix.lower() == ".jpg"
    # JPEG readers use BT.601, so convert the numbers before encoding.
    colour = ":out_color_matrix=bt601:out_range=pc" if jpeg else ""
    filters = prefix + (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease{colour},"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )
    picture = (["-filter_complex", f"[{stream}]{filters}[out]", "-map", "[out]"]
               if stream == "0:g:0" else ["-map", stream, "-vf", filters])
    for seek in seeks:
        output.unlink(missing_ok=True)
        ffmpeg.run([
            "ffmpeg", "-v", "error", "-y", *input_args,
            *(["-noaccurate_seek"] if info.video_frames == 1 and seek is not None else []),
            *(["-ss", str(seek)] if seek is not None else []),
            "-i", str(source), *picture, "-frames:v", "1", "-threads", "1",
            *(["-q:v", "2", "-pix_fmt", "yuvj420p", "-colorspace", "bt470bg",
               "-color_primaries", "bt709", "-color_trc", "bt709"]
              if jpeg else []),
            "-update", "1", str(output),
        ], timeout_s=ffmpeg.DEFAULT_TIMEOUT_S)
        if output.is_file() and output.stat().st_size > 0:
            return
    raise InputError(str(source), "no frame decoded near the shot midpoint; try again")
