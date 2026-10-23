"""Convert renderer alpha clips through RGB to tagged BT.709."""

from __future__ import annotations

import re
from pathlib import Path

from yap_studio import ffmpeg, paths
from yap_studio.errors import ProcessingError, ValidationError

_BT709_FILTER = (
    "scale=in_color_matrix={input_matrix}:in_range=tv:out_range=pc:"
    "flags=accurate_rnd+full_chroma_int,format=rgba64le,"
    "scale=out_color_matrix=bt709:out_range=tv:"
    "flags=accurate_rnd+full_chroma_int,format=yuva444p10le,"
    "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
)


def _require_frames(info: ffmpeg.MediaInfo, frames: int) -> None:
    if info.video_frames != frames:
        error = ProcessingError(
            "verify alpha clip",
            -1,
            f"Expected {frames} frames, got {info.video_frames}.",
            code="layer_frame_mismatch",
        )
        error.suggested_action = "Correct the composition duration and render the layer again."
        raise error


def convert_to_bt709(src: Path, out: Path, *, frames: int, timeout_s: float | None = None) -> Path:
    """Atomically publish a BT.709 ProRes 4444 clip after verifying frames and tags."""
    if frames < 1:
        raise ValueError("frames must be positive")
    if out.suffix.lower() != ".mov":
        raise ValidationError("out", "alpha clips require a .mov suffix")
    with paths.operation():
        source = paths.validate_input(src)
        info = ffmpeg.probe(source)
        _require_frames(info, frames)
        if info.color_space == "bt709":
            input_matrix = "bt709"
        elif info.color_space in {None, "unknown", "bt470bg", "smpte170m"}:
            input_matrix = "bt601"
        else:
            error = ProcessingError(
                "convert alpha clip",
                -1,
                f"Unsupported source colour matrix: {info.color_space}.",
                code="layer_colour_tags",
            )
            error.suggested_action = "Render the layer with a BT.709 or BT.601 colour matrix."
            raise error
        deadline = timeout_s
        if deadline is None:
            duration = info.video_duration_s or info.duration_s or frames / 30
            deadline = 120 + duration * (info.width or 1) * (info.height or 1) / 2e6
        with paths.atomic_output(out, allowed_existing_suffixes=frozenset()) as temporary:
            ffmpeg.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(source),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-vf",
                    _BT709_FILTER.format(input_matrix=input_matrix),
                    "-c:v",
                    "prores_ks",
                    "-profile:v",
                    "4444",
                    "-pix_fmt",
                    "yuva444p10le",
                    "-vendor",
                    "apl0",
                    str(temporary),
                ],
                timeout_s=deadline,
            )
            converted = ffmpeg.probe(temporary)
            _require_frames(converted, frames)
            tags = (converted.color_space, converted.color_transfer, converted.color_primaries)
            if tags != ("bt709", "bt709", "bt709") or not re.fullmatch(
                r"yuva444p1\dle", converted.pix_fmt or ""
            ):
                error = ProcessingError(
                    "verify alpha clip",
                    -1,
                    f"Expected BT.709 alpha, got tags {tags} and format {converted.pix_fmt}.",
                    code="layer_colour_tags",
                )
                error.suggested_action = "Check ffmpeg's ProRes 4444 support and render again."
                raise error
    return out
