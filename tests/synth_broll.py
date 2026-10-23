"""Tiny synthetic clips and stills for catalog tests."""

from __future__ import annotations

import struct
from pathlib import Path

import synth

from yap_studio import ffmpeg

display_rotation_supported = synth.display_rotation_supported


def clip(
    out: Path, *, size: str = "160x90", audio: bool = True,
    camera: bool = False, colour: str = "blue",
    duration: float = 2.0, fps: int = 30,
) -> Path:
    """Build a short clip with optional audio and QuickTime camera tags."""
    command = [
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        f"color=c={colour}:s={size}:r={fps}:d={duration}",
    ]
    if audio:
        command.extend([
            "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={duration}",
        ])
    command.extend(["-map", "0:v:0"])
    if audio:
        command.extend(["-map", "1:a:0"])
    command.extend(["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"])
    if audio:
        command.extend(["-c:a", "aac"])
    if camera:
        command.extend([
            "-movflags", "use_metadata_tags",
            "-metadata", "com.apple.quicktime.make=Example Camera",
            "-metadata", "com.apple.quicktime.model=Pocket Model",
            "-metadata", "com.apple.quicktime.location.ISO6709=+01.0000+002.0000/",
        ])
    command.append(str(out))
    synth._run(command, timeout=30)
    return out


def png(out: Path, *, size: str = "80x80") -> Path:
    """Build a small PNG still."""
    return _still(out, "png", size)


def jpeg(out: Path, *, size: str = "80x80", camera: bool = False) -> Path:
    """Build a small JPEG still, optionally with synthetic EXIF make and model."""
    _still(out, "mjpeg", size)
    if camera:
        make, model = b"Example Camera\x00", b"Pocket Model\x00"
        # A little-endian TIFF IFD with two ASCII entries in a JPEG APP1 segment.
        offset = 8 + 2 + 2 * 12 + 4
        tiff = b"II" + struct.pack("<HIH", 42, 8, 2)
        tiff += struct.pack("<HHII", 0x010F, 2, len(make), offset)
        tiff += struct.pack("<HHII", 0x0110, 2, len(model), offset + len(make))
        tiff += struct.pack("<I", 0) + make + model
        exif = b"Exif\x00\x00" + tiff
        picture = out.read_bytes()
        out.write_bytes(picture[:2] + b"\xff\xe1" + struct.pack(
            ">H", len(exif) + 2
        ) + exif + picture[2:])
    return out


def _still(out: Path, codec: str, size: str) -> Path:
    synth._run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        f"color=c=green:s={size}", "-map", "0:v:0", "-frames:v", "1", "-c:v", codec,
        "-threads", "1", str(out),
    ], timeout=30)
    return out


def top_bottom_clip(out: Path) -> Path:
    """Build an asymmetric white-top, black-bottom picture for rotation comparisons."""
    ffmpeg.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        "color=c=black:s=160x90:r=30:d=2,drawbox=x=0:y=0:w=iw:h=ih/2:color=white:t=fill",
        "-map", "0:v:0", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        str(out),
    ], timeout_s=30)
    return out


def hlg(out: Path, *, size: str = "64x64") -> Path:
    """Build a tiny HEVC clip with HLG transfer metadata."""
    synth._run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        f"testsrc2=s={size}:r=30:d=0.2", "-c:v", "libx265", "-preset", "ultrafast",
        "-map", "0:v:0",
        "-x265-params", "pools=1:frame-threads=1:log-level=error:colorprim=bt2020:"
        "transfer=arib-std-b67:colormatrix=bt2020nc",
        "-pix_fmt", "yuv420p10le", "-color_trc", "arib-std-b67",
        "-color_primaries", "bt2020", "-colorspace", "bt2020nc", str(out),
    ], timeout=30)
    return out


def audio_only(out: Path) -> Path:
    """Build a short AAC file without a video stream."""
    synth._run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        "sine=frequency=440:duration=0.2", "-map", "0:a:0", "-c:a", "aac", str(out),
    ], timeout=30)
    return out


def rotated(source: Path, out: Path, *, rotation: int = 90) -> Path:
    """Copy a clip and set its display rotation in degrees."""
    synth._run([
        "ffmpeg", "-v", "error", "-y", "-display_rotation", str(rotation),
        "-i", str(source), "-map", "0", "-c", "copy", str(out),
    ], timeout=30)
    return out
