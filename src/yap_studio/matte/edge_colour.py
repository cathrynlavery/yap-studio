"""Remove old background colour from the visible rim of a person."""

from __future__ import annotations

import os
import selectors
import subprocess
import tempfile
import time
from collections.abc import Sequence
from contextlib import ExitStack, closing
from itertools import zip_longest
from pathlib import Path

import numpy as np

from yap_studio import ffmpeg, paths
from yap_studio.errors import ProcessingError
from yap_studio.matte.frames import decode_frames
from yap_studio.matte.refine import bilinear_at, box_mean, dilate, downsample

EDGE_KEEP_PX = 6  # Six pixels at a 1080-pixel short side preserve the mask's soft outer rim.


def background_matte_filter(width: int, height: int) -> str:
    """Firm translucent interior pixels while keeping the raw alpha near the outside."""
    radius = max(2, round(min(width, height) * EDGE_KEEP_PX / 1080))
    outside = "lut=c0='if(lte(val,5),255,0)'" + ",dilation" * radius
    return (
        "split=3[edge_raw][edge_firm][edge_outside];"
        "[edge_firm]lut=c0='clip((val-16)*255/80,0,255)',gblur=sigma=1[edge_solid];"
        f"[edge_outside]{outside}[edge_keep];"
        "[edge_solid][edge_raw][edge_keep]maskedmerge"
    )


def edge_zone(alpha: np.ndarray) -> np.ndarray:
    """Keep partial alpha near the exterior, excluding translucent interior patches."""
    radius = max(2, round(min(alpha.shape) / 100))
    return (alpha > 0) & (alpha < 255) & dilate(alpha <= 5, radius)


def _estimate(picture: np.ndarray, weights: np.ndarray, y: np.ndarray,
              x: np.ndarray) -> np.ndarray:
    """Pull weighted colour to coarse grids, then push estimates into empty regions."""
    factor = 4
    mass = downsample(weights.astype(np.float32), factor)
    colour = downsample(picture.astype(np.float32) * weights[..., None], factor)
    levels = []
    while True:
        blurred_mass, blurred_colour = box_mean(mass, 2), box_mean(colour, 2)
        levels.append((blurred_colour / np.maximum(blurred_mass[..., None], 1e-6),
                       blurred_mass > 1e-6))
        if max(mass.shape) == 1:
            break
        mass, colour = downsample(mass, 2), downsample(colour, 2)
    estimate, valid = levels.pop()
    if not valid.any():
        estimate[:] = picture.mean(axis=(0, 1))
    while levels:
        local, valid = levels.pop()
        yy, xx = np.nonzero(~valid)
        local[yy, xx] = bilinear_at(estimate, (yy + 0.5) / 2 - 0.5,
                                   (xx + 0.5) / 2 - 0.5)
        estimate = local
    return bilinear_at(estimate, (y + 0.5) / factor - 0.5, (x + 0.5) / factor - 0.5)


def clean_edge_colours(yuv: np.ndarray, alpha: np.ndarray,
                       zone: np.ndarray | None = None) -> np.ndarray:
    """Unmix the old background in affine Y'CbCr, touching only the exterior rim."""
    if zone is None:
        zone = edge_zone(alpha)
    if not zone.any():
        return yuv.copy()
    y, x = np.nonzero(zone)
    foreground = _estimate(yuv, alpha >= 250, y, x)
    background = _estimate(yuv, alpha <= 5, y, x)
    a = alpha[y, x, None].astype(np.float32) / 255
    unmixed = np.clip((yuv[y, x].astype(np.float32) - (1 - a) * background)
                      / np.maximum(a, 0.05), 0, 255)
    weight = np.clip((a - 0.05) / 0.45, 0, 1)
    result = yuv.copy()
    result[y, x] = np.round(weight * unmixed + (1 - weight) * foreground).astype(np.uint8)
    return result


def write_edge_layer(video: Path, matte: Path, first: int, count: int, out: Path, *,
                     commands: list[tuple[str, ...]] | None = None) -> Path | None:
    """Write a lossless edge window, or publish nothing when every edge zone is empty."""
    from yap_studio.matte.make import _stderr

    if first < 0 or count <= 0:
        raise ProcessingError("ffmpeg", 0, "edge layer needs a non-empty frame window",
                              code="matte_frame_count")
    info = ffmpeg.probe(video, count_frames=False)
    width, height = info.width, info.height
    colour, properties = [], []
    for flag, field, value in (("-color_primaries", "color_primaries", info.color_primaries),
                               ("-color_trc", "color_trc", info.color_transfer),
                               ("-colorspace", "colorspace", info.color_space),
                               ("-color_range", "range", info.color_range)):
        if value and value not in {"unknown", "unspecified"}:
            colour.extend([flag, value])
            properties.append(f"{field}={value}")
    if properties:
        colour.extend(["-vf", "setparams=" + ":".join(properties)])
    deadline = time.monotonic() + 120 + count
    with paths.operation():
        paths.protect_input(video)
        paths.protect_input(matte)
        target = paths.validate_output(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".yap_tmp_", dir=target.parent) as folder, \
                tempfile.TemporaryFile() as stderr:
            temporary = Path(folder) / target.name
            command = ffmpeg._safe_command([
                "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuva444p",
                "-s", f"{width}x{height}", "-framerate", str(info.fps), "-i", "pipe:0",
                "-an", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuva444p",
                *colour, "-fflags", "+bitexact", "-flags:v", "+bitexact", str(temporary),
            ])
            if commands is not None:
                commands.append(tuple(command))
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=stderr, bufsize=0)
            assert encoder.stdin is not None

            def check() -> None:
                if encoder.poll() not in (None, 0):
                    raise ProcessingError(" ".join(command), encoder.returncode, _stderr(stderr))

            try:
                os.set_blocking(encoder.stdin.fileno(), False)
                with ExitStack() as stack, selectors.DefaultSelector() as ready:
                    pictures = stack.enter_context(closing(decode_frames(video, width, height,
                        deadline, first=first, count=count, planes=3,
                        commands=commands, check=check)))
                    masks = stack.enter_context(closing(decode_frames(matte, width, height,
                        deadline, first=first, count=count, commands=commands, check=check)))
                    ready.register(encoder.stdin, selectors.EVENT_WRITE)
                    used = 0
                    has_edge = False
                    for picture, alpha in zip_longest(pictures, masks):
                        if picture is None or alpha is None:
                            raise ProcessingError("ffmpeg", 0, "edge video and matte frame counts "
                                                  "differ", code="matte_frame_count")
                        zone = edge_zone(alpha)
                        has_edge = has_edge or bool(zone.any())
                        cleaned = clean_edge_colours(picture, alpha, zone=zone)
                        layer = np.empty((4, height, width), dtype=np.uint8)
                        layer[0], layer[1:3], layer[3] = 16, 128, zone.astype(np.uint8) * 255
                        layer[:3, zone] = cleaned[zone].T
                        raw = memoryview(layer).cast("B")
                        while raw:
                            check()
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise subprocess.TimeoutExpired(command, 0)
                            if ready.select(min(remaining, 0.05)):
                                try:
                                    size = os.write(encoder.stdin.fileno(), raw)
                                except BlockingIOError:
                                    continue
                                raw = raw[size:]
                        used += 1
                    if used != count:
                        raise ProcessingError("ffmpeg", 0,
                            f"edge layer expected {count} frames, decoded {used}",
                            code="matte_frame_count")
                encoder.stdin.close()
                encoder.wait(timeout=max(0, deadline - time.monotonic()))
                check()
            except BrokenPipeError:
                encoder.wait(timeout=max(0, deadline - time.monotonic()))
                raise ProcessingError(" ".join(command), encoder.returncode, _stderr(stderr)) \
                    from None
            except subprocess.TimeoutExpired:
                raise ProcessingError(" ".join(command), -1, "edge layer timed out",
                                      code="matte_timeout") from None
            finally:
                encoder.stdin.close()
                if encoder.poll() is None:
                    encoder.kill()
                encoder.wait()
            if has_edge:
                with paths.atomic_output(target) as published:
                    os.replace(temporary, published)
                return target
    return None


def prepare_edge_layers(video: Path, matte: Path, windows: Sequence[tuple[int, int]],
                        work_dir: Path, *, commands: list[tuple[str, ...]] | None = None,
                        frames: tuple[int, int] | None = None,
                        ) -> list[tuple[int, int, Path]]:
    """Prepare one layer per half-open output window that contains edge pixels."""
    result = []
    for first, last in dict.fromkeys(windows):
        if frames is not None:
            first, last = max(first, frames[0]), min(last, frames[1] + 1)
        if last <= first:
            continue
        target = work_dir / f"edge-{first}-{last - first}.mkv"
        index = 1
        while target.exists():
            target = work_dir / f"edge-{first}-{last - first}-{index}.mkv"
            index += 1
        layer = write_edge_layer(video, matte, first, last - first, target, commands=commands)
        if layer is not None:
            result.append((first, last - first, layer))
    return result
