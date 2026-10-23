"""Share person coverage, silhouette and motion samples from one matte decode."""

from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

import numpy as np

from yap_studio import ffmpeg
from yap_studio.render.camera3d import HEIGHT, WIDTH, measure_silhouette
from yap_studio.render.camera3d_look import person_refusal


def sample_matte(matte: Path, frames: int,
                 shot_starts: Sequence[int]) -> tuple[np.ndarray, list[dict]]:
    """Check coverage at 5 fps and keep the pilot's 1 fps silhouette clock."""
    boundaries = sorted({0, frames, *(f for f in shot_starts if 0 < f < frames)})
    points = set(range(0, frames, 6)) | set(boundaries[:-1])
    for first, last in zip(boundaries[:-1], boundaries[1:], strict=True):
        if last - first < 6:
            points.add((first + last) // 2)
    selected = sorted(points)
    select = "not(mod(n\\,6))" + "".join(f"+eq(n\\,{f})" for f in selected if f % 6)
    command = ["ffmpeg", "-v", "error", "-i", str(matte), "-map", "0:v:0", "-an",
               "-vf", f"select=lt(n\\,{frames})*({select}),format=gray",
               "-fps_mode", "passthrough", "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"]
    pending = bytearray()
    size = WIDTH * HEIGHT
    sections, samples = [], []
    section_count = max(1, (frames + 15) // 30)
    decoded = 0
    with closing(ffmpeg.run_stream(command, timeout_s=120 + frames / 15)) as chunks:
        for chunk in chunks:
            pending.extend(chunk)
            while len(pending) >= size:
                if decoded >= len(selected):
                    raise ValueError("too many person matte samples")
                frame = selected[decoded]
                decoded += 1
                alpha = np.frombuffer(bytes(pending[:size]), dtype=np.uint8).reshape(HEIGHT, WIDTH)
                del pending[:size]
                if np.count_nonzero(alpha >= 32) / size < .005:
                    seconds = frame // 30
                    raise person_refusal(f"no person at {seconds // 60}:{seconds % 60:02}")
                if frame % 30 == 0 and frame // 30 < section_count:
                    summary = measure_silhouette(alpha)
                    sections.append({"start": frame / 30, "end": frame / 30 + 1, **summary})
                if frame % 6 == 0:
                    samples.append(np.rint(alpha.reshape(384, 5, 216, 5).mean(axis=(1, 3)))
                                   .astype(np.uint8))
    if pending or decoded != len(selected) or not sections or not samples:
        raise ValueError("could not sample complete person matte frames")
    sections[-1]["end"] = max(frames / 30, sections[-1]["end"])
    return np.stack(samples), sections
