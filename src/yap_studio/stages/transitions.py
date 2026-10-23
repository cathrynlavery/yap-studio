"""Render content-keyed shader windows from exact frames of the verified cut."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from yap_studio import ffmpeg, paths, receipt
from yap_studio.errors import YapError
from yap_studio.project.receipts import projection_sha256
from yap_studio.render import hyperframes
from yap_studio.stages.timeline import ResolvedTimeline

FPS = 30
LOOKS = frozenset({"whip_pan", "cinematic_zoom", "light_leak"})
TEMPLATES = Path(__file__).parents[1] / "templates"


class TransitionError(YapError):
    """Refuse incomplete frame extraction, cache entries or rendered windows."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_type="transition_error", code="transition_invalid",
                         suggested_action="Correct the transition inputs and assemble again.")


def template_file(look: str) -> Path:
    """Return an engine-owned template, refusing arbitrary composition paths."""
    if look not in LOOKS:
        raise TransitionError(f"Unknown transition look: {look!r}.")
    return TEMPLATES / f"transition_{look}" / "template.html"


def _extract(cut: Path, directory: Path, first: int, frames: int) -> list[Path]:
    """Select exact decoded frame indices without seeking, scaling or duplication."""
    directory.mkdir()
    ffmpeg.run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(cut), "-map", "0:v:0", "-an",
        "-vf", f"select=between(n\\,{first}\\,{first + frames - 1})",
        "-fps_mode", "passthrough", "-start_number", "0", "-threads", "2",
        str(directory / "%03d.png"),
    ], timeout_s=max(ffmpeg.DEFAULT_TIMEOUT_S, 30 + (first + frames) / FPS * 12))
    expected = [directory / f"{index:03d}.png" for index in range(frames)]
    if sorted(directory.glob("*.png")) != expected:
        raise TransitionError(f"Extracted {len(list(directory.glob('*.png')))} frames; "
                              f"expected {frames} starting at cut frame {first}.")
    return expected


def _assert_layer(path: Path, frames: int) -> None:
    """Check the opaque full-canvas movie before caching or compositing it."""
    info = ffmpeg.probe(path)
    if ((info.width, info.height, info.video_frames, info.fps) != (1080, 1920, frames, FPS)
            or (info.color_primaries, info.color_transfer, info.color_space) != ("bt709",) * 3
            or info.has_audio):
        raise TransitionError("Transition movie must be silent BT.709, 1080x1920 at 30 fps "
                              f"with exactly {frames} frames.")


def _cached(folder: Path, key: str, frames: int) -> Path | None:
    """Accept only an atomically published cache entry with its current movie hash."""
    if not folder.exists():
        return None
    movie = paths.validate_input(folder / "transition.mov")
    try:
        record = json.loads((folder / "receipt.json").read_bytes())
        valid = (record["schema"] == "yap-studio/transition-cache@1" and record["key"] == key
                 and record["sha256"] == receipt.sha256_file(movie))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise TransitionError(f"Incomplete transition cache: {folder}.") from exc
    if not valid:
        raise TransitionError(f"Transition cache identity changed: {folder}.")
    _assert_layer(movie, frames)
    return movie


def render_transitions(project_dir: Path, timeline: ResolvedTimeline, cut: Path,
                       work_dir: Path) -> list[dict[str, Any]]:
    """Extract both join halves and return cached, exact-window movie placements."""
    items = [item for item in timeline.elements if item.kind == "transition"]
    if not items:
        return []
    with paths.operation():
        root = Path(project_dir).resolve()
        source = paths.validate_input(cut)
        info = ffmpeg.probe(source)
        if (info.width, info.height, info.video_frames, info.fps) != (
            1080, 1920, timeline.total_frames, FPS,
        ):
            raise TransitionError("Verified cut dimensions or frame count disagree with timeline.")
        work = paths.validate_output_dir(work_dir)
        cache = paths.validate_output_dir(root / ".yap/cache/transitions")
        placements = []
        for item in items:
            look, frames = item.payload["look"], item.payload.get("frames", 12)
            join = item.payload["join_frame"]
            if (isinstance(frames, bool) or not isinstance(frames, int)
                    or frames not in range(6, 25, 2)
                    or item.start_frame != join - frames // 2
                    or item.end_frame != join + frames // 2
                    or item.start_frame <= 0 or item.end_frame >= timeline.total_frames):
                raise TransitionError(f"{item.id}: invalid transition frame window.")
            template = paths.validate_input(template_file(look))
            with tempfile.TemporaryDirectory(prefix="transition-", dir=work) as temporary:
                composition = Path(temporary)
                outgoing = _extract(source, composition / "from", item.start_frame, frames // 2)
                incoming = _extract(source, composition / "to", join, frames // 2)
                key = projection_sha256({
                    "look": look, "frames": frames, "fps": FPS,
                    "renderer": hyperframes.HYPERFRAMES_VERSION,
                    "workers": hyperframes.RENDER_WORKERS,
                    "capture": hyperframes.RENDER_CAPTURE,
                    "template": receipt.sha256_file(template),
                    "images": [receipt.sha256_file(frame) for frame in [*outgoing, *incoming]],
                })
                folder = cache / key
                movie = _cached(folder, key, frames)
                if movie is None:
                    html = template.read_text(encoding="utf-8").replace(
                        'data-duration="0.399"',
                        f'data-duration="{hyperframes.layer_duration_s(frames, FPS)}"',
                    )
                    paths.atomic_write_text(composition / "index.html", html)
                    rendered = composition / "transition.mov"
                    hyperframes.render_layer(composition, rendered, frames=frames, fps=FPS,
                                             variables={"frames": frames, "fps": FPS})
                    _assert_layer(rendered, frames)
                    with tempfile.TemporaryDirectory(prefix=".transition-", dir=cache) as staged:
                        staging = Path(staged)
                        shutil.copyfile(rendered, staging / "transition.mov")
                        paths.atomic_write_text(staging / "receipt.json", json.dumps({
                            "schema": "yap-studio/transition-cache@1", "key": key,
                            "sha256": receipt.sha256_file(rendered),
                        }) + "\n")
                        try:
                            os.rename(staging, folder)
                        except OSError:
                            if not folder.is_dir():
                                raise
                        movie = _cached(folder, key, frames)
                        assert movie is not None
                placements.append({"id": item.id, "path": str(movie),
                                   "start_frame": item.start_frame, "frames": frames})
        return placements
