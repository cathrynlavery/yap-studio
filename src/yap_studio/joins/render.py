"""Encode and atomically cache silent, lossless smooth join windows."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from yap_studio import ffmpeg, paths, process, receipt
from yap_studio.joins import helper, plan
from yap_studio.joins.helper import Interpolator, JoinError
from yap_studio.joins.plan import JoinDecision, JoinPlan, build_plan
from yap_studio.project.receipts import projection_sha256
from yap_studio.stages.timeline import ResolvedTimeline, TimelineRefusal

VERSION = 2
# Bumping this reruns smooth assemblies to check cards behind the speaker.
BEHIND_CARD_JOIN_GUARD_VERSION = 1


def identity(moves: dict) -> dict[str, Any]:
    """Identify smooth picture work without invoking the processor."""
    return {"mode": "smooth", "fallback": plan.fallback(moves),
            "window_frames": plan.WINDOW_FRAMES, "limits": plan.limits(), "version": VERSION,
            "behind_card_join_guard": BEHIND_CARD_JOIN_GUARD_VERSION,
            "helper": helper.source_digest(), "macos": platform.mac_ver()[0]}


def _assert_layer(movie: Path, frames: int) -> None:
    info = ffmpeg.probe(movie)
    if ((info.width, info.height, info.video_frames, info.fps) != (1080, 1920, frames, 30)
            or (info.color_primaries, info.color_transfer, info.color_space) != ("bt709",) * 3
            or info.has_audio):
        raise JoinError("Smooth join movie must be silent BT.709, 1080x1920 at 30 fps "
                        f"with exactly {frames} frames; found {info}.")


def _cached(folder: Path, key: str, frames: int) -> Path | None:
    if not folder.exists():
        return None
    movie = folder / "window.mp4"
    try:
        record = json.loads((folder / "receipt.json").read_bytes())
        valid = (record["schema"] == "yap-studio/smooth-join-cache@1" and record["key"] == key
                 and record["sha256"] == receipt.sha256_file(paths.validate_input(movie)))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise JoinError(f"Incomplete smooth join cache: {folder}.") from exc
    if not valid:
        raise JoinError(f"Smooth join cache identity changed: {folder}.")
    _assert_layer(movie, frames)
    return movie


def _render_window(cache: Path, join: JoinDecision, before: bytes, after: bytes, *,
                   width: int, height: int, phases: list[float],
                   interpolator: Interpolator) -> Path:
    """Release one window's generated frames before reading the next endpoint pair."""
    key = projection_sha256({
        "before": hashlib.sha256(before).hexdigest(), "after": hashlib.sha256(after).hexdigest(),
        "phases": phases, "window_frames": plan.WINDOW_FRAMES,
        "helper": helper.source_digest(), "macos": platform.mac_ver()[0],
        "ffmpeg": ffmpeg.tool_versions()["ffmpeg"], "version": VERSION,
    })
    folder = cache / key
    movie = _cached(folder, key, join.frames)
    if movie is not None:
        return movie
    frames = interpolator.interpolate(before, after, width=width, height=height, phases=phases)
    size = width * height * 6
    if len(frames) != join.frames or any(len(frame) != size for frame in frames):
        raise JoinError("Smooth join interpolator returned the wrong frame count or size.")
    with tempfile.TemporaryDirectory(prefix=".smooth-", dir=cache) as temporary:
        staging = Path(temporary)
        rendered = staging / "window.mp4"
        command = ffmpeg._safe_command([
            "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb48le",
            "-s", f"{width}x{height}", "-r", "30", "-i", "pipe:0",
            "-vf", "scale=out_color_matrix=bt709:out_range=tv:"
            "flags=accurate_rnd+full_chroma_int,format=yuv420p,"
            "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=limited",
            "-c:v", "libx264", "-qp", "0", "-preset", "veryfast", "-bf", "0",
            "-threads", "2", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-colorspace", "bt709", "-color_range", "tv", "-an", str(rendered),
        ])
        result = process.spawn(command, input_bytes=b"".join(frames), timeout_s=120, tool="ffmpeg")
        if result.returncode:
            raise JoinError(f"Smooth join movie encoding failed: {result.stderr}")
        _assert_layer(rendered, join.frames)
        paths.atomic_write_text(staging / "receipt.json", json.dumps({
            "schema": "yap-studio/smooth-join-cache@1", "key": key,
            "sha256": receipt.sha256_file(rendered),
        }) + "\n")
        try:
            os.rename(staging, folder)
        except OSError:
            if not folder.is_dir():
                raise
        movie = _cached(folder, key, join.frames)
        assert movie is not None
    return movie


def _endpoint(reader: Iterator[tuple[int, bytes]], expected: int) -> bytes:
    index, raw = next(reader, (None, b""))
    if index != expected:
        raise JoinError("Smooth join frame extraction returned an incomplete frame sequence.")
    return raw


def render_windows(project_dir: Path, decisions: JoinPlan, *, cut: Path | None = None,
                   interpolator: Interpolator | None = None) -> list[dict[str, Any]]:
    """Stream one smooth window at a time and reuse content-validated movies."""
    joins = [join for join in decisions.joins if join.handling == "smooth"]
    if not joins:
        return []
    interpolator = interpolator if interpolator is not None else helper.AppleInterpolator()
    source = cut or Path(project_dir) / "cut/cut.mp4"
    indices = sorted({index for join in joins
                      for index in (join.start_frame - 1, join.start_frame + join.frames)})
    # Allow 120 seconds each for interpolation and encoding while the decoder is paused.
    with paths.operation(), closing(plan.decode_frames(
            source, indices, width=decisions.width, height=decisions.height,
            extra_timeout_s=240 * len(joins))) as reader:
        cache = paths.validate_output_dir(Path(project_dir) / ".yap/cache/smooth-joins")
        phases = [k / (plan.WINDOW_FRAMES + 1) for k in range(1, plan.WINDOW_FRAMES + 1)]
        placements = []
        last_index, last_frame = -1, None
        for join in joins:
            a_index, b_index = join.start_frame - 1, join.start_frame + join.frames
            before = last_frame if last_index == a_index else None
            last_frame = None
            if before is None:
                before = _endpoint(reader, a_index)
            after = _endpoint(reader, b_index)
            movie = _render_window(cache, join, before, after, width=decisions.width,
                                   height=decisions.height, phases=phases,
                                   interpolator=interpolator)
            placements.append({"id": f"smooth-{join.index}", "path": str(movie),
                               "start_frame": join.start_frame, "frames": join.frames,
                               "join_frame": join.join_frame})
            last_index, last_frame = b_index, after
            del before, after
        if next(reader, None) is not None:
            raise JoinError("Smooth join frame extraction returned an extra frame sequence.")
        return placements


def render_joins(project_dir: Path, timeline: ResolvedTimeline, cut: Path,
                 moves: dict, *, placements: Sequence[dict] = (),
                 ) -> tuple[list[dict[str, Any]], JoinPlan | None]:
    """Plan from retained contracts and render only when the style chooses smooth."""
    if timeline.camera_3d or moves.get("jump_cuts", "hard") != "smooth":
        return [], None
    record = json.loads((cut.parent / "receipt.json").read_bytes())
    pieces = record.get("plan", {}).get("pieces")
    if pieces is None:
        raise JoinError("The cut receipt has no pieces for jump cuts.",
                        fix="Run yap-studio cut DIR, then assemble again.")
    words = json.loads((cut.parent / "cut-words.json").read_bytes())["words"]
    decisions = build_plan(project_dir, timeline, pieces, words, moves, cut=cut)
    placed = {item["id"]: item for item in placements}
    for item in timeline.elements:
        placement = placed.get(item.id, {})
        if (item.kind == "graphic" and placement.get("layer", item.layer) == "behind"
                and item.payload.get("template") != "reel_feed" and any(
                    join.handling == "smooth"
                    and max(item.start_frame, join.start_frame)
                    < min(item.end_frame, join.start_frame + join.frames)
                    for join in decisions.joins)):
            clearance = placement.get("clearance", {})
            gap = clearance.get("gap_px")
            margin = clearance.get("margin_px")
            if gap is not None and "margin_px" not in clearance:
                from yap_studio.faces.clearance import face_clearance_px
                from yap_studio.project.status import ProjectContext

                margin = face_clearance_px(ProjectContext(project_dir).render_style().data)
            if gap is None or gap < margin:
                if item.payload.get("template") != "playing_grid":
                    front = ", or use layer front" if item.layer == "behind" else ""
                    raise TimelineRefusal(
                        f"{item.id}: the card behind the speaker is too close to her head "
                        "without person restoration; "
                        f"move the card off the smooth join{front}",
                        code="card_behind_join",
                        suggested_action=f"Move the card off the smooth join{front}.")
                raise TimelineRefusal(
                    f"{item.id}: behind tiles are too close to the speaker "
                    "without person restoration; "
                    "move the grid off the smooth join, or use layer front",
                    code="card_grid_behind_join",
                    suggested_action="Move the grid off the smooth join, or use layer front.")
        if (item.kind != "graphic" or item.payload.get("template") != "reel_feed"
                or item.payload.get("slots", {}).get("ending", "handoff") != "handoff"):
            continue
        if any(join.handling == "smooth"
               and join.start_frame <= item.end_frame - 1 < join.start_frame + join.frames
               for join in decisions.joins):
            raise TimelineRefusal(
                f"{item.id}: handoff needs the cut picture on its last frame; "
                "move the feed off the smooth join, or use ending fade",
                code="card_video_handoff",
                suggested_action="Move the feed off the smooth join, or use ending fade.")
    return render_windows(project_dir, decisions, cut=cut), decisions
