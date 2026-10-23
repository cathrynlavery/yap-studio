"""Pure cut-to-output face geometry on the resolved picture clock."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from yap_studio.faces.detect import find_gaps
from yap_studio.faces.model import Box, FaceBox, FacesData, VideoIdentity
from yap_studio.stages.green_screen import layout_windows
from yap_studio.zoom_curve import ZoomCurve

if TYPE_CHECKING:
    from yap_studio.stages.broll_split import SplitLayout
    from yap_studio.stages.timeline import ResolvedTimeline


@dataclass(frozen=True)
class ZoomPiece:
    """A zoom's picture piece, retaining the whole element's animation window."""

    start_frame: int
    end_frame: int
    curve: ZoomCurve
    base_w: int | None = None
    base_h: int | None = None


def picture_zooms(inputs: Sequence[tuple], cut_source: Any) -> list[ZoomPiece]:
    """Use the renderer's actual source ranges and native cover dimensions."""
    zooms = []
    for item, piece, source, _ in inputs:
        base_w, base_h = 1080, 1920
        if source is not cut_source:
            factor = max(1080 / source.info.width, 1920 / source.info.height)
            base_w = math.ceil(source.info.width * factor / 2) * 2
            base_h = math.ceil(source.info.height * factor / 2) * 2
        zooms.append(ZoomPiece(
            max(item.start_frame, piece.out_f0),
            min(item.end_frame, piece.out_f0 + piece.n_frames),
            ZoomCurve.from_payload(item.start_frame, item.end_frame, item.payload), base_w, base_h,
        ))
    return zooms


def behind_face_windows(
    timeline: ResolvedTimeline, start: int, end: int,
) -> tuple[tuple[int, int], ...]:
    """Check behind cards wherever assembly skips restoring the keyed person."""
    windows: list[tuple[int, int]] = []
    for item in sorted(timeline.elements, key=lambda item: item.start_frame):
        if item.kind not in {"zoom", "insert", "transition"}:
            continue
        first, last = max(start, item.start_frame), min(end, item.end_frame)
        if first >= last:
            continue
        if windows and first <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], last))
        else:
            windows.append((first, last))
    return tuple(windows)


def map_picture_faces(
    cut: FacesData, timeline: ResolvedTimeline, *,
    zooms: Sequence[ZoomPiece], video: VideoIdentity,
    splits: Sequence[tuple[int, int, SplitLayout]] = (),
    camera: Sequence[tuple[float, float, float]] | None = None,
) -> tuple[FacesData, list[dict]]:
    """Share the final face windows and clock between planners and assembly."""
    mapped, sources = map_cut_faces(
        cut, total_frames=timeline.total_frames, end_hold_frames=timeline.end_hold_frames,
        zooms=zooms, inserts=[(item.start_frame, item.end_frame) for item in timeline.elements
                             if item.kind == "insert"
                             and item.payload.get("media", {}).get("fit") != "split"],
        splits=splits,
        transitions=[(item.start_frame, item.end_frame) for item in timeline.elements
                     if item.kind == "transition"], video=video,
        layouts=layout_windows(timeline),
        camera=camera,
    )
    # Plan with the same coordinate precision the final faces record will carry.
    return FacesData.from_dict(mapped.to_dict()), sources


def zoom_ratios(piece: ZoomPiece, frame: int) -> tuple[float, float]:
    """Match the picture scale's even-pixel truncation at an output frame.

    Unknown take geometry uses the cut canvas; mapping adds a 2 px allowance
    on every edge for this estimate.
    """
    scale = piece.curve.at(frame)[0]
    base_w, base_h = (
        (1080, 1920) if piece.base_w is None or piece.base_h is None
        else (piece.base_w, piece.base_h)
    )
    return math.trunc(base_w * scale / 2) * 2 / base_w, (
        math.trunc(base_h * scale / 2) * 2 / base_h
    )


def map_cut_faces(
    cut: FacesData, *, total_frames: int, end_hold_frames: int,
    zooms: Sequence[ZoomPiece], inserts: Sequence[tuple[int, int]], video: VideoIdentity,
    transitions: Sequence[tuple[int, int]] = (),
    layouts: Sequence[tuple[int, int, tuple, int]] = (),
    splits: Sequence[tuple[int, int, SplitLayout]] = (),
    camera: Sequence[tuple[float, float, float]] | None = None,
) -> tuple[FacesData, list[dict]]:
    """Map measured cut faces through zooms, picture replacements and the end hold."""
    sources: list[dict] = []
    detector = {**cut.detector, "mapped_from": "cut/cut.mp4", "cut_sha256": cut.video.sha256}
    sampling = {"cut": cut.sampling, "frame_sources": sources}
    reason = cut.reason if cut.status != "measured" else (
        f"cut face frame count {cut.video.frames} differs from timeline total_frames {total_frames}"
        if cut.video.frames != total_frames else None
    )
    if reason is not None:
        return FacesData("unmeasured", reason, video, detector, sampling,
                         (), (), cut.warnings), sources
    if video.frames != total_frames + end_hold_frames:
        raise ValueError("output video frame count must equal cut frames plus end hold frames")
    if camera is not None and (len(camera) != total_frames or any(
        len(transform) != 3 or any(not math.isfinite(value) for value in transform)
        or transform[0] <= 0 for transform in camera
    )):
        raise ValueError("camera needs a finite positive scale and origin for every cut frame")
    frames = []
    for frame in range(video.frames):
        if frame >= total_frames:
            faces, source = frames[-1] if camera is not None else cut.frames[-1], "hold"
        elif any(start <= frame < end for start, end in transitions):
            faces, source = (), "transition"
        elif any(start <= frame < end for start, end in inserts):
            faces, source = (), "insert"
        elif layout := next((layout for start, end, layout in splits
                             if start <= frame < end), None):
            faces = tuple(FaceBox(layout.map_box(face.box), face.confidence, face.source)
                          for face in cut.frames[frame])
            source = "split"
        else:
            piece = next((piece for piece in zooms
                          if piece.start_frame <= frame < piece.end_frame), None)
            faces, source = cut.frames[frame], "cut"
            if camera is not None:
                scale, x, y = camera[frame]
                faces = tuple(FaceBox(Box(x + scale * face.box.x, y + scale * face.box.y,
                                         scale * face.box.w, scale * face.box.h),
                                      face.confidence, face.source) for face in faces)
                source = "camera3d"
            if piece is not None:
                rx, ry = zoom_ratios(piece, frame)
                _, cx, cy = piece.curve.at(frame)
                cx, cy = cx * 1080, cy * 1920
                pad = 2 if piece.base_w is None or piece.base_h is None else 0
                faces = tuple(FaceBox(Box(
                    cx + (face.box.x - cx) * rx - pad,
                    cy + (face.box.y - cy) * ry - pad,
                    face.box.w * rx + 2 * pad, face.box.h * ry + 2 * pad,
                ), face.confidence, face.source) for face in faces)
                source = "zoom_estimate" if pad else "zoom"
            layout = next((window for window in layouts if window[0] <= frame < window[1]), None)
            if layout is not None:
                start, _, (bx, by, width, _), wipe = layout
                scale = width / 1080
                moved = tuple(FaceBox(Box(bx + scale * face.box.x, by + scale * face.box.y,
                                         scale * face.box.w, scale * face.box.h),
                                      face.confidence, face.source) for face in faces)
                faces = () if frame < start + wipe else moved
                source = "background_wipe" if frame < start + wipe else "background_layout"
        frames.append(faces)
        if sources and sources[-1]["source"] == source:
            sources[-1]["end_frame"] = frame + 1
        else:
            sources.append({"start_frame": frame, "end_frame": frame + 1, "source": source})
    return FacesData("measured", None, video, detector, sampling, tuple(frames),
                     tuple(find_gaps(frames)), cut.warnings), sources
