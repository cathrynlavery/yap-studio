"""Exact pinned stills and trusted picture geometry for feedback notes."""

from __future__ import annotations

import json
import math
import subprocess
from bisect import bisect_right
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path, PureWindowsPath

import numpy as np

from yap_studio import ffmpeg, paths
from yap_studio.errors import YapError, shorten_text
from yap_studio.faces.clearance import Layout
from yap_studio.faces.model import Box, FacesData, load_faces
from yap_studio.faces.project import cached_cut_faces
from yap_studio.feedback.versions import LayerWindow, VersionManifest, version_layers
from yap_studio.project.receipts import read_receipt
from yap_studio.receipt import sha256_file
from yap_studio.review.model import Comment, frame_mid
from yap_studio.stages.timeline import FPS as TIMELINE_FPS
from yap_studio.stages.timeline import ResolvedTimeline, resolve_timeline
from yap_studio.watch.images import png_bytes

CAPTION_PADDING_PX = 24  # Include a nearby tap around the phrase's glyphs.
FACE_GROWTH = 0.2  # Include the head around the measured face rectangle.
RING_MIN_RADIUS_PX = 12  # Keep small stills readable.
RING_RADIUS_RATIO = 0.03  # Scale the ring to the short canvas side.
RING_MIN_WIDTH_PX = 3  # Preserve a visible white ring at small sizes.
RING_WIDTH_DIVISOR = 4  # Use one quarter of the radius for the white ring.
OUTLINE_WIDTH_PX = 3  # Separate both ring edges from light pictures.
DOT_MIN_RADIUS_PX = 3  # Keep the precise spot visible on small pictures.
DOT_RADIUS_DIVISOR = 5  # Leave space between the dot and ring.
WHITE = (255, 255, 255)  # Make the ring readable on dark pictures.
NEAR_BLACK = (16, 18, 21)  # Make the ring readable on light pictures.
CORAL = (255, 94, 77)  # Mark the exact canvas point.
RGB_CHANNELS = 3  # The decoder and PNG writer exchange RGB pixels.
DECODE_TIMEOUT_S = 30  # Bound each still decode without blocking the round indefinitely.
STILL_ERROR_LIMIT = 200  # Keep batch failure reasons short enough to read beside a note.
UNDER_ORDER = ("caption", "title", "card", "overlay", "speaker", "insert", "background")
_UNSAVED = "The comment's version is not saved in this project."


def decode_frame(media: Path, *, frame: int | None, time_s: float, fps: Fraction | None,
                 width: int, height: int, video_index: int) -> np.ndarray:
    """Decode one native RGB frame at its middle or containing the requested time."""
    seek = float(frame_mid(frame, fps)) if frame is not None and fps is not None else time_s
    command = [ffmpeg.resolve_tool("ffmpeg"), "-v", "error", "-ss", str(seek)]
    # Keep the containing frame and select it after the demuxer's keyframe seek.
    command += ["-noaccurate_seek", "-copyts", "-start_at_zero"]
    command += ["-i", str(media), "-map", f"0:{video_index}", "-an"]
    if frame is not None and fps is not None:
        command += ["-vf", f"select=gte(round(t*{fps.numerator}/{fps.denominator})\\,{frame})"]
    else:
        # Sample the last PTS at or before the seek, buffering only two frames.
        command += ["-vf", f"settb=AVTB,setpts=PTS-{seek}/TB,fps=1:start_time=0:round=up"]
    command += ["-frames:v", "1", "-fps_mode", "passthrough", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-"]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                timeout=DECODE_TIMEOUT_S, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise YapError("Could not decode the pinned frame.") from exc
    if result.returncode or len(result.stdout) != width * height * RGB_CHANNELS:
        raise YapError("Could not decode the pinned frame.")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, RGB_CHANNELS)


def draw_pin(pixels: np.ndarray, x: int, y: int) -> np.ndarray:
    """Draw a clipped ring and centre dot without changing the decoded frame."""
    height, width, _ = pixels.shape
    radius = max(RING_MIN_RADIUS_PX, round(RING_RADIUS_RATIO * min(width, height)))
    ring_width = max(RING_MIN_WIDTH_PX, radius // RING_WIDTH_DIVISOR)
    dot = max(DOT_MIN_RADIUS_PX, radius // DOT_RADIUS_DIVISOR)
    rows, columns = np.ogrid[:height, :width]
    distance = np.hypot(columns - x, rows - y)
    result = pixels.copy()
    result[(distance <= radius + OUTLINE_WIDTH_PX)
           & (distance >= radius - ring_width - OUTLINE_WIDTH_PX)] = NEAR_BLACK
    result[(distance <= radius) & (distance >= radius - ring_width)] = WHITE
    result[distance <= dot] = CORAL
    return result


def _contains(box: Box, x: int, y: int, pad_x: float = 0, pad_y: float = 0) -> bool:
    return (box.x - pad_x <= x <= box.x1 + pad_x
            and box.y - pad_y <= y <= box.y1 + pad_y)


def _hit(kind: str, id: str | None, text: str | None, box: Box, source: str) -> dict:
    return {"kind": kind, "id": id, "text": text, "box": box.to_dict(), "source": source}


def layout_hits(data: dict, frame: int | None, x: int, y: int) -> list[dict]:
    """Return active phrase and element hits from a validated clearance layout."""
    layout = Layout.from_dict(data)
    if frame is None:
        return []
    hits = []
    for phrase in layout.captions:
        if not phrase.start_frame <= frame < phrase.end_frame:
            continue
        boxes = [box for box in phrase.boxes if box.overlaps_frames(frame, frame + 1)]
        if not boxes:
            continue
        left, top = min(box.x for box in boxes), min(box.y for box in boxes)
        right, bottom = max(box.x1 for box in boxes), max(box.y1 for box in boxes)
        union = Box(left, top, right - left, bottom - top)
        if _contains(union, x, y, CAPTION_PADDING_PX, CAPTION_PADDING_PX):
            hits.append(_hit("caption", phrase.id, phrase.text, union, "layout"))
    records = {record["id"]: record for record in data.get("elements", [])}
    for element in layout.elements:
        if (element.start_frame <= frame < element.end_frame
                and _contains(element.box, x, y)):
            raw = records[element.id]
            hits.append(_hit(element.kind, raw.get("element_id", element.id),
                             raw.get("label"), element.box, "layout"))
    return sorted(hits, key=lambda hit: UNDER_ORDER.index(hit["kind"]))


def face_hits(faces: FacesData, frame: int | None, x: int, y: int) -> list[dict]:
    """Return grown face hits and a full picture insert from frame provenance."""
    if frame is None or not 0 <= frame < faces.video.frames:
        return []
    hits = [_hit("speaker", None, None, face.box, "faces") for face in faces.faces_at(frame)
            if _contains(face.box, x, y, face.box.w * FACE_GROWTH, face.box.h * FACE_GROWTH)]
    sources = faces.sampling.get("frame_sources", [])
    if not isinstance(sources, list):
        sources = []
    if any(isinstance(window, dict) and window.get("source") == "insert"
           and type(window.get("start_frame")) is int and type(window.get("end_frame")) is int
           and window["start_frame"] <= frame < window["end_frame"]
           for window in sources):
        hits.append(_hit("insert", None, None,
                         Box(0, 0, faces.video.width, faces.video.height), "faces"))
    return hits


def _order_under(hits: list[dict], layout: dict | None, frame: int | None) -> list[dict]:
    behind_cards = []
    if layout is not None and frame is not None and any(hit["kind"] == "speaker" for hit in hits):
        for record in layout.get("elements", []):
            if (record["kind"] == "card" and "face_windows" in record
                    and record["start_frame"] <= frame < record["end_frame"]
                    and not any(start <= frame < end for start, end in record["face_windows"])):
                box = Box(record["x"], record["y"], record["w"], record["h"])
                behind_cards.append(_hit("card", record.get("element_id", record["id"]),
                                         record.get("label"), box, "layout"))
    return sorted(hits, key=lambda hit: (
        UNDER_ORDER.index("speaker" if hit in behind_cards else hit["kind"]),
        hit in behind_cards,
    ))


def _timeline_matches(root: Path, version: VersionManifest) -> bool:
    inputs = (("edit", "edit.json"), ("removals", "removals.json"),
              ("cut_words", "cut/cut-words.json"), ("spans", "cut/spans.json"))
    expected = {}
    for key, name in inputs:
        snapshot = version.snapshots.get(key)
        expected[name] = snapshot["sha256"] if snapshot is not None else None
    timeline = version.snapshots.get("timeline")
    for snapshot in timeline["lanes"].values() if timeline is not None else ():
        expected[snapshot["file"]] = snapshot["sha256"]
    live = {name: root / name for _, name in inputs}
    live.update({file.relative_to(root).as_posix(): file
                 for file in (root / "timeline").glob("*.json")})
    return expected == {name: sha256_file(file) if file.is_file() else None
                        for name, file in live.items()}


def _sidecar_source(root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    relative = Path(value)
    windows = PureWindowsPath(value)
    if (not relative.parts or relative.is_absolute() or windows.anchor or ".." in relative.parts
            or ".." in windows.parts):
        return None
    source = root / relative
    try:
        return source if source.resolve().is_relative_to(root.resolve()) else None
    except (OSError, RuntimeError, ValueError):
        return None


def _assembled_media(root: Path, version: VersionManifest) -> bool:
    source = _sidecar_source(root, version.media.get("source"))
    if source is None or source.resolve() == (root / "cut/cut.mp4").resolve():
        return False
    receipt = read_receipt(root / "final") or {}
    outputs = receipt.get("outputs")
    return (receipt.get("stage") == "assemble" and receipt.get("status") == "verified"
            and isinstance(outputs, list) and any(
                isinstance(output, dict) and output.get("path") == version.media["source"]
                and output.get("sha256") == version.media["sha256"] for output in outputs))


@dataclass
class PinContext:
    """Share source reads and at most one timeline resolution within a batch."""

    root: Path
    versions: tuple[VersionManifest, ...]
    current: VersionManifest | None
    current_cut: bool = False
    cut_layers: tuple[LayerWindow, ...] | None = None
    sources: dict = field(default_factory=dict)
    timeline_read: bool = False
    timeline: ResolvedTimeline | None = None

    def timeline_for(self, version: VersionManifest | None) -> ResolvedTimeline | None:
        """Use the live timeline only for the unchanged current saved version."""
        if version is None or self.current is None or version.id != self.current.id:
            return None
        if not self.timeline_read:
            self.timeline_read = True
            try:
                if _assembled_media(self.root, version) and _timeline_matches(self.root, version):
                    timeline = resolve_timeline(self.root)
                    frame_count = version.media.get("frame_count")
                    total_frames = timeline.total_frames + timeline.end_hold_frames
                    if timeline.valid and (frame_count is None or total_frames == frame_count):
                        self.timeline = timeline
            except (YapError, OSError, ValueError):
                pass
        return self.timeline

    def source_for(self, version: VersionManifest | None) -> dict:
        """Probe media once and read only sidecars that match its hash and size."""
        key = version.id if version is not None else None
        if key in self.sources:
            return self.sources[key]
        media = version.media_path if version is not None else self.root / "cut/cut.mp4"
        info = ffmpeg.probe(media, count_frames=False)
        if not info.width or not info.height or info.video_index is None:
            raise YapError("Could not read the pinned frame's canvas size.")
        digest = sha256_file(media)
        if version is not None and digest != version.media["sha256"]:
            raise YapError("The saved video changed after this version was saved.")
        source = (_sidecar_source(self.root, version.media["source"])
                  if version is not None else media)
        result = {"media": media, "info": info, "layout": None, "faces": None, "layers": None}
        try:
            if source is not None:
                raw = json.loads(source.with_suffix(".layout.json").read_text(encoding="utf-8"))
                layout = Layout.from_dict(raw)
                if (raw.get("media_sha256") == digest
                        and (layout.width, layout.height) == (info.width, info.height)):
                    result["layout"] = raw
        except (YapError, OSError, UnicodeError, ValueError, TypeError, KeyError):
            pass
        try:
            faces = None
            if version is None or source == self.root / "cut/cut.mp4":
                faces = cached_cut_faces(self.root, media, digest=digest)
            elif source is not None:
                faces = load_faces(source.with_suffix(".faces.json"))
            if (faces is not None and faces.status == "measured" and faces.video.sha256 == digest
                    and (faces.video.width, faces.video.height) == (info.width, info.height)):
                result["faces"] = faces
        except (YapError, OSError, ValueError):
            pass
        if version is not None:
            try:
                if (self.cut_layers is not None and not self.current_cut
                        and self.current is not None and version.id == self.current.id):
                    result["layers"] = self.cut_layers
                else:
                    result["layers"] = version_layers(version)
            except YapError:
                pass
        else:
            result["layers"] = self.cut_layers
        self.sources[key] = result
        return result


def pin_spot(comment: Comment, *, batch_id: str, version_id: str | None,
             context: PinContext) -> dict:
    """Build a pinned still and geometry without making a missing source fail a round."""
    assert comment.pin is not None
    pin = comment.pin
    result = {**pin, "canvas": None, "still": None, "still_error": None,
              "under": [], "on_screen": [],
              "known": dict.fromkeys(("layout", "faces", "layers", "timeline"), False)}
    version = next((item for item in context.versions if item.id == version_id), None)
    if ((version_id is not None and version is None)
            or (version_id is None and not context.current_cut)):
        result["still_error"] = _UNSAVED
        return result
    try:
        source = context.source_for(version)
    except (YapError, OSError) as exc:
        result["still_error"] = (shorten_text(str(exc), STILL_ERROR_LIMIT)
                                 if isinstance(exc, YapError) else "Could not read the video.")
        return result
    info = source["info"]
    width, height = info.width, info.height
    x = min(width - 1, math.floor(pin["x"] * width))
    y = min(height - 1, math.floor(pin["y"] * height))
    result["canvas"] = {"width": width, "height": height, "x": x, "y": y}
    fps = Fraction(version.media["fps"]) if version is not None else info.fps
    pin_time = (frame_mid(pin["frame"], fps) if pin["frame"] is not None and fps is not None
                else Fraction(str(pin["time_s"])))
    try:
        pixels = decode_frame(source["media"], frame=pin["frame"], time_s=pin["time_s"], fps=fps,
                              width=width, height=height, video_index=info.video_index)
        folder = context.root / "feedback" / batch_id
        target = folder / f"{version_id or 'cut'}-{comment.id}.png"
        if target.parent != folder or "\\" in comment.id:
            raise YapError("Could not write the pinned still: invalid comment id.")
        with paths.atomic_output(target, frozenset({".png"})) as temporary:
            temporary.write_bytes(png_bytes(draw_pin(pixels, x, y)))
        result["still"] = target.relative_to(context.root).as_posix()
    except (YapError, OSError) as exc:
        result["still_error"] = (shorten_text(str(exc), STILL_ERROR_LIMIT)
                                 if isinstance(exc, YapError) else "Could not write the still.")
    frame = pin["frame"]
    faces = source["faces"]
    if faces is not None:
        result["known"]["faces"] = True
        if frame is None and faces.times is not None:
            frame = bisect_right(faces.times, pin["time_s"]) - 1
        result["under"].extend(face_hits(faces, frame, x, y))
    if source["layout"] is not None:
        result["known"]["layout"] = source["layout"].get("measured") is not False
        if not result["known"]["layout"]:
            result["layout_reason"] = source["layout"].get("reason")
        result["under"].extend(layout_hits(source["layout"], frame, x, y))
    layers = source["layers"]
    if layers is not None:
        result["known"]["layers"] = True
        for layer in layers:
            if (layer.kind not in ("caption_gap", "music_level")
                    and Fraction(str(layer.start)) <= pin_time < Fraction(str(layer.end))):
                result["on_screen"].append({"kind": layer.kind, "id": layer.id,
                                            "text": layer.text, "source": "layers"})
    timeline = context.timeline_for(version)
    if timeline is not None:
        result["known"]["timeline"] = True
        for element in timeline.elements:
            active = (Fraction(element.start_frame) / TIMELINE_FPS <= pin_time
                      < Fraction(element.end_frame) / TIMELINE_FPS)
            if not active:
                continue
            fit = (element.payload.get("media", {}).get("fit", "cover")
                   if element.kind == "insert" else None)
            if element.kind == "insert" and fit in ("cover", "contain", "crop"):
                result["under"] = [hit for hit in result["under"]
                                   if not (hit["kind"] == "insert" and hit["id"] is None)]
                result["under"].append(_hit("insert", element.id, None,
                                             Box(0, 0, width, height), "timeline"))
            elif element.kind in ("moment", "zoom", "insert"):
                result["on_screen"].append({"kind": element.kind, "id": element.id,
                                            "text": element.payload.get("style", fit),
                                            "source": "timeline"})
    result["under"] = _order_under(result["under"], source["layout"], frame)
    unique = {}
    for item in result["on_screen"]:
        unique.setdefault((item["kind"], item["id"]), item)
    result["on_screen"] = list(unique.values())
    return result
