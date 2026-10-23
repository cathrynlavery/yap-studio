"""Measure room and stillness, then place moving copies from later in the same take."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, closing
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np

from yap_studio import ffmpeg, paths
from yap_studio.captions import caption_layout_settings
from yap_studio.edl import load_edit_list
from yap_studio.errors import ValidationError
from yap_studio.faces.clearance import head_box
from yap_studio.faces.model import Box, FacesData
from yap_studio.matte import chroma
from yap_studio.matte.edge_colour import clean_edge_colours
from yap_studio.plan import Plan, PlanSegment
from yap_studio.render.moments import (
    BYTE_MAX,
    ENCODE_THREADS,
    MOMENT_EASE_FRAMES,
    PROCESS_TIMEOUT_S,
    REFERENCE_WIDTH_PX,
    YUV_NEUTRAL,
    YUV_WHITE,
    _background,
    _decode,
    _dilate,
    _mix,
    _pixels,
    _smoothstep,
    _warp,
)

if TYPE_CHECKING:
    from yap_studio.stages.timeline import ResolvedElement

# Bump for changed clone pixels or measurements and new refusals so assemblies run again.
CLONE_RENDER_VERSION = 2
# One second lets the flash resolve into a moving person.
CLONE_MIN_S = 1.0
# Two seconds keep the copies a short beat rather than a second performance.
CLONE_MAX_S = 2.0
# A short delay separates the two appearances.
CLONE_STAGGER_FRAMES = 9
# Three white frames make the entrance read as a flash.
CLONE_WHITE_FRAMES = 3
# Colour settles six frames after the copy appears.
CLONE_COLOUR_FRAMES = 6
# A visible gap separates the neighbouring face boxes at delivery width.
CLONE_GAP_PX = 24
# The lowest fifth fades when a scaled picture ends above the canvas bottom.
CLONE_FADE_SHARE = 0.2
# Later source poses must be at least a second beyond the window or previous copy.
CLONE_SOURCE_GAP_FRAMES = 30
# Half-second steps bound decoding while giving the pose time to change.
CLONE_SOURCE_STEP_FRAMES = 15
# Twelve candidates bound the search without probing the whole take.
CLONE_MAX_CANDIDATES = 12
# Six frames (0.2 s) catch a pan or bump within a 1 to 2 s window while keeping the decode small.
CLONE_CAMERA_STEP_FRAMES = 6
# ffmpeg refuses selections near 110 terms, so keep each pass well below its parser limit.
CLONE_SELECT_TERMS_MAX = 60
# Expand the person exclusion to keep rim colour out of the camera measurement.
CLONE_MASK_DILATE_PX = 8
# Four delivery pixels tolerate tiny camera vibration without accepting a pan.
CLONE_SHIFT_MAX_PX = 4
# Six luma levels match the matte smoothing stillness threshold.
CLONE_DIFF_MAX = 6
# Leave breathing room between a copy's face and the frame edge.
CLONE_EDGE_MARGIN_PX = 24
# Smaller faces lose too much presence next to the live person.
CLONE_SCALE_MIN = 0.55
# At most a tenth of a copy's face may hide under the live person.
CLONE_FACE_HIDDEN_MAX = 0.1
# Half-size measurements save decoding and keep the thresholds in delivery pixels.
CLONE_MEASURE_DIVISOR = 2
# At least half the window must contain one measured face for reliable placement.
CLONE_FACE_SHARE_MIN = 0.5
# Tiny alpha values belong to background during camera measurement.
CLONE_CAMERA_ALPHA_MIN = 0.01
# Half-alpha pixels define the conservative union of the live person.
CLONE_PERSON_ALPHA_MIN = 0.5
# FFT normalization must stay finite for flat synthetic rooms.
CLONE_FFT_FLOOR = 1e-12
# Plan uses assembly's fixed delivery frame clock.
CLONE_FPS = 30
# Plan uses assembly's audio sample clock to map cut pieces.
CLONE_SAMPLE_RATE = 48_000

_FIXES = {
    "clone_keyed": "pick another Moment style, or remove the clone flash; clone flash needs "
    "a take filmed without a green or blue screen",
    "clone_faces": "run on a Mac that can measure faces with yap-studio faces detect "
    "VIDEO --out FACES.json, or choose another moment style",
    "clone_source": "place clone_flash earlier in the take so at least about 2 s of the same "
    "take follows it, or choose another moment style",
    "clone_camera": "use clone_flash on a part of the take where the camera does not move, "
    "or choose another moment style",
    "clone_room": "choose a line where the speaker sits off centre or further from the camera, "
    "or choose another moment style",
}


def keyed_message(screen: str) -> str:
    """Explain why keyed copies cannot keep a clean edge."""
    return (f"clone_flash cannot clean the {screen} screen edge on the copies, "
            f"so they would carry a {screen} fringe")


def _refuse(code: str, message: str) -> NoReturn:
    from yap_studio.stages.timeline import TimelineRefusal

    raise TimelineRefusal(message, code=code, suggested_action=_FIXES[code])


def _boxes(faces: FacesData, first: int, last: int) -> list[Box]:
    return [faces.frames[frame][0].box for frame in range(first, last)
            if frame < len(faces.frames) and len(faces.frames[frame]) == 1]


def _median(boxes: Sequence[Box]) -> Box:
    return Box(*np.median([(b.x, b.y, b.w, b.h) for b in boxes], axis=0).tolist())


def _union(boxes: Sequence[Box]) -> Box:
    x, y = min(b.x for b in boxes), min(b.y for b in boxes)
    return Box(x, y, max(b.x1 for b in boxes) - x, max(b.y1 for b in boxes) - y)


def candidate_frames(pieces: Sequence[PlanSegment], item: ResolvedElement
                     ) -> list[tuple[int, int, str]]:
    """Find source-ordered cut windows wholly inside one retained piece of this take."""
    window = [p for p in pieces if p.out_f0 < item.end_frame
              and item.start_frame < p.out_f0 + p.n_frames]
    takes = {p.source_id for p in window}
    covered = sum(min(item.end_frame, p.out_f0 + p.n_frames)
                  - max(item.start_frame, p.out_f0) for p in window)
    if len(takes) != 1 or None in takes or covered != item.end_frame - item.start_frame:
        _refuse("clone_source", "clone_flash needs a window in one take")
    take = window[0].source_id
    last = max(p.src_f0 + min(item.end_frame, p.out_f0 + p.n_frames) - p.out_f0 - 1
               for p in window)
    length = item.end_frame - item.start_frame
    candidates = []
    for piece in sorted((p for p in pieces if p.source_id == take),
                        key=lambda p: (p.src_f0, p.out_f0)):
        first = max(piece.src_f0, last + CLONE_SOURCE_GAP_FRAMES)
        for source in range(first, piece.src_f0 + piece.n_frames - length + 1,
                            CLONE_SOURCE_STEP_FRAMES):
            candidates.append((piece.out_f0 + source - piece.src_f0, source, take))
    return sorted(candidates, key=lambda c: (c[1], c[0]))[:CLONE_MAX_CANDIDATES]


def camera_pair(a: np.ndarray, aa: np.ndarray, b: np.ndarray, ba: np.ndarray) -> dict[str, float]:
    """Measure shared background shift in 1080-wide pixels and luma difference."""
    unit = a.shape[1] / REFERENCE_WIDTH_PX
    excluded = _dilate((aa > CLONE_CAMERA_ALPHA_MIN) | (ba > CLONE_CAMERA_ALPHA_MIN),
                       max(1, round(CLONE_MASK_DILATE_PX * unit)))
    known = ~excluded
    if not known.any():
        return {"shift_px": float(REFERENCE_WIDTH_PX), "difference": float(BYTE_MAX)}
    luma_a, luma_b = a[..., 0].astype(float), b[..., 0].astype(float)
    difference = float(np.median(abs(luma_a[known] - luma_b[known])))
    # Mean filling followed by centring removes the artificial edge of the exclusion mask.
    luma_a = np.where(known, luma_a - luma_a[known].mean(), 0)
    luma_b = np.where(known, luma_b - luma_b[known].mean(), 0)
    hann = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
    cross = np.fft.fft2(luma_a * hann) * np.conj(np.fft.fft2(luma_b * hann))
    cross /= np.maximum(abs(cross), CLONE_FFT_FLOOR)
    peak = np.unravel_index(np.argmax(np.fft.ifft2(cross).real), aa.shape)
    shift = [value if value <= limit // 2 else value - limit
             for value, limit in zip(peak, aa.shape, strict=True)]
    return {"shift_px": float(np.hypot(*shift) * REFERENCE_WIDTH_PX / a.shape[1]),
            "difference": difference}


def _still(measurement: dict) -> bool:
    return (measurement["shift_px"] <= CLONE_SHIFT_MAX_PX
            and measurement["difference"] <= CLONE_DIFF_MAX)


def _camera_frames(first: int, last: int) -> list[int]:
    """Sample the whole span, including its first and last frames."""
    return sorted({*range(first, last, CLONE_CAMERA_STEP_FRAMES), last - 1})


def _place(side: str, face: Box, extent: Box, source: Box, source_extent: Box,
           person: np.ndarray, size: tuple[int, int],
           caption_band: tuple[float, float], caption_gap_px: float) -> dict | None:
    width, height = size
    unit = width / REFERENCE_WIDTH_PX
    gap, margin = CLONE_GAP_PX * unit, CLONE_EDGE_MARGIN_PX * unit
    span = extent.x - gap - margin if side == "left" else width - margin - extent.x1 - gap
    if source.w <= 0 or source.h <= 0:
        return None
    scale = min(1.0, span / source.w)
    if scale < CLONE_SCALE_MIN:
        return None
    x = extent.x - gap - source.w * scale if side == "left" else extent.x1 + gap
    placed = Box(x, face.y + face.h / 2 - source.h * scale / 2,
                 source.w * scale, source.h * scale)
    # Check the moving face's full extent, not just its median pose.
    bounds = Box(placed.x + (source_extent.x - source.x) * scale,
                 placed.y + (source_extent.y - source.y) * scale,
                 source_extent.w * scale, source_extent.h * scale)
    head = head_box(bounds)
    caption_gap = caption_gap_px * unit
    if (bounds.x < 0 or bounds.x1 > width or bounds.y < 0
            or bounds.y1 > height
            or (head.y < caption_band[1] + caption_gap
                and caption_band[0] - caption_gap < head.y1)):
        return None
    sy, sx = person.shape[0] / height, person.shape[1] / width
    x0, y0 = max(0, int(np.floor(bounds.x * sx))), max(0, int(np.floor(bounds.y * sy)))
    x1 = min(person.shape[1], int(np.ceil(bounds.x1 * sx)))
    y1 = min(person.shape[0], int(np.ceil(bounds.y1 * sy)))
    hidden = person[y0:y1, x0:x1]
    if not hidden.size or hidden.mean() > CLONE_FACE_HIDDEN_MAX:
        return None
    if bounds.x < extent.x1 and extent.x < bounds.x1:
        return None
    return {"side": side, "scale": scale, "source_face": source.to_dict(),
            "source_face_union": source_extent.to_dict(), "placed_face": placed.to_dict(),
            "placed_face_union": bounds.to_dict()}


def measure_clones(pictures: Mapping[int, np.ndarray], mattes: Mapping[int, np.ndarray],
                   faces: FacesData, pieces: Sequence[PlanSegment], item: ResolvedElement, *,
                   size: tuple[int, int], caption_band: tuple[float, float],
                   caption_gap_px: float) -> dict[str, Any]:
    """Pure measurement on decoded arrays, with all decisions on the full moment window."""
    first, last = item.start_frame, item.end_frame
    length = last - first
    boxes = _boxes(faces, first, last)
    if faces.status != "measured" or len(boxes) < length * CLONE_FACE_SHARE_MIN:
        _refuse("clone_faces", "clone_flash needs one measured face on half the window")
    face, extent = _median(boxes), _union(boxes)
    candidates = candidate_frames(pieces, item)
    if not candidates:
        _refuse("clone_source", "clone_flash has no later source window in the same take")
    measurements = [camera_pair(pictures[first], mattes[first], pictures[f], mattes[f])
                    for f in _camera_frames(first, last)]
    if not all(_still(m) for m in measurements):
        _refuse("clone_camera", "clone_flash needs a still camera throughout the window")
    camera = max(measurements, key=lambda m: (m["shift_px"], m["difference"]))
    person = np.logical_or.reduce([mattes[f] >= CLONE_PERSON_ALPHA_MIN for f in range(first, last)])
    middle = first + length // 2
    left = extent.x
    right = size[0] - extent.x1
    sides = ("left", "right") if left >= right else ("right", "left")
    usable, copies = [], []
    measured_candidates = 0
    for start, source, take in candidates:
        source_boxes = _boxes(faces, start, start + length)
        if len(source_boxes) != length:
            continue
        measured_candidates += 1
        measurements = [camera_pair(pictures[middle], mattes[middle], pictures[f], mattes[f])
                        for f in _camera_frames(start, start + length)]
        if not all(_still(m) for m in measurements):
            continue
        worst = max(measurements, key=lambda m: (m["shift_px"], m["difference"]))
        usable.append((start, source, take, source_boxes, worst))
    if not measured_candidates:
        _refuse("clone_source", "clone_flash has no later window with exactly one face")
    if not usable:
        _refuse("clone_camera", "clone_flash has no later single-face window with a still camera")
    for side in sides:
        for start, source, take, source_boxes, worst in usable:
            if copies and source < copies[0]["take_source_start_frame"] + CLONE_SOURCE_GAP_FRAMES:
                continue
            placement = _place(side, face, extent, _median(source_boxes), _union(source_boxes),
                               person, size, caption_band, caption_gap_px)
            if placement is None:
                continue
            appear = CLONE_STAGGER_FRAMES if copies else 0
            copies.append({**placement, "appear_frame": appear, "source_start_frame": start,
                "take": take, "take_source_start_frame": source, "frames": length - appear,
                "camera": worst})
            break
    if not copies:
        _refuse("clone_room", "clone_flash has no room for a face beside the live person")
    return {"id": item.id, "style": "clone_flash", "size": list(size), "face": face.to_dict(),
            "face_union": extent.to_dict(), "camera": camera, "copies": copies}


def clone_frame(live: np.ndarray, alpha: np.ndarray,
                sources: Sequence[tuple[np.ndarray, np.ndarray] | None], record: dict, *,
                index: int, frames: int) -> np.ndarray:
    """Composite premultiplied copies behind the cleaned live silhouette."""
    fade = _smoothstep((frames - 1 - index) / MOMENT_EASE_FRAMES)
    if not fade:
        return live.copy()
    height, width = live.shape[:2]
    unit_x, unit_y = width / record["size"][0], height / record["size"][1]
    layers = []
    white = np.array((YUV_WHITE, YUV_NEUTRAL, YUV_NEUTRAL))
    for copy, source in zip(record["copies"], sources, strict=True):
        age = index - copy["appear_frame"]
        if age < 0 or source is None:
            continue
        picture, mask = source
        colour = _smoothstep((age - CLONE_WHITE_FRAMES + 1)
                             / (CLONE_COLOUR_FRAMES - CLONE_WHITE_FRAMES + 1))
        # The first three complete frames stay solid white.
        if age < CLONE_WHITE_FRAMES:
            colour = 0
        cleaned = clean_edge_colours(picture, _pixels(mask * BYTE_MAX))
        pixels = _mix(white, cleaned, colour) * mask[..., None]
        src, placed = copy["source_face"], copy["placed_face"]
        center = ((src["x"] + src["w"] / 2) * unit_x,
                  (src["y"] + src["h"] / 2) * unit_y)
        target = ((placed["x"] + placed["w"] / 2) * unit_x,
                  (placed["y"] + placed["h"] / 2) * unit_y)
        scale = copy["scale"]
        warped = _warp(np.dstack((pixels, mask)), 0, scale, center, output_center=target)
        if scale < 1:
            bottom = target[1] + (height - center[1]) * scale
            ramp = np.clip((bottom - np.arange(height)) / (height * scale * CLONE_FADE_SHARE),
                           0, 1)
            warped *= ramp[:, None, None]
        layers.append(warped * fade)
    if not layers:
        return live.copy()
    present = np.logical_or.reduce([layer[..., 3] > 0 for layer in layers])
    if not present.any():
        return live.copy()
    out = _background(live, alpha).astype(float)
    for layer in layers:
        out = out * (1 - layer[..., 3, None]) + layer[..., :3]
    clean = clean_edge_colours(live, _pixels(alpha * BYTE_MAX))
    out = _pixels(_mix(out, clean, alpha))
    out[~present] = live[~present]
    return out


def _frame_runs(selected: Sequence[int]) -> list[tuple[int, int]]:
    """Group sorted frame numbers into inclusive runs without duplicates."""
    runs = []
    for frame in sorted(set(selected)):
        if runs and frame == runs[-1][1] + 1:
            runs[-1] = runs[-1][0], frame
        else:
            runs.append((frame, frame))
    return runs


def _measure_frames(source: Path, selected: Sequence[int], size: tuple[int, int], *,
                    gray: bool, commands: list[tuple[str, ...]]) -> dict[int, np.ndarray]:
    width, height = size
    channels = 1 if gray else 3
    runs = _frame_runs(selected)
    pictures = {}
    for offset in range(0, len(runs), CLONE_SELECT_TERMS_MAX):
        chunk = runs[offset:offset + CLONE_SELECT_TERMS_MAX]
        frames = [frame for first, last in chunk for frame in range(first, last + 1)]
        select = "+".join(f"eq(n,{first})" if first == last else f"between(n,{first},{last})"
                          for first, last in chunk)
        command = ["ffmpeg", "-nostdin", "-v", "error", "-threads", str(ENCODE_THREADS),
            "-filter_threads", str(ENCODE_THREADS), "-i", str(source), "-an", "-vf",
            f"trim=end_frame={frames[-1] + 1},select='{select}',scale={width}:{height},"
            f"format={'gray' if gray else 'yuv444p'}", "-fps_mode", "passthrough",
            "-threads", str(ENCODE_THREADS), "-f", "rawvideo", "pipe:1"]
        result = ffmpeg.run(command, binary=True, timeout_s=PROCESS_TIMEOUT_S)
        commands.append(tuple(result.args))
        if len(result.stdout) != len(frames) * channels * height * width:
            raise ValidationError("moment.decode", "missing clone measurement frames")
        data = np.frombuffer(result.stdout, np.uint8).reshape(len(frames), channels, height, width)
        arrays = data[:, 0].astype(float) / BYTE_MAX if gray else data.transpose(0, 2, 3, 1)
        pictures.update(zip(frames, arrays, strict=True))
    return pictures


@dataclass
class Clones:
    """A measured whole-window decision, independent of the pilot window."""

    item: ResolvedElement
    record: dict[str, Any]

    def frames(self, cut: Path, matte: Path, first: int, last: int, size: tuple[int, int],
               commands: list[tuple[str, ...]]) -> Iterator[np.ndarray]:
        with ExitStack() as stack:
            live = stack.enter_context(closing(_decode(cut, first, last, size, commands=commands)))
            masks = stack.enter_context(closing(_decode(matte, first, last, size, gray=True,
                                                        commands=commands)))
            sources = []
            for copy in self.record["copies"]:
                appear = self.item.start_frame + copy["appear_frame"]
                start = copy["source_start_frame"] + max(0, first - appear)
                end = copy["source_start_frame"] + max(0, last - appear)
                if start >= end:
                    sources.append(None)
                    continue
                pictures = stack.enter_context(closing(_decode(cut, start, end, size,
                                                               commands=commands)))
                alphas = stack.enter_context(closing(_decode(matte, start, end, size, gray=True,
                                                             commands=commands)))
                sources.append(zip(pictures, alphas, strict=True))
            for frame, picture, alpha in zip(range(first, last), live, masks, strict=True):
                local = frame - self.item.start_frame
                decoded = [next(source, None) if source is not None and local >= c["appear_frame"]
                           else None for c, source in zip(self.record["copies"], sources,
                                                         strict=True)]
                if any(value is None and local >= c["appear_frame"]
                       for c, value in zip(self.record["copies"], decoded, strict=True)):
                    raise ValidationError("moment.decode", "missing clone source picture or matte")
                yield clone_frame(picture, alpha, decoded, self.record, index=local,
                                  frames=self.item.end_frame - self.item.start_frame)


def prepare(root: Path, project_dir: Path, cut: Path, matte: Path, item: ResolvedElement,
            style: dict, commands: list[tuple[str, ...]]) -> Clones:
    """Decode matching picture and matte selections in bounded passes at half size."""
    from yap_studio.faces.project import cut_faces
    from yap_studio.stages.captions import caption_band_geometry
    from yap_studio.stages.timeline import CANVAS_HEIGHT_PX

    metadata = chroma.cutout_metadata(matte)
    if metadata is not None:
        _refuse("clone_keyed", keyed_message(metadata["screen"]))
    faces = cut_faces(project_dir, cut)
    if faces.status != "measured":
        _refuse("clone_faces", "clone_flash needs measured cut faces")
    plan = Plan.build(load_edit_list(paths.validate_input(root / "cut/spans.json")),
                      fps=Fraction(CLONE_FPS), sample_rate=CLONE_SAMPLE_RATE)
    candidates = candidate_frames(plan.segments, item)
    if not candidates:
        _refuse("clone_source", "clone_flash has no later source window in the same take")
    length = item.end_frame - item.start_frame
    selected = sorted({*range(item.start_frame, item.end_frame),
                       *(f for first, _, _ in candidates
                         for f in _camera_frames(first, first + length))})
    size = faces.video.width, faces.video.height
    measure_size = tuple(max(1, value // CLONE_MEASURE_DIVISOR) for value in size)
    pictures = _measure_frames(cut, selected, measure_size, gray=False, commands=commands)
    masks = _measure_frames(matte, selected, measure_size, gray=True, commands=commands)
    settings = caption_layout_settings(style["captions"], style["brand"])
    bands = [caption_band_geometry(settings,
        safe_bottom_px=style["internal"]["caption_safe_bottom_px"], backing=backing)
        for backing in ((), ("box",), ("shadow",))]
    unit = size[1] / CANVAS_HEIGHT_PX
    band = min(b.y for b in bands) * unit, max(b.y + b.height for b in bands) * unit
    record = measure_clones(pictures, masks, faces, plan.segments, item,
                             size=size, caption_band=band,
                             caption_gap_px=style["graphics"]["caption_gap_px"])
    return Clones(item, record)
