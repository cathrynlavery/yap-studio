"""Render opt-in person moments in Y'CbCr without changing the picture's colour matrix."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import closing, nullcontext
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from yap_studio import ffmpeg, paths
from yap_studio.errors import ValidationError
from yap_studio.matte import chroma
from yap_studio.text_hook import TEXT_HOOK_ID

if TYPE_CHECKING:
    from yap_studio.stages.timeline import ResolvedElement

# Changing this value invalidates assemblies made with an older effect implementation.
MOMENT_RENDER_VERSION = 3
MOMENT_STYLES = ("color_pop", "background_blur", "spotlight", "sticker_freeze", "outline",
                 "clone_flash")
# Short beats need enough time to read without holding the picture for a whole sentence.
MOMENT_MIN_S = 0.8
# A moment ends before it becomes a sustained scene treatment.
MOMENT_MAX_S = 3.0
# Space between starts keeps nearby lines from competing for emphasis.
MOMENT_MIN_GAP_S = 10.0
# Six output frames give a brief, smooth entrance and return to live picture.
MOMENT_EASE_FRAMES = 6
# Two beats keep moments scarce within one video.
MOMENT_MAX_COUNT = 2
# A sticker needs a readable freeze, including its entrance.
STICKER_HOLD_MIN_S = 0.6
# Longer freezes hide too much of the spoken performance.
STICKER_HOLD_MAX_S = 1.2
# The longest sticker window includes the hold and six frames of return at 30 fps.
STICKER_WINDOW_MAX_S = 1.4
# Geometry scales from the delivery canvas width.
REFERENCE_WIDTH_PX = 1080
# A ten-pixel white border makes the sticker silhouette readable.
STICKER_RING_PX = 10
# A slight counterclockwise tilt makes the frozen person read as a sticker.
STICKER_TILT_DEG = -4.0
# A small enlargement separates the sticker from its original pose.
STICKER_SCALE = 1.06
# The flash clears quickly enough to leave the words readable.
STICKER_FLASH_FRAMES = 4
# A soft, offset shadow lifts the sticker from the frozen backdrop.
STICKER_SHADOW_PX = 12
# The downward shadow offset suggests a sticker floating above the picture.
STICKER_SHADOW_OFFSET_PX = 10
# A half-strength shadow gives depth without a black fringe.
STICKER_SHADOW_OPACITY = 0.5
# Portrait blur is about two and a half percent of frame width.
BACKGROUND_BLUR_SHARE = 0.025
# A tighter light falloff keeps the room visibly darker near the person.
SPOTLIGHT_BLUR_SHARE = 0.05
# A modest gain lifts the silhouette without fully lighting nearby background.
SPOTLIGHT_LIGHT_GAIN = 1.4
# Thirty percent brightness makes the room darken clearly while retaining context.
SPOTLIGHT_BACKGROUND = 0.3
# A four-pixel outside ring makes the accent visible at delivery size.
OUTLINE_RING_PX = 4
# A sixteen-pixel glow gives the outside ring a visible soft halo.
OUTLINE_GLOW_PX = 16
# A stronger glow makes the accent readable against a bright room.
OUTLINE_GLOW_OPACITY = 0.8
# Only clean background samples may fill the person before blurring.
BACKGROUND_ALPHA_MAX = 0.01
# The half-alpha silhouette gives a white sticker border even with soft source edges.
STICKER_EDGE_ALPHA = 0.5
# Three box passes approximate a soft Gaussian using only numpy.
BLUR_PASSES = 3
# BT.709 luma weights convert an accent swatch, never the picture itself.
BT709_LUMA = (0.2126, 0.7152, 0.0722)
# Black, white and neutral chroma code values in limited-range eight-bit pictures.
YUV_BLACK, YUV_WHITE, YUV_NEUTRAL = 16, 235, 128
# Limited-range chroma has a 224-code excursion while pixels occupy eight bits.
YUV_CHROMA_RANGE, BYTE_MAX = 224, 255
# Single-threaded lossless encoding makes repeated layers byte-for-byte reproducible.
ENCODE_THREADS = 1
# FFV1 level three stores independent, lossless frame slices.
FFV1_LEVEL = 3
# Allow cut-prefix decoding and CPU effects without an unbounded subprocess deadline.
PROCESS_TIMEOUT_S = 600


def check_moments(elements: Sequence[ResolvedElement], style: Mapping, finding: Callable, *,
                  keyed_screen: str | None = None) -> None:
    """Append moment findings without changing existing rule order or element order."""
    from yap_studio.render.moment_clones import _FIXES, CLONE_MAX_S, CLONE_MIN_S, keyed_message
    from yap_studio.stages.timeline import FPS

    moments = sorted((item for item in elements if item.kind == "moment"),
                     key=lambda item: (item.start_frame, item.id))
    allowed = style.get("moments", {}).get("styles", [])
    for index, item in enumerate(moments):
        name = item.payload["style"]
        if name not in allowed:
            finding("moment_off", f"{name} is off in this style", item,
                    suggested_action=f"turn on {name} under moments in the style, "
                    "or remove the moment")
        minimum, maximum = ((MOMENT_MIN_S, STICKER_WINDOW_MAX_S)
                            if name == "sticker_freeze" else (MOMENT_MIN_S, MOMENT_MAX_S))
        if name == "clone_flash":
            minimum, maximum = CLONE_MIN_S, CLONE_MAX_S
            if keyed_screen is not None:
                finding("clone_keyed", keyed_message(keyed_screen), item,
                        suggested_action=_FIXES["clone_keyed"])
        duration = float((item.end_frame - item.start_frame) / FPS)
        if not minimum <= duration <= maximum:
            finding("moment_length", f"{name} window must be {minimum:g} to {maximum:g} s",
                    item, suggested_action=f"set {item.id} to {minimum:g} to {maximum:g} s")
        if index >= MOMENT_MAX_COUNT:
            finding("moment_count", "use at most two moments in a video", item,
                    severity="warning", suggested_action=f"remove {item.id} or keep only two")
        for other in moments[:index]:
            if item.start_frame - other.start_frame < MOMENT_MIN_GAP_S * FPS:
                finding("moment_close", "moment starts are less than 10 s apart", other, item,
                        severity="warning",
                        suggested_action="keep one, or move one at least 10 s away")
        for other in elements:
            if (other.id == item.id
                    or (style.get("hook", {}).get("style") == "text_hook"
                        and other.kind == "title" and other.id == TEXT_HOOK_ID)
                    or (other.kind == "moment" and other not in moments[:index])
                    or not (other.kind in {"insert", "background", "zoom", "transition", "moment",
                                          "grade_compare"}
                            or (other.kind == "graphic" and other.layer == "behind")
                            or (name == "clone_flash"
                                and other.kind in {"title", "graphic", "overlay"}))):
                continue
            if item.start_frame < other.end_frame and other.start_frame < item.end_frame:
                finding("moment_overlap", f"moment cannot overlap {other.id}", other, item,
                        suggested_action=f"move or remove {item.id} so it does not overlap "
                        f"{other.id}")


def person_colours(yuv: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Edge colour cleanup for person composites plugs in here; keep pixels unchanged today."""
    return yuv


def _mix(background: np.ndarray, foreground: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    if isinstance(alpha, np.ndarray):
        alpha = alpha[..., None]
    return background * (1 - alpha) + foreground * alpha


def _pixels(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0, BYTE_MAX).astype(np.uint8)


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3 - 2 * value)


def _ease(index: int, frames: int) -> float:
    return _smoothstep(min(index, frames - 1 - index) / MOMENT_EASE_FRAMES)


def _blur(values: np.ndarray, radius: int) -> np.ndarray:
    """Use separable running sums so blur cost does not grow with its radius."""
    out = values.astype(np.float64)
    if radius <= 0:
        return out
    size = 2 * radius + 1
    for _ in range(BLUR_PASSES):
        for axis in (0, 1):
            padding = [(0, 0)] * out.ndim
            padding[axis] = (radius, radius)
            padded = np.pad(out, padding, mode="edge")
            padding[axis] = (1, 0)
            sums = np.cumsum(np.pad(padded, padding), axis=axis)
            out = (np.take(sums, range(size, sums.shape[axis]), axis=axis)
                   - np.take(sums, range(sums.shape[axis] - size), axis=axis)) / size
    return out


def _dilate(alpha: np.ndarray, radius: int) -> np.ndarray:
    out = alpha.copy()
    for axis in (0, 1):
        padding = [(0, 0), (0, 0)]
        padding[axis] = (radius, radius)
        padded = np.pad(out, padding)
        out = np.zeros_like(out)
        for offset in range(2 * radius + 1):
            slices = [slice(None), slice(None)]
            slices[axis] = slice(offset, offset + alpha.shape[axis])
            np.maximum(out, padded[tuple(slices)], out=out)
    return out


def _background(yuv: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Fill from the nearest clean row or column sample before any blur or transform."""
    known = alpha <= BACKGROUND_ALPHA_MAX
    if not known.any():
        return yuv.copy()
    height, width = alpha.shape
    yy, xx = np.indices(alpha.shape)
    distance = np.full(alpha.shape, height + width)
    out = yuv.copy()
    for axis, coords in ((0, yy), (1, xx)):
        limit = alpha.shape[axis]
        before = np.maximum.accumulate(np.where(known, coords, -limit), axis=axis)
        after = np.flip(np.minimum.accumulate(np.flip(
            np.where(known, coords, 2 * limit), axis=axis), axis=axis), axis=axis)
        nearest = np.where(coords - before <= after - coords, before, after)
        valid = (nearest >= 0) & (nearest < limit)
        delta = abs(nearest - coords)
        use = valid & (delta < distance)
        nearest = np.clip(nearest, 0, limit - 1)
        sample = yuv[nearest, xx] if axis == 0 else yuv[yy, nearest]
        out[use] = sample[use]
        distance[use] = delta[use]
    # Fully enclosed rows and columns use clean samples propagated from the first fill.
    missing = distance == height + width
    if missing.any():
        out = _background(out, missing.astype(float))
    return out


def _accent(color: str) -> np.ndarray:
    rgb = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)]) / BYTE_MAX
    luma = float(rgb @ BT709_LUMA)
    cb = (rgb[2] - luma) / (2 * (1 - BT709_LUMA[2]))
    cr = (rgb[0] - luma) / (2 * (1 - BT709_LUMA[0]))
    return np.array([YUV_BLACK + (YUV_WHITE - YUV_BLACK) * luma,
                     YUV_NEUTRAL + YUV_CHROMA_RANGE * cb,
                     YUV_NEUTRAL + YUV_CHROMA_RANGE * cr])


def _warp(values: np.ndarray, angle: float, scale: float, center: tuple[float, float], *,
          output_center: tuple[float, float] | None = None,
          ) -> np.ndarray:
    """Inverse bilinear sample premultiplied person pixels and their alpha together."""
    height, width = values.shape[:2]
    yy, xx = np.indices((height, width), dtype=float)
    cx, cy = center
    tx, ty = output_center if output_center is not None else center
    cosine, sine = math.cos(angle), math.sin(angle)
    x = ((xx - tx) * cosine + (yy - ty) * sine) / scale + cx
    y = (-(xx - tx) * sine + (yy - ty) * cosine) / scale + cy
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    dx, dy = x - x0, y - y0
    out = np.zeros_like(values, dtype=float)
    for ox, oy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        sx, sy = x0 + ox, y0 + oy
        valid = (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)
        weight = (dx if ox else 1 - dx) * (dy if oy else 1 - dy) * valid
        sample = values[np.clip(sy, 0, height - 1), np.clip(sx, 0, width - 1)]
        out += sample * weight[..., None]
    return out


def _sticker(yuv: np.ndarray, alpha: np.ndarray, pop: float, *,
             person: np.ndarray | None = None) -> np.ndarray:
    width = yuv.shape[1]
    unit = width / REFERENCE_WIDTH_PX
    radius = max(1, round(STICKER_RING_PX * unit))
    outer = np.maximum(alpha, _dilate((alpha >= STICKER_EDGE_ALPHA).astype(float), radius))
    white = np.array([YUV_WHITE, YUV_NEUTRAL, YUV_NEUTRAL])
    person = (person_colours(yuv, alpha) if person is None else person).astype(float)
    premultiplied = person * alpha[..., None] + white * (outer - alpha)[..., None]
    yy, xx = np.indices(alpha.shape)
    mass = alpha.sum()
    center = ((float((xx * alpha).sum() / mass), float((yy * alpha).sum() / mass))
              if mass else ((width - 1) / 2, (alpha.shape[0] - 1) / 2))
    warped = _warp(np.dstack((premultiplied, outer)),
                   math.radians(STICKER_TILT_DEG * pop), 1 + (STICKER_SCALE - 1) * pop, center)
    silhouette = warped[..., 3]
    shadow = _blur(silhouette, max(1, round(STICKER_SHADOW_PX * unit)))
    offset = max(1, round(STICKER_SHADOW_OFFSET_PX * unit))
    shadow = np.pad(shadow, ((offset, 0), (offset, 0)))[:-offset, :-offset]
    background = _background(yuv, alpha).astype(float)
    background = _mix(background, np.array([YUV_BLACK, YUV_NEUTRAL, YUV_NEUTRAL]),
                      shadow * STICKER_SHADOW_OPACITY)
    return background * (1 - silhouette[..., None]) + warped[..., :3]


def moment_frame(style: str, live: np.ndarray, alpha: np.ndarray, *, index: int,
                 frames: int, fps: float, accent_color: str, frozen: np.ndarray | None = None,
                 frozen_alpha: np.ndarray | None = None,
                 sticker: np.ndarray | None = None, person: np.ndarray | None = None,
                 frozen_person: np.ndarray | None = None) -> np.ndarray:
    """Return one opaque planar-video picture, with exact live endpoints except the flash."""
    picture = live.astype(float)
    alpha = np.clip(alpha, 0, 1)
    width = live.shape[1]
    amount = _ease(index, frames)
    if style == "sticker_freeze":
        if frozen is None or frozen_alpha is None:
            raise ValueError("sticker_freeze needs the moment's first picture and matte")
        pop = _smoothstep(index / MOMENT_EASE_FRAMES)
        held = sticker if sticker is not None and pop == 1 else _sticker(
            frozen, frozen_alpha, pop, person=frozen_person,
        )
        # The frozen source remains the first frame throughout the hold and return.
        hold = min(round(STICKER_HOLD_MAX_S * fps), frames - MOMENT_EASE_FRAMES)
        returning = _smoothstep((index - hold + 1) / (frames - hold)) if index >= hold else 0
        out = _mix(held, picture, returning)
        flash = max(0, 1 - index / STICKER_FLASH_FRAMES)
        return _pixels(_mix(out, np.array([YUV_WHITE, YUV_NEUTRAL, YUV_NEUTRAL]), flash))
    if style not in MOMENT_STYLES:
        raise ValueError(f"Unknown moment style: {style!r}")
    if not amount:
        return live.copy()
    if style == "outline":
        unit = width / REFERENCE_WIDTH_PX
        ring = _dilate(alpha, max(1, round(OUTLINE_RING_PX * unit))) - alpha
        glow = _blur(ring, max(1, round(OUTLINE_GLOW_PX * unit))) * (1 - alpha)
        edge = np.clip(ring + glow * OUTLINE_GLOW_OPACITY, 0, 1)
        # Keep the line and glow outside every person pixel, including soft matte edges.
        edge *= alpha == 0
        changed = _mix(picture, _accent(accent_color), edge)
    else:
        background = picture.copy()
        if style == "color_pop":
            background[..., 1:] = YUV_NEUTRAL
        elif style == "background_blur":
            background = _blur(_background(live, alpha),
                               max(1, round(width * BACKGROUND_BLUR_SHARE)))
        elif style == "spotlight":
            light = np.clip(_blur(alpha, max(1, round(width * SPOTLIGHT_BLUR_SHARE)))
                            * SPOTLIGHT_LIGHT_GAIN, 0, 1)
            brightness = SPOTLIGHT_BACKGROUND + (1 - SPOTLIGHT_BACKGROUND) * light
            background[..., 0] = YUV_BLACK + (picture[..., 0] - YUV_BLACK) * brightness
        changed = _mix(background, person_colours(picture, alpha) if person is None else person,
                       alpha)
    return _pixels(_mix(picture, changed, amount))


def _decode(source: Path, first: int, last: int, size: tuple[int, int], *,
            gray: bool = False, freeze: int | None = None,
            commands: list[tuple[str, ...]], person_matte: Path | None = None,
            cutout: dict | None = None, fps: Fraction | None = None) -> Iterator[np.ndarray]:
    width, height = size
    channels = 1 if gray else 3
    frame_bytes = width * height * channels
    select = f"between(n,{first},{last - 1})"
    if freeze is not None and freeze < first:
        select += f"+eq(n,{freeze})"
    selection = f"trim=end_frame={last},select='{select}'"
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source)]
    if person_matte is not None and cutout is not None:
        assert fps is not None
        # Use one frame clock for MP4 pictures and FFV1 masks.
        selection += f",settb={fps.denominator}/{fps.numerator},setpts=N"
        filters = [f"[0:v]{selection}[person]", f"[1:v]{selection},format=gray[alpha]"]
        person, alpha = chroma.append_despill(filters, "person", "alpha", cutout)
        filters.extend([f"[{alpha}]nullsink", f"[{person}]scale=out_color_matrix=bt709:"
                        "out_range=tv,format=yuv444p[cleaned]"])
        command += ["-i", str(person_matte), "-filter_complex_threads", "1",
                    "-filter_complex", ";".join(filters), "-map", "[cleaned]", "-an"]
    else:
        command += ["-an", "-vf", selection + f",format={'gray' if gray else 'yuv444p'}"]
    command += ["-fps_mode", "passthrough", "-threads", str(ENCODE_THREADS),
                "-f", "rawvideo", "pipe:1"]
    commands.append(tuple(command))
    buffer = bytearray()
    with closing(ffmpeg.run_stream(command, timeout_s=PROCESS_TIMEOUT_S,
                                  chunk_size=frame_bytes)) as stream:
        for chunk in stream:
            buffer.extend(chunk)
            while len(buffer) >= frame_bytes:
                values = np.frombuffer(bytes(buffer[:frame_bytes]), np.uint8)
                del buffer[:frame_bytes]
                yield (values.reshape(height, width).astype(float) / BYTE_MAX if gray else
                       values.reshape(channels, height, width).transpose(1, 2, 0))
    if buffer:
        raise ValidationError("moment.decode", "partial picture frame; rebuild the cut matte")


def render_moments(cut: Path, matte: Path, items: Sequence[ResolvedElement], work_dir: Path, *,
                   accent_color: str, frames: tuple[int, int] | None = None,
                   commands: list[tuple[str, ...]] | None = None,
                   prepared: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Render only visible window frames, plus the first source frame for a pilot freeze."""
    placements = []
    commands = commands if commands is not None else []
    work = paths.validate_output_dir(work_dir)
    info = ffmpeg.probe(cut, count_frames=False)
    if not info.width or not info.height or not info.fps:
        raise ValidationError("moment.cut", "moments need a cut with size and frame rate")
    size = info.width, info.height
    cutout = chroma.cutout_metadata(matte)
    for number, item in enumerate(items):
        first = max(item.start_frame, frames[0]) if frames else item.start_frame
        last = min(item.end_frame, frames[1] + 1) if frames else item.end_frame
        if first >= last:
            continue
        freeze = item.start_frame if item.payload["style"] == "sticker_freeze" else None
        movie = paths.validate_output(work / f"moment-{number}.mkv",
                                      allowed_existing_suffixes=frozenset())
        raw = paths.validate_output(work / f"moment-{number}.yuv",
                                    allowed_existing_suffixes=frozenset())
        count = 0
        try:
            if item.payload["style"] == "clone_flash":
                if prepared is None or item.id not in prepared:
                    raise ValidationError("moment.project", "prepare new moments with the project")
                with closing(prepared[item.id].frames(cut, matte, first, last, size,
                                                      commands)) as pictures, raw.open("wb") as out:
                    for picture in pictures:
                        out.write(picture.transpose(2, 0, 1).tobytes())
                        count += 1
            else:
                count = _write_live_moment(cut, matte, item, first, last, size, freeze, commands,
                                            raw, info.fps, accent_color, cutout)
            if count != last - first:
                raise ValidationError("moment.decode", "picture or matte has the wrong frame count")
            command = ["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pixel_format",
                       "yuv444p", "-video_size", f"{info.width}x{info.height}", "-framerate",
                       str(info.fps), "-color_range", "tv", "-colorspace", "bt709",
                       "-color_trc", "bt709", "-color_primaries", "bt709",
                       "-i", str(raw), "-an", "-frames:v", str(count),
                       "-c:v", "ffv1",
                       "-level", str(FFV1_LEVEL), "-threads", str(ENCODE_THREADS),
                       "-pix_fmt", "yuv444p", "-fflags", "+bitexact", "-flags:v", "+bitexact",
                       "-color_range", "tv", "-colorspace", "bt709", "-color_trc", "bt709",
                       "-color_primaries", "bt709", str(movie)]
            result = ffmpeg.run(command, timeout_s=PROCESS_TIMEOUT_S)
            commands.append(tuple(result.args))
            verified = ffmpeg.probe(movie)
            if (verified.width, verified.height, verified.fps, verified.video_frames,
                    verified.pix_fmt, verified.video_codec, verified.has_audio) != (
                    info.width, info.height, info.fps, count, "yuv444p", "ffv1", False):
                raise ValidationError("moment.layer", "lossless layer differs from its window")
        finally:
            raw.unlink(missing_ok=True)
        placements.append({"id": item.id, "path": str(movie), "start_frame": first,
                           "frames": count, "width": info.width, "height": info.height})
    return placements


def _write_live_moment(cut: Path, matte: Path, item: ResolvedElement, first: int, last: int,
                       size: tuple[int, int], freeze: int | None, commands: list[tuple[str, ...]],
                       raw: Path, fps: Fraction, accent_color: str, cutout: dict | None) -> int:
    """Keep the original live effects and sticker freeze on their existing decode path."""
    count = 0
    with closing(_decode(cut, first, last, size, freeze=freeze, commands=commands)) as live:
        # Keep the live picture and background separate from the cleaned person.
        person_frames = _decode(cut, first, last, size, freeze=freeze, commands=commands,
                                person_matte=matte, cutout=cutout, fps=fps) \
            if cutout is not None and item.payload["style"] != "outline" else None
        with closing(_decode(matte, first, last, size, gray=True, freeze=freeze,
                             commands=commands)) as masks, raw.open("wb") as output, \
                (closing(person_frames) if person_frames is not None
                 else nullcontext()) as people:
            frozen = frozen_alpha = frozen_person = sticker = None
            if freeze is not None:
                frozen, frozen_alpha = next(live, None), next(masks, None)
                frozen_person = next(people, None) if people is not None else None
                if (frozen is None or frozen_alpha is None
                        or (people is not None and frozen_person is None)):
                    raise ValidationError("moment.decode", "missing freeze picture or matte")
                sticker = _sticker(frozen, frozen_alpha, 1, person=frozen_person)
                if freeze == first:
                    out = moment_frame(item.payload["style"], frozen, frozen_alpha,
                                       index=0, frames=item.end_frame - item.start_frame,
                                       fps=float(fps), accent_color=accent_color, frozen=frozen,
                                       frozen_alpha=frozen_alpha, sticker=sticker,
                                       frozen_person=frozen_person)
                    output.write(out.transpose(2, 0, 1).tobytes())
                    count += 1
            for picture, alpha in zip(live, masks, strict=True):
                person = next(people, None) if people is not None else None
                if people is not None and person is None:
                    raise ValidationError("moment.decode", "missing person picture")
                out = moment_frame(item.payload["style"], picture, alpha,
                                   index=first + count - item.start_frame,
                                   frames=item.end_frame - item.start_frame,
                                   fps=float(fps), accent_color=accent_color, frozen=frozen,
                                   frozen_alpha=frozen_alpha, sticker=sticker, person=person,
                                   frozen_person=frozen_person)
                output.write(out.transpose(2, 0, 1).tobytes())
                count += 1
            if people is not None and next(people, None) is not None:
                raise ValidationError("moment.decode", "extra person picture")
    return count


def overlay_moments(graph: Any, base: str, cut: Path, matte: Path | None,
                    timeline: Any, work: Path, project_dir: Path,
                    commands: list[tuple[str, ...]], conversions: list[dict[str, Any]], *,
                    frames: tuple[int, int] | None = None,
                    style: dict[str, Any] | None = None, root: Path | None = None,
                    records: list[dict[str, Any]] | None = None, cube: Path | None = None) -> str:
    """Place opaque effect windows before graphics, using the real project's saved accent."""
    from yap_studio.grade import graph_filter
    from yap_studio.project.status import ProjectContext
    from yap_studio.render.moment_clones import prepare
    from yap_studio.stages.assemble_picture import _source
    from yap_studio.stages.timeline import TimelineRefusal

    items = [item for item in timeline.elements if item.kind == "moment"]
    if not items:
        return base
    style = style if style is not None else ProjectContext(project_dir).render_style().data

    def refuse(rule: str, message: str, *items: ResolvedElement,
               severity: str = "violation", suggested_action: str = "") -> None:
        if severity == "violation":
            raise TimelineRefusal(message, code=rule, suggested_action=suggested_action)

    check_moments(timeline.elements, style, refuse)
    if matte is None:
        raise ValidationError("moment.matte", "moments need the cut's person matte")
    prepared = {item.id: prepare(root or project_dir, project_dir, cut, matte, item, style,
                                 commands)
                for item in items if item.payload["style"] == "clone_flash"}
    if records is not None:
        records.extend(effect.record for effect in prepared.values())
    for placement in render_moments(cut, matte, items, work,
                                    accent_color=style["graphics"]["accent_color"],
                                    frames=frames, commands=commands, prepared=prepared):
        source = _source(Path(placement["path"]), commands, conversions,
                         owner=placement["id"], convert_hdr=False)
        layer = graph.filter(graph.input(source, owner=placement["id"]),
                             f"setsar=1,settb=1/{source.info.fps},"
                             f"setpts=N+{placement['start_frame']}")
        if cube is not None:
            layer = graph.filter(layer, graph_filter(cube))
        base = graph.overlay(base, layer, placement["start_frame"],
                             placement["start_frame"] + placement["frames"])
    return base
