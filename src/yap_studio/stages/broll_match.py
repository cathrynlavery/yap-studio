"""Measure visible B-roll and gently match it to nearby talking-head picture."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from yap_studio import ffmpeg, paths
from yap_studio.canvas import Canvas
from yap_studio.edl import load_edit_list
from yap_studio.errors import ProcessingError
from yap_studio.plan import Plan
from yap_studio.project.model import load_project
from yap_studio.stages.broll_framing import insert_fit
from yap_studio.stages.broll_split import DEFAULT_SEAM_Y, WIDTH
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline

if TYPE_CHECKING:
    from yap_studio.stages.assemble_picture import _Source

STRENGTH = 0.7  # Partial moves preserve the shot's character.
GAMMA = 2.2  # Approximate display RGB in linear light for exposure and balance.
LUMA = np.array((0.2126, 0.7152, 0.0722))  # Display RGB luma weights.
MEASURE_WIDTH = 108  # Small portrait samples keep decoding and statistics cheap.
MEASURE_HEIGHT = 192  # The long sample side preserves the delivery aspect ratio.
FPS = 30  # Resolved picture windows use the delivery frame clock.
SAMPLE_RATE = 48_000  # Plan uses the same audio clock as assembly.
TAKE_MARGIN = 15  # Include the talking head just outside an insert's window.
SAMPLE_POSITIONS = ((1, 6), (1, 2), (5, 6))  # Three interior samples avoid clip edges.
SPREAD_PERCENTILES = (10, 90)  # Ignore extreme shadows and highlights for contrast.
NEUTRAL_PERCENTILE = 40  # Least saturated pixels estimate white balance.
MIN_SPREAD = 0.02  # Flat pictures must not divide by zero.
MIN_MEDIAN = 0.01  # Finite exposure ratios include black pictures.
MIN_TAKE_LUMA = 0.03  # Near-black takes contain no useful reference picture.
MIN_SATURATION = 0.01  # Saturation ratios stay finite for grey pictures.
GREYSCALE_SATURATION = 0.03  # Monochrome shots keep their original colour intent.
WB_FLOOR = 1e-4  # Empty colour channels must not create infinite gains.
EXPOSURE_LIMIT = 0.7  # Limit the shift to less than one stop in either direction.
DARK_MEDIAN = 0.15  # Dark scenes receive no exposure lift.
DARK_RAMP = 0.20  # Lift reaches its full allowance at a median of 0.35.
CONTRAST_LIMITS = (0.85, 1.20)  # Keep local contrast close to the source.
GAIN_LIMITS = (0.90, 1.10)  # Preserve warm sunsets and other strong casts.
SATURATION_LIMITS = (0.75, 1.30)  # Avoid washed-out or exaggerated colour.
UNCHANGED_STOPS = 0.02  # Smaller exposure shifts are invisible in delivery.
UNCHANGED_CONTRAST = 0.01  # Skip negligible contrast changes.
UNCHANGED_GAIN = 0.005  # Skip negligible white balance changes.
UNCHANGED_SATURATION = 0.01  # Skip negligible saturation changes.
SHOULDER = 0.8  # Roll lifted highlights off before clipping.
SHOULDER_RANGE = 0.2  # The highlight shoulder approaches display white smoothly.
STATS_DECIMALS = 4  # Stable measurements feed deterministic match parameters.
PARAM_DECIMALS = 3  # Small parameter quantization makes LUT reuse predictable.
LUT_SIZE = 33  # A compact grid supports smooth tetrahedral interpolation.
LUT_DECIMALS = 6  # Cube values retain enough precision for the interpolation grid.
RGB_MAX = 255  # Decoding uses eight-bit display RGB values.
CHANNELS = 3  # Measurement pools RGB pixels without padding or alpha.
RGBA_CHANNELS = 4  # Overlay samples retain alpha to identify visible pixels.
VISIBLE_ALPHA = 128  # Ignore pixels whose transparency outweighs their colour.
MIN_VISIBLE_FRACTION = 0.05  # Sparse artwork has too little picture to match reliably.
FILTER_THREADS = 1  # Stable HDR filter arithmetic still allows parallel decoding.
READ_MARGIN = 0.5  # Retiming and frame-rate rounding need a few trailing source frames.
MIN_TIMEOUT = 60  # Allow startup and short source decoding on a shared machine.
TIMEOUT_OVERHEAD = 30  # Allow process startup and filter setup before the decoding budget.
TIMEOUT_PER_SECOND = 12  # Budget only the source span consumed by the measurement.
WINDOW_BLEED = 2  # Assembly extends rounded windows two pixels under their chrome.
WINDOW_ALIGNMENT = 2  # Window cover scaling uses even dimensions, as in assembly.
MASK_EDGE = 0.5  # The rounded mask places its antialiased edge at the pixel centre.

Parameters = dict[str, Any] | Literal["unchanged"]
MatchKey = str | tuple[str, int]


@dataclass(frozen=True)
class _Shot:
    """A visible source and its measurement window on the output clock."""

    key: MatchKey
    id: str
    kind: str
    start_frame: int
    end_frame: int
    payload: dict[str, Any]
    source: _Source


def measure(pixels: np.ndarray) -> dict[str, Any]:
    """Pool display pixels and estimate balance from their least saturated portion."""
    rgb = pixels.reshape(-1, CHANNELS).astype(np.float64) / RGB_MAX
    luma = rgb @ LUMA
    chroma = rgb.max(axis=1) - rgb.min(axis=1)
    low, high = np.percentile(luma, SPREAD_PERCENTILES)
    neutral = rgb[chroma <= np.percentile(chroma, NEUTRAL_PERCENTILE)] ** GAMMA
    red, green, blue = neutral.mean(axis=0)
    return {
        "median": round(float(np.median(luma)), STATS_DECIMALS),
        "spread": round(float(max(high - low, MIN_SPREAD)), STATS_DECIMALS),
        "saturation": round(float(chroma.mean()), STATS_DECIMALS),
        "wb": [round(float(value / max(green, WB_FLOOR)), STATS_DECIMALS)
               for value in (red, blue)],
    }


def _linear(median: float) -> float:
    return max(median, MIN_MEDIAN) ** GAMMA


def _unchanged(params: dict[str, Any]) -> bool:
    return (abs(params["stops"]) < UNCHANGED_STOPS
            and abs(params["contrast"] - 1) < UNCHANGED_CONTRAST
            and all(abs(gain - 1) < UNCHANGED_GAIN for gain in params["gains"])
            and abs(params["saturation"] - 1) < UNCHANGED_SATURATION)


def match_parameters(take: dict[str, Any], broll: dict[str, Any]) -> Parameters:
    """Bound partial exposure, contrast, balance and saturation moves."""
    stops = float(np.clip(STRENGTH * math.log2(_linear(take["median"]) /
                                             _linear(broll["median"])),
                          -EXPOSURE_LIMIT, EXPOSURE_LIMIT))
    if stops > 0:
        stops = min(stops, EXPOSURE_LIMIT * float(np.clip(
            (broll["median"] - DARK_MEDIAN) / DARK_RAMP, 0, 1)))
    contrast = float(np.clip((take["spread"] / broll["spread"]) ** STRENGTH,
                             *CONTRAST_LIMITS))
    gains = np.array([float(np.clip((max(target, WB_FLOOR) / max(source, WB_FLOOR))
                                    ** STRENGTH, *GAIN_LIMITS))
                      for target, source in zip(take["wb"], broll["wb"], strict=True)])
    gains = np.array([gains[0], 1, gains[1]])
    gains /= gains @ LUMA
    saturation = float(np.clip((take["saturation"] /
                                max(broll["saturation"], MIN_SATURATION)) ** STRENGTH,
                               *SATURATION_LIMITS))
    if broll["saturation"] < GREYSCALE_SATURATION:
        gains = np.ones(CHANNELS)
        saturation = 1
    stops, contrast, saturation = (round(value, PARAM_DECIMALS)
                                   for value in (stops, contrast, saturation))
    gains = np.round(gains, PARAM_DECIMALS)
    params = {"stops": stops, "contrast": contrast, "gains": gains.tolist(),
              "saturation": saturation, "pivot": round(float(np.clip(
                  _linear(broll["median"]) * 2 ** stops, 0, 1)) ** (1 / GAMMA), PARAM_DECIMALS)}
    return "unchanged" if _unchanged(params) else params


def transform(rgb: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Apply linear balance and exposure, a shoulder, contrast and saturation."""
    linear = rgb ** GAMMA * np.array(params["gains"]) * 2 ** params["stops"]
    shoulder = linear > SHOULDER
    linear[shoulder] = SHOULDER + SHOULDER_RANGE * (
        1 - np.exp(-(linear[shoulder] - SHOULDER) / SHOULDER_RANGE))
    rgb = np.clip(linear, 0, 1) ** (1 / GAMMA)
    rgb = params["pivot"] + (rgb - params["pivot"]) * params["contrast"]
    luma = (rgb @ LUMA)[..., None]
    return np.clip(luma + (rgb - luma) * params["saturation"], 0, 1)


def write_lut(work: Path, params: Parameters) -> Path | None:
    """Write a reusable, content-addressed RGB cube with red changing fastest."""
    if params == "unchanged" or _unchanged(params):
        return None
    grid = np.indices((LUT_SIZE,) * CHANNELS).reshape(CHANNELS, -1).T[:, ::-1]
    values = transform(grid / (LUT_SIZE - 1), params)
    data = (f"LUT_3D_SIZE {LUT_SIZE}\n" + "".join(
        " ".join(f"{value:.{LUT_DECIMALS}f}" for value in row) + "\n"
        for row in values)).encode("ascii")
    folder = paths.validate_output_dir(work / "broll-match")
    target = folder / f"{hashlib.sha256(data).hexdigest()}.cube"
    if target.exists():
        if paths.validate_input(target).read_bytes() != data:
            raise ValueError(f"{target.name}: colour match LUT contents changed.")
        return target
    target = paths.validate_output(target, allowed_existing_suffixes=frozenset())
    target.write_bytes(data)
    return target


def _lut_filter(path: Path) -> str:
    # Escape the option parser first, then the enclosing filtergraph parser.
    option = "".join("\\" + char if char in "\\':" else char for char in str(path))
    escaped = "".join("\\" + char if char in "\\'[],;" else char for char in option)
    return f"lut3d=file={escaped}:interp=tetrahedral"


def _decode(source: _Source, expression: str, count: int, width: int, height: int,
            seconds: float, commands: list[tuple[str, ...]], *, owner: str,
            in_s: float | None = None, channels: int = CHANNELS) -> np.ndarray:
    seek = [] if in_s is None else ["-ss", f"{in_s:.12f}", "-t", f"{seconds:.12f}"]
    command = ["ffmpeg", "-nostdin", "-v", "error", "-filter_threads", str(FILTER_THREADS),
               *seek,
               *(source.hdr.input_args if source.hdr else []), "-i", str(source.path),
               "-map", "0:g:0" if source.grid else f"0:{source.info.video_index}",
               "-an", "-vf", expression, "-fps_mode", "passthrough", "-frames:v", str(count),
               "-f", "rawvideo", "-"]
    try:
        result = ffmpeg.run(command, binary=True, timeout_s=max(
            MIN_TIMEOUT, TIMEOUT_OVERHEAD + TIMEOUT_PER_SECOND * seconds))
    except ProcessingError as exc:
        from yap_studio.stages.assemble_picture import PictureError

        error = PictureError(f"{owner}: colour measurement failed; {exc.message}")
        error.suggested_action = "Set color_match false on this shot and run again."
        raise error from exc
    commands.append(tuple(result.args))
    pixels = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width, channels)
    if len(pixels) != count:
        raise ValueError(f"{source.path.name}: colour measurement did not decode every sample.")
    return pixels


def _geometry(item: ResolvedElement | _Shot, *, frame: Canvas | None = None
              ) -> tuple[str, int, int]:
    """Measure only visible pixels, excluding contain padding and blurred fill."""
    width, height = MEASURE_WIDTH, MEASURE_HEIGHT
    if frame is not None and frame.width > frame.height:
        width, height = MEASURE_HEIGHT, MEASURE_WIDTH
    media = item.payload["media"]
    fit = media.get("fit", "cover") if item.kind == "insert" else item.payload.get("fit", "cover")
    if item.kind == "insert" and fit == "split":
        height = round(width * DEFAULT_SEAM_Y / WIDTH)
    if item.kind in {"overlay", "card_window"}:
        _, _, box_width, box_height = item.payload["box_px"]
        width = max(1, round(MEASURE_HEIGHT * box_width / max(box_width, box_height)))
        height = max(1, round(MEASURE_HEIGHT * box_height / max(box_width, box_height)))
    if item.kind == "card_window":
        focus_x, focus_y = media.get("focus", [0.5, 0.5])
        crop = ""
        if "crop" in media:
            x, y, w, h = media["crop"]
            crop = f"crop={w}:{h}:{x}:{y}:exact=1,"
        cover = (f"{crop}scale=w='ceil(max({width}/(iw*sar),{height}/ih)*iw*sar/"
                 f"{WINDOW_ALIGNMENT})*{WINDOW_ALIGNMENT}':"
                 f"h='ceil(max({width}/(iw*sar),{height}/ih)*ih/"
                 f"{WINDOW_ALIGNMENT})*{WINDOW_ALIGNMENT}',setsar=1,"
                 f"crop={width}:{height}:x='(iw-{width})*{focus_x:.12f}':"
                 f"y='(ih-{height})*{focus_y:.12f}':exact=1,setsar=1,format=rgba")
        radius = min(float(item.payload["radius"]) + WINDOW_BLEED, box_width / 2, box_height / 2)
        x, y = f"X*{box_width}/{width}", f"Y*{box_height}/{height}"
        dx = f"max(max({radius}-{x},{x}-({box_width - 1}-{radius})),0)"
        dy = f"max(max({radius}-{y},{y}-({box_height - 1}-{radius})),0)"
        mask = f"255*clip({radius}+{MASK_EDGE}-sqrt({dx}*{dx}+{dy}*{dy}),0,1)"
        return (cover + f",geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='{mask}',format=rgba",
                width, height)
    crop = ""
    if fit == "crop":
        x, y, w, h = media["crop"]
        crop = f"crop={w}:{h}:{x}:{y}:exact=1,"
    contain = fit == "contain"
    scale = (f"scale={width}:{height}:force_original_aspect_ratio="
             f"{'decrease' if contain else 'increase'}")
    if not contain:
        scale += f",crop={width}:{height}:exact=1"
    pixel_format = "rgba" if item.kind == "overlay" else "rgb24"
    return crop + scale + f",setsar=1,format={pixel_format}", width, height


def match_broll(root: Path, timeline: ResolvedTimeline, cut: _Source,
                sources: dict[str, _Source], work: Path,
                *, prefix: Callable[[_Source], str], still_loop: Callable[[_Source], str],
                commands: list[tuple[str, ...]], placements: Sequence[dict] = (),
                color_match: bool | None = None,
                window_sources: Mapping[tuple[str, int], _Source] | None = None,
                window_rect: Callable[[dict], tuple[int, int, int, int]] | None = None,
                ) -> tuple[dict[MatchKey, str], list[dict[str, Any]]]:
    """Decode the take once and return filters keyed by element or card window."""
    items = [_Shot(item.id, item.id, item.kind, item.start_frame, item.end_frame,
                   item.payload, sources[item.id]) for item in timeline.elements
             if item.kind in {"insert", "overlay"} and item.payload.get("color_match") is True]
    media_windows = [(placement["id"], index, window) for placement in placements
                     for index, window in enumerate(placement.get("windows", []))
                     if window["source"]["kind"] == "media"]
    if media_windows:
        if color_match is None:
            from yap_studio.project.status import ProjectContext

            color_match = ProjectContext(root).render_style().data["broll"]["color_match"] == "on"
        if color_match:
            assert window_sources is not None and window_rect is not None
            for identifier, index, window in media_windows:
                key = (identifier, index)
                start = window["start_frame"]
                items.append(_Shot(key, f"{identifier} window {index}", "card_window", start,
                                   start + window["frames"], {
                                       "media": window["source"], "box_px": window_rect(window),
                                       "radius": window["radius"],
                                   }, window_sources[key]))
    if not items:
        return {}, []
    frame = Canvas(cut.info.width, cut.info.height, 0)
    measure_width, measure_height = ((MEASURE_HEIGHT, MEASURE_WIDTH)
                                    if frame.width > frame.height else
                                    (MEASURE_WIDTH, MEASURE_HEIGHT))
    talking_heads = {take.id for take in load_project(root / "project.json").takes
                     if not take.voiceover}
    if not talking_heads:
        return {}, [{"id": item.id, "kind": item.kind, "skipped": "voiceover video"}
                    for item in items]
    spans = load_edit_list(paths.validate_input(root / "cut/spans.json"))
    plan = Plan.build(spans, fps=Fraction(FPS), sample_rate=SAMPLE_RATE)
    if plan.total_frames != timeline.total_frames:
        raise ValueError("Cut spans and the resolved timeline have different frame counts.")
    windows = [(piece.out_f0, piece.out_f0 + piece.n_frames) for piece in plan.segments
               if piece.source_id in talking_heads]
    candidates = {}
    for item in items:
        length = item.end_frame - item.start_frame
        nearby = [item.start_frame - TAKE_MARGIN,
                  *(item.start_frame + length * n // d for n, d in SAMPLE_POSITIONS),
                  item.end_frame - 1 + TAKE_MARGIN]
        clamped = {max(0, min(frame, timeline.total_frames - 1)) for frame in nearby}
        candidates[item.id] = sorted(frame for frame in clamped
                                     if any(first <= frame < last for first, last in windows))
    frames = sorted({frame for selected in candidates.values() for frame in selected})
    take_pixels = {}
    if frames:
        # The cut has already been conformed to SDR and the delivery frame clock.
        select = "+".join(f"eq(n,{frame})" for frame in frames)
        pixels = _decode(cut, f"select='{select}',scale={measure_width}:{measure_height},"
                         "format=rgb24", len(frames), measure_width, measure_height,
                         (frames[-1] + 1) / FPS, commands,
                         owner=", ".join(item.id for item in items if candidates[item.id]))
        take_pixels = dict(zip(frames, pixels, strict=True))
    filters, records = {}, []
    for item in items:
        selected = candidates[item.id]
        reason = "no talking-head picture near this moment" if not selected else ""
        take = measure(np.array([take_pixels[frame] for frame in selected])) if selected else {}
        if take and take["median"] < MIN_TAKE_LUMA:
            reason = "talking-head picture is too dark"
        if reason:
            records.append({"id": item.id, "kind": item.kind, "skipped": reason})
            continue
        source = item.source
        length = item.end_frame - item.start_frame
        local = [0] if source.image else sorted({length * n // d for n, d in SAMPLE_POSITIONS})
        media = item.payload["media"]
        fit = (insert_fit(media, source.info, (frame.width, frame.height))
               if item.kind == "insert" else item.payload.get("fit", "cover"))
        if item.kind == "insert" and fit != media.get("fit", "cover"):
            item = replace(item, payload={**item.payload, "media": {**media, "fit": fit}})
        geometry, width, height = _geometry(item, frame=frame)
        # Contain has no padding, so its actual sample dimensions follow the source.
        if fit == "contain":
            factor = min(width / source.info.width, height / source.info.height)
            width = max(1, round(source.info.width * factor))
            height = max(1, round(source.info.height * factor))
        if source.image:
            expression = ",".join(part for part in (
                source.params, source.hdr.filter_prefix if source.hdr else None, geometry,
            ) if part)
            seconds, in_s = 1 / FPS, None
        else:
            speed = media.get("speed", 1)
            seconds, in_s = length / FPS * speed + READ_MARGIN, media.get("in_s", 0)
            select = "+".join(f"eq(n,{frame})" for frame in local)
            expression = (f"{prefix(source)},setpts=(PTS-STARTPTS)/{speed:.12f},"
                          f"fps={FPS}:start_time=0:round=near,tpad=stop_mode=clone:stop={length},"
                          f"select='{select}',{geometry}")
        pixels = _decode(source, expression, len(local), width, height,
                         seconds, commands, owner=item.id, in_s=in_s,
                         channels=CHANNELS if item.kind == "insert" else RGBA_CHANNELS)
        if item.kind != "insert":
            visible = pixels[..., CHANNELS] >= VISIBLE_ALPHA
            if np.count_nonzero(visible) < visible.size * MIN_VISIBLE_FRACTION:
                records.append({"id": item.id, "kind": item.kind,
                                "skipped": "too few visible pixels"})
                continue
            pixels = pixels[..., :CHANNELS][visible]
        broll = measure(pixels)
        params = match_parameters(take, broll)
        lut = write_lut(work, params)
        if lut is None:
            records.append({"id": item.id, "kind": item.kind, "skipped": "unchanged"})
            continue
        filters[item.key] = _lut_filter(lut) + (
            ",format=yuv444p" if item.kind == "insert" else ",format=rgba")
        records.append({"id": item.id, "kind": item.kind, "take_frames": selected,
                        "take": take, "broll": broll,
                        **{key: value for key, value in params.items() if key != "pivot"},
                        "lut": lut.stem})
    return filters, records
