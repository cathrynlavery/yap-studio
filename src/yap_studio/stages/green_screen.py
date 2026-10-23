"""Opt-in background layouts, deterministic colour measurements and picture layers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

from yap_studio import ffmpeg
from yap_studio.plan import snap_frame
from yap_studio.stages.tall_box import SHRINK_BOX_PX, chosen_framing

if TYPE_CHECKING:
    from yap_studio.captions import Phrase
    from yap_studio.project.model import CutWord
    from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline

# Keep the face large while exposing about half of the background.
BOTTOM_SCALE = 0.65
# Corner layouts leave most of the background visible.
CORNER_SCALE = 0.45
# Soften frame edges that sit inside the canvas.
EDGE_FADE_PX = 24
# A short eased entrance lasts 0.3 seconds at 30 fps.
WIPE_FRAMES = 9
# Feather the moving boundary across several rows.
WIPE_FEATHER_PX = 64
# Keep the background light subtle at the silhouette.
WRAP_STRENGTH = 0.25
# Limit the light band to the silhouette edge.
WRAP_EDGE_SIGMA_PX = 6
# Remove background detail with a blur at quarter size.
WRAP_BLUR_SIGMA_PX = 12
# Pull chroma gently toward the background.
PULL_STRENGTH = 0.12
# Bound each channel correction in 8-bit code values.
PULL_CLAMP = 12
# Preserve two thirds of the original skin colour.
SKIN_KEEP = 2 / 3
# Refresh look receipts when mask and blur filters change.
MASK_RULE = 3
# Refresh receipts when layout face and caption rules change.
LAYOUT_RULE = 3
# Allow slow look filters enough time on a busy machine.
LOOK_DECODE_FACTOR = 8


def person_box(layout: str) -> tuple[int, int, int, int]:
    """Return the speaker frame on the 1080 by 1920 canvas."""
    scale = BOTTOM_SCALE if layout == "bottom" else CORNER_SCALE
    width, height = round(1080 * scale), round(1920 * scale)
    return {"full": (0, 0, 1080, 1920),
            "bottom": ((1080 - width) // 2, 1920 - height, width, height),
            "corner_left": (0, 1920 - height, width, height),
            "corner_right": (1080 - width, 1920 - height, width, height)}[layout]


def fade_edges(layout: str) -> tuple[str, ...]:
    """Fade only speaker frame edges that lie inside the canvas."""
    return {"full": (), "bottom": ("left", "right", "top"),
            "corner_left": ("right", "top"), "corner_right": ("left", "top")}[layout]


def uses_look(payload: dict) -> bool:
    """Keep legacy payloads on their existing rendering path."""
    return any(payload.get(key, legacy) != legacy for key, legacy in (
        ("layout", "full"), ("blend", "none"), ("enter", "cut"), ("motion", "none")))


def decode_seconds(item: ResolvedElement) -> float:
    """Budget look filtering by window length while preserving legacy allowances."""
    return (item.end_frame - item.start_frame) / 30 * LOOK_DECODE_FACTOR \
        if uses_look(item.payload) else 0.0


def layout_windows(timeline: ResolvedTimeline) -> list[tuple[int, int, tuple, int]]:
    """Share half-open speaker layout windows with every face consumer."""
    return [(item.start_frame, item.end_frame, person_box(item.payload["layout"]),
             item.payload.get("wipe_frames", WIPE_FRAMES
                              if item.payload.get("enter") == "wipe" else 0))
            for item in timeline.elements if item.kind == "background"
            and item.payload.get("layout", "full") != "full"] + [
                (item.start_frame, item.end_frame, SHRINK_BOX_PX, 0)
                for item in timeline.elements if item.kind == "overlay"
                and item.payload.get("placement") == "tall_behind"
                and chosen_framing(item.payload) == "shrink"]


def caption_breaks(words: Sequence[CutWord],
                   windows: Sequence[tuple[int, int, tuple, int]]) -> set[str]:
    """Start a new phrase at each layout boundary on the anchor frame grid."""
    word_frames = [(word, snap_frame(Fraction(str(word.start)), Fraction(30))) for word in words]
    breaks = set()
    for start, end, _, _ in windows:
        for frame in (start, end):
            word = next((word for word, start_frame in word_frames if start_frame >= frame), None)
            if word is not None:
                breaks.add(word.id)
    return breaks


def clip_layout_phrases(phrases: Sequence[Phrase],
                        windows: Sequence[tuple[int, int, tuple, int]]) -> list[Phrase]:
    """Stop trailing holds at layout boundaries without cutting the last word."""
    boundaries = sorted({frame for start, end, _, _ in windows for frame in (start, end)})
    clipped = []
    for phrase in phrases:
        start = snap_frame(Fraction(str(phrase.start)), Fraction(30))
        end = snap_frame(Fraction(str(phrase.end)), Fraction(30))
        boundary = next((frame for frame in boundaries if start < frame <= end), None)
        # A hold can pass the boundary while snapping to the same end frame.
        if boundary is not None and phrase.end > boundary / 30:
            phrase = replace(phrase, end=max(boundary / 30, phrase.words[-1].end))
        clipped.append(phrase)
    return clipped


def resolve_payload(payload: dict, style: dict) -> None:
    """Append resolved look fields without supplying authored defaults."""
    if not uses_look(payload):
        return
    if payload.get("layout", "full") != "full":
        payload["person_box_px"] = list(person_box(payload["layout"]))
    if payload.get("enter") == "wipe":
        payload["wipe_frames"] = WIPE_FRAMES
    if payload.get("motion") == "push":
        payload["push_scale"] = style["broll"]["still_push_scale"]
        payload["push_direction"] = "out" if style["broll"]["still_push"] == "out" else "in"


def findings(payload: dict, frames: int, *, image: bool) -> list[tuple[str, str, str]]:
    """Reject moving clip pushes and entrances without enough hold time."""
    result = []
    if payload.get("motion") == "push" and not image:
        result.append(("background_motion", "motion: push needs a photo or screenshot",
                       "Set motion to none, or use a photo or screenshot."))
    if payload.get("enter") == "wipe" and frames < 2 * WIPE_FRAMES:
        result.append(("background_enter", "enter: wipe needs at least 18 frames",
                       "Use enter: cut, or end the background later."))
    return result


def rule_constants() -> dict:
    """Identify every rule that affects the rendered look."""
    return {key: value for key, value in globals().items()
            if key.isupper() and key != "TYPE_CHECKING"}


def wipe_alpha(frame: int, row: int) -> float:
    """Evaluate the eased top-down wipe on its local frame clock."""
    progress = 1 - (1 - min(frame + 1, WIPE_FRAMES) / WIPE_FRAMES) ** 3
    return max(0., min(1., (progress * (1920 + WIPE_FEATHER_PX) - row) / WIPE_FEATHER_PX))


def pull_offset(background: tuple, person: tuple) -> tuple[int, int, int]:
    """Round and clamp a BT.709 chroma correction without a luma pull."""
    weights = (.2126, .7152, .0722)
    bg_y = sum(a * b for a, b in zip(background, weights, strict=True))
    person_y = sum(a * b for a, b in zip(person, weights, strict=True))
    return tuple(max(-PULL_CLAMP, min(PULL_CLAMP, round(PULL_STRENGTH * (
        bg - bg_y - her + person_y)))) for bg, her in zip(background, person, strict=True))


def _split(graph: Any, branch: str, count: int = 2) -> list[str]:
    labels = [f"green{len(graph.filters)}_{i}" for i in range(count)]
    if count == 1:
        return [branch]
    graph.filters.append(f"[{branch}]split={count}" + "".join(f"[{label}]" for label in labels))
    return labels


def _push(graph: Any, branch: str, item: Any) -> str:
    """Match the photo insert push math on a fitted background canvas."""
    frames = item.end_frame - item.start_frame
    travel = item.payload["push_scale"] - 1
    scale = (f"1+{travel:.12f}*({frames - 1}-n)/{max(frames - 1, 1)}"
             if item.payload["push_direction"] == "out"
             else f"1+{travel:.12f}*n/{max(frames - 1, 1)}")
    sw, sh = f"trunc(4320*({scale})/2)*2", f"trunc(7680*({scale})/2)*2"
    crop = ("pad=4752:8448:x='(ow-iw)/2':y='(oh-ih)/2':eval=frame,"
            "crop=4320:7680:x=216:y=384:exact=1"
            if item.payload["push_direction"] == "out" else
            f"crop=4320:7680:x='({sw}-4320)/2':y='({sh}-7680)/2':exact=1")
    return graph.filter(branch, "scale=4320:7680:flags=bicubic,"
                        f"scale=w='{sw}':h='{sh}':eval=frame,{crop},"
                        "scale=1080:1920:flags=bicubic:out_range=tv")


def background_layer(graph: Any, source: Any, item: Any, *, push: bool = True) -> str:
    """Fit and trim background media on a local 30 fps clock."""
    from yap_studio.stages.assemble_picture import _overlay

    branch = _overlay(graph, source, item, fade=False, motion=False)
    branch = graph.filter(branch, "setpts=N,format=gbrp")
    if push and item.payload.get("motion") == "push":
        branch = _push(graph, branch, item)
    return branch


def _wipe(graph: Any, rgba: str, frames: int) -> str:
    if not frames:
        return rgba
    picture, alpha = _split(graph, rgba)
    progress = f"(1-pow(1-min(N+1,{frames})/{frames},3))"
    alpha = graph.filter(alpha, "alphaextract,geq=lum='lum(X,Y)*clip(("
                         f"{progress}*(1920+{WIPE_FEATHER_PX})-Y)/{WIPE_FEATHER_PX},0,1)':"
                         f"enable='lt(n,{frames})'")
    return graph.filter(f"{picture}][{alpha}", "alphamerge")


def _pull(graph: Any, picture: str, offset: tuple) -> str:
    original, adjusted, skin = _split(graph, picture, 3)
    adjusted = graph.filter(adjusted, "lutrgb=" + ":".join(
        f"{channel}='clip(val+({value}),0,255)'"
        for channel, value in zip("rgb", offset, strict=True)))
    skin = graph.filter(skin, "scale=out_color_matrix=bt709:out_range=full,format=yuv444p")
    cb, cr = f"green_cb{len(graph.filters)}", f"green_cr{len(graph.filters)}"
    graph.filters.append(f"[{skin}]extractplanes=u+v[{cb}][{cr}]")
    cb = graph.filter(cb, "lut=c0='255*between(val,77,127)'")
    cr = graph.filter(cr, "lut=c0='255*between(val,133,173)'")
    skin = graph.filter(f"{cb}][{cr}", "blend=all_mode=multiply")
    # Keep the mask gray until its strength applies to every colour channel.
    skin = graph.filter(skin, "gblur=sigma=1.5,format=gray,"
                        f"lut=c0='val*{SKIN_KEEP}',format=gbrp")
    return graph.filter(f"{adjusted}][{original}][{skin}", "maskedmerge=planes=7")


def _wrap_band(graph: Any, alpha: str, blurred: str) -> str:
    blurred = graph.filter(blurred, "lut=c0='255-val'")
    band = graph.filter(f"{alpha}][{blurred}", "blend=all_mode=multiply")
    # Keep the picture in colour when the merge inputs can scale.
    return graph.filter(band, "format=gray,"
                        f"lut=c0='min(2*val,255)*{WRAP_STRENGTH}',format=gbrp")


def person_layer(graph: Any, person: str, background: str, item: Any, offset: tuple) -> str:
    """Transform picture and firm alpha together, then wrap in canvas space."""
    layout = item.payload.get("layout", "full")
    x, y, width, height = person_box(layout)
    person = graph.filter(person, f"trim=start_frame={item.start_frame}:"
                          f"end_frame={item.end_frame},setpts=N,format=rgba")
    picture, alpha = _split(graph, person)
    picture = graph.filter(picture, "format=gbrp")
    alpha = graph.filter(alpha, "alphaextract")
    if item.payload.get("blend") == "wrap":
        picture = _pull(graph, picture, offset)
    picture = graph.filter(picture, f"scale={width}:{height}:flags=bicubic")
    alpha = graph.filter(alpha, f"scale={width}:{height}:flags=bicubic")
    factors = {"left": "X", "right": "W-1-X", "top": "Y"}
    fade = "*".join(f"clip(({factors[edge]})/{EDGE_FADE_PX},0,1)"
                    for edge in fade_edges(layout))
    if fade:
        mask = f"green_edge{len(graph.filters)}"
        graph.filters.append(f"nullsrc=s={width}x{height}:r=30,format=gray,"
                             f"geq=lum='255*{fade}':enable='lt(n,1)',"
                             "trim=end_frame=1,loop=loop=-1:size=1,"
                             f"trim=end_frame={item.end_frame - item.start_frame},"
                             f"settb=1/30,setpts=N[{mask}]")
        alpha = graph.filter(f"{alpha}][{mask}", "blend=all_mode=multiply")
    picture = graph.filter(picture, f"pad=1080:1920:{x}:{y}:color=black")
    alpha = graph.filter(alpha, f"pad=1080:1920:{x}:{y}:color=black")
    if item.payload.get("blend") == "wrap":
        alpha, edge, blurred = _split(graph, alpha, 3)
        blurred = graph.filter(blurred, f"gblur=sigma={WRAP_EDGE_SIGMA_PX}")
        band = _wrap_band(graph, edge, blurred)
        background = graph.filter(background, "format=gbrp,scale=270:480:flags=area,"
                                  f"gblur=sigma={WRAP_BLUR_SIGMA_PX / 4:g},"
                                  "scale=1080:1920:flags=bicubic")
        picture = graph.filter(f"{picture}][{background}][{band}", "maskedmerge=planes=7")
    result = graph.filter(f"{picture}][{alpha}", "alphamerge,format=rgba")
    result = _wipe(graph, result, item.payload.get("wipe_frames", 0))
    return graph.filter(result, f"settb=1/30,setpts=N+{item.start_frame}")


def _sample(graph: Any, branch: str, frames: int, commands: list,
            *, decode_seconds: float) -> np.ndarray:
    picks = sorted({round(i * max(frames - 1, 0) / 7) for i in range(8)})
    select = "+".join(f"eq(n,{frame})" for frame in picks)
    branch = graph.filter(branch, f"select='{select}',scale=54:96:flags=area,format=rgba")
    command = ["ffmpeg", "-nostdin", "-v", "error", "-filter_complex_threads", "1",
               *graph.inputs, "-filter_complex", ";".join(graph.filters), "-map", f"[{branch}]",
               "-an", "-fps_mode", "passthrough", "-frames:v", str(len(picks)),
               "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"]
    result = ffmpeg.run(command, binary=True, timeout_s=max(60, 30 + decode_seconds * 12))
    commands.append(tuple(result.args))
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, 96, 54, 4)


def measure_pull(source: Any, cut: Any, matte: Any, item: Any, commands: list) -> dict:
    """Measure fixed frame picks before layout scaling or entrance animation."""
    from yap_studio.stages.assemble_picture import _Graph

    frames = item.end_frame - item.start_frame
    graph = _Graph()
    bg = _sample(graph, background_layer(graph, source, item, push=False),
                 1 if source.image else frames, commands,
                 decode_seconds=source.info.video_duration_s or frames / 30)
    graph = _Graph()
    picture = graph.filter(graph.input(cut, owner="cut"),
                           f"trim=start_frame={item.start_frame}:end_frame={item.end_frame},"
                           "settb=1/30,setpts=N,format=rgb24")
    alpha = graph.filter(graph.input(matte, owner="cut matte"),
                         f"trim=start_frame={item.start_frame}:end_frame={item.end_frame},"
                         "settb=1/30,setpts=N,format=gray,"
                         "lut=c0='clip((val-16)*255/80,0,255)',gblur=sigma=1")
    person = _sample(graph, graph.filter(f"{picture}][{alpha}", "alphamerge"), frames, commands,
                     decode_seconds=item.end_frame / 30)
    pixels = person[..., :3][person[..., 3] >= 200]
    her = tuple(float(v) for v in pixels.mean(axis=0)) if len(pixels) else (0., 0., 0.)
    background = tuple(float(v) for v in bg[..., :3].mean(axis=(0, 1, 2)))
    return {"background_rgb": list(background), "person_rgb": list(her),
            "offset_rgb": list(pull_offset(background, her)), "strength": PULL_STRENGTH,
            "clamp": PULL_CLAMP, "skin_keep": SKIN_KEEP}


def prepare(graph: Any, backgrounds: list, sources: dict, people: dict,
            cut: Any, matte: Any, commands: list,
            *, layer_filter: str | None = None) -> tuple[list, list, tuple]:
    """Build look layers while leaving every legacy branch unchanged."""
    look = [item for item in backgrounds if uses_look(item.payload)]
    if not look:
        return backgrounds, [], ()
    legacy = [item for item in backgrounds if not uses_look(item.payload)]
    branches = _split(graph, people.pop("background"), len(look) + bool(legacy))
    if legacy:
        people["background"] = branches.pop()
    layers, measurements = [], []
    for item, person in zip(look, branches, strict=True):
        blend = item.payload.get("blend") == "wrap"
        pull = measure_pull(sources[item.id], cut, matte, item, commands) if blend else None
        background = background_layer(graph, sources[item.id], item)
        if layer_filter is not None:
            # The whole-picture grade reaches backgrounds before her light wraps them.
            background = graph.filter(graph.filter(background, "format=rgba"), layer_filter)
        wrap_source = None
        if blend:
            background, wrap_source = _split(graph, background)
        speaker = person_layer(graph, person, wrap_source, item,
                               tuple(pull["offset_rgb"]) if pull else (0, 0, 0))
        background = _wipe(graph, graph.filter(background, "format=rgba"),
                           item.payload.get("wipe_frames", 0))
        background = graph.filter(background, f"settb=1/30,setpts=N+{item.start_frame}")
        layers.append((item, background, speaker))
        measurements.append({"id": item.id, "layout": item.payload.get("layout", "full"),
                             "person_box_px": list(person_box(item.payload.get("layout", "full"))),
                             "enter": item.payload.get("enter", "cut"),
                             "motion": item.payload.get("motion", "none"),
                             "wrap": {"strength": WRAP_STRENGTH,
                                      "edge_sigma_px": WRAP_EDGE_SIGMA_PX,
                                      "blur_sigma_px": WRAP_BLUR_SIGMA_PX} if blend else None,
                             "pull": pull})
    return legacy, layers, tuple(measurements)
