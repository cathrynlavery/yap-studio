"""Composite exact picture windows into the silent delivery stream for assembly."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from yap_studio import ffmpeg, letters, paths, receipt
from yap_studio.canvas import Canvas, canvas
from yap_studio.cards_video import window_rect
from yap_studio.edl import load_edit_list
from yap_studio.errors import ProcessingError, YapError
from yap_studio.faces.model import Box
from yap_studio.grade import (
    GradeSpec,
    describe,
    graph_alpha_filter,
    graph_filter,
    rgba_filter,
    write_cube,
)
from yap_studio.hdr import HdrConversion, conversion, hdr_kind
from yap_studio.matte import chroma
from yap_studio.matte.check import measure_behind_layers
from yap_studio.matte.edge_colour import background_matte_filter, prepare_edge_layers
from yap_studio.matte.project import CutMatteError, ensure_cut_matte
from yap_studio.media import catalog
from yap_studio.plan import Plan, PlanSegment
from yap_studio.project.model import load_project
from yap_studio.project.receipts import projection_sha256
from yap_studio.render.moments import overlay_moments
from yap_studio.stages import green_screen, tall_box
from yap_studio.stages.behind_window import number_redraw
from yap_studio.stages.broll_match import match_broll
from yap_studio.stages.broll_split import (
    check_insert_bounds,
    check_split_voiceover,
    split_frame,
    split_layouts,
)
from yap_studio.stages.captions import captions
from yap_studio.stages.graphics import graphics
from yap_studio.stages.number_visibility import measure_number_layers
from yap_studio.stages.timeline import (
    ResolvedElement,
    ResolvedTimeline,
    _is_image,
    resolve_timeline,
)
from yap_studio.text_hook import TEXT_HOOK_ID
from yap_studio.zoom_curve import SCALE_LIMIT, ZoomCurve

WIDTH = 1080
HEIGHT = 1920
FPS = 30
CODEC = "libx264"
CRF = 16
PRESET = "slow"
PIX_FMT = "yuv420p"
COLOUR = "bt709"
MAX_ZOOM = 1.10
STILL_RULE = 1
_HEIF_EXTENSIONS = {".heic", ".heif"}
_STILL_PARAMS = ("range", "colorspace", "color_primaries", "color_trc", "chroma_location")


class PictureError(YapError):
    """Refuse a picture whose inputs or encoded stream break the assembly contract."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message, error_type="picture_error", code="picture_invalid",
            suggested_action="Correct the named picture inputs, re-run their stages and assemble.",
        )


class PictureProcessingError(PictureError, ProcessingError):
    """Keep process exit semantics while naming the failed picture input and fix."""

    def __init__(self, error: ProcessingError, owners: list[tuple[str, ResolvedElement | None]],
                 project_dir: Path, *, frames: tuple[int, int] | None = None) -> None:
        ProcessingError.__init__(self, error.command, error.returncode, error.full_stderr,
                                 code=error.code)
        owners = sorted(owners, key=lambda entry: entry[1] is None)
        if owners:
            self.args = (f"{'; '.join(owner for owner, _ in owners)}: {self.message}",)
        rerun = f"yap-studio run {shlex.quote(str(project_dir))}"
        if frames is not None:
            rerun += f" --frames {frames[0]}-{frames[1]}"
        element = next((item for _, item in owners if item is not None), None)
        if element is None:
            self.suggested_action = f"Run yap-studio doctor to check ffmpeg, then {rerun} again."
            return
        if element.lane == "engine" and element.id.startswith("steps-"):
            self.suggested_action = f"Set moves.jump_cuts to hard, then run {rerun} again."
            return
        remove = (f"yap-studio timeline remove --lane {shlex.quote(element.lane)} "
                  f"--id {shlex.quote(element.id)} {shlex.quote(str(project_dir))}")
        resource = any(text in error.full_stderr.lower() for text in (
            "resource temporarily unavailable", "cannot allocate memory", "too many open files",
        ))
        media = element.payload.get("media", {})
        filename = media.get("source_name") or Path(media.get("path", owners[0][0])).name
        self.suggested_action = (
            f"Close other apps that are rendering video and run {rerun} again. "
            f"If it stops at the same element again, remove or replace {element.id} "
            f"with {remove}, then run {rerun} again."
            if resource else
            f"Check that {filename} opens. Replace or remove {element.id} with {remove}, "
            f"then run {rerun} again."
        )


@dataclass(frozen=True)
class PictureResult:
    """Verified picture length and process/conversion evidence for the final receipt."""

    total_frames: int
    hold_frames: int
    commands: tuple[tuple[str, ...], ...]
    warnings: tuple[str, ...]
    conversions: tuple[dict[str, Any], ...]
    behind_layers: tuple[dict[str, Any], ...] = ()
    frames: tuple[int, int] | None = None
    stills: tuple[dict[str, Any], ...] = ()
    grade: dict[str, Any] | None = None
    grade_compare: tuple[dict[str, Any], ...] = ()
    splits: tuple[dict[str, Any], ...] = ()
    broll_match: tuple[dict[str, Any], ...] = ()
    cutout: dict[str, Any] | None = None
    green_screen: tuple[dict[str, Any], ...] = ()
    moments: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible receipt fields; total_frames includes the hold."""
        result = {"total_frames": self.total_frames, "hold_frames": self.hold_frames,
                  "commands": [list(command) for command in self.commands],
                  "warnings": list(self.warnings), "conversions": list(self.conversions)}
        if self.behind_layers:
            result["behind_layers"] = list(self.behind_layers)
        if self.frames is not None:
            result["frames"] = list(self.frames)
        if self.stills:
            result["stills"] = list(self.stills)
        if self.grade is not None:
            result["grade"] = self.grade
        if self.grade_compare:
            result["grade_compare"] = list(self.grade_compare)
        if self.splits:
            result["splits"] = list(self.splits)
        if self.broll_match:
            result["broll_match"] = list(self.broll_match)
        if self.cutout is not None:
            result["cutout"] = self.cutout
        if self.green_screen:
            result["green_screen"] = list(self.green_screen)
        if self.moments:
            result["moments"] = list(self.moments)
        return result


@dataclass(frozen=True)
class _Source:
    """A probed picture source with its shared HDR rule applied before scaling."""

    path: Path
    info: ffmpeg.MediaInfo
    hdr: HdrConversion | None
    grid: bool = False
    still: bool = False
    image: bool = False
    params: str | None = None
    hdr_converted: bool = False


class _Graph:
    """Build independent decode branches and one ordered overlay chain."""

    def __init__(self) -> None:
        self.inputs: list[str] = []
        self.filters: list[str] = []
        self.count = 0
        self.owners: dict[int, tuple[str, ResolvedElement | None]] = {}

    def input(self, source: _Source, *, input_args: tuple[str, ...] = (),
              owner: str | None = None, element: ResolvedElement | None = None) -> str:
        """Add an input with hardware options local to that source only."""
        index = self.count
        self.count += 1
        media = element.payload.get("media", {}) if element else {}
        filename = media.get("source_name") or Path(media.get("path", source.path)).name
        self.owners[index] = (f"{element.id} ({filename})" if element else
                              owner or str(source.path), element)
        self.inputs.extend([*input_args, *(source.hdr.input_args if source.hdr else []),
                            "-i", str(source.path)])
        return f"{index}:g:0" if source.grid else f"{index}:{source.info.video_index}"

    def filter(self, source: str, expression: str) -> str:
        """Connect a branch to a unique output pad."""
        label = f"p{len(self.filters)}"
        self.filters.append(f"[{source}]{expression}[{label}]")
        return label

    def overlay(self, base: str, layer: str, start: int, end: int,
                *, x: int | str = 0, y: int = 0, enable: str | None = None) -> str:
        """Overlay only the exclusive frame window, passing the base at layer EOF."""
        return self.filter(
            f"{base}][{layer}",
            f"overlay=x={x}:y={y}:eof_action=pass:repeatlast=0:format=auto:"
            f"enable='{enable if enable is not None else f'between(n,{start},{end - 1})'}'",
        )


def _timeout(seconds: float) -> float:
    """Allow decoding long source prefixes as well as the delivery encode."""
    return max(60.0, 30.0 + 12.0 * seconds)


def _probe(path: Path, commands: list[tuple[str, ...]]) -> ffmpeg.MediaInfo:
    """Probe stream metadata and retain the probe command alongside the encode."""
    info = ffmpeg.probe(path, count_frames=False)
    commands.append((ffmpeg.find_tool("ffprobe"), "-hide_banner", "-v", "error",
                     "-show_format", "-show_streams", "-of", "json", str(path)))
    return info


def _source(path: Path, commands: list[tuple[str, ...]],
            conversions: list[dict[str, Any]], *, owner: str,
            convert_hdr: bool = True) -> _Source:
    """Validate picture and select the shared HDR conversion before rendering."""
    path = paths.validate_input(path)
    info = _probe(path, commands)
    try:
        grid = catalog.heif_grid(path, commands=commands, strict=True)
    except (YapError, OSError, ValueError, TypeError) as exc:
        raise PictureError(f"{owner}: photo's tile grid could not be read.") from exc
    if grid is not None:
        width, height, rotation = grid
        info = replace(info, width=width, height=height, rotation=rotation,
                       orientation="vertical" if height > width else
                       "horizontal" if width > height else "square")
    if not info.has_video or not info.width or not info.height:
        raise PictureError(f"{owner}: source has no usable picture dimensions.")
    selected = conversion(info, commands=commands) if convert_hdr else None
    if selected:
        conversions.append({"input": owner, "kind": hdr_kind(info), **selected.describe()})
    still = path.suffix.lower() in _HEIF_EXTENSIONS
    return _Source(path, info, selected, grid is not None, still, still or _is_image(info, path))


def _still_params(properties: Any) -> str:
    """Validate stored frame properties and restore them without changing pixels."""
    if not isinstance(properties, dict):
        raise ValueError("Missing frame properties")
    for field in ("pix_fmt", *_STILL_PARAMS):
        value = properties[field]
        if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9_-]+", value):
            raise ValueError(f"Invalid frame property: {field}")
    if properties["pix_fmt"] in {"unknown", "unspecified", "none"}:
        raise ValueError("Missing pixel format")
    return "setparams=" + ":".join(f"{field}={properties[field]}" for field in _STILL_PARAMS)


def _decoded_properties(stderr: str, photo: Path) -> dict[str, str]:
    """Read the single measured frame, refusing missing or ambiguous properties."""
    lines = [line for line in stderr.splitlines() if "[showinfo@still @" in line]
    pictures = [line for line in lines if re.search(r"\bn:\s*", line)]
    colours = [line for line in lines if "color_range:" in line]
    try:
        if len(pictures) != 1 or len(colours) != 1:
            raise ValueError("Expected one measured frame")
        frame = re.search(r"\bn:\s*0\s.*\bfmt:(\S+)\s+cl:(\S+)", pictures[0])
        colour = re.search(r"color_range:(\S+)\s+color_space:(\S+)\s+"
                           r"color_primaries:(\S+)\s+color_trc:(\S+)\s*$", colours[0])
        if frame is None or colour is None:
            raise ValueError("Unreadable measured frame")
        values = (*frame.groups(), *colour.groups())
        properties = dict(zip(("pix_fmt", "chroma_location", "range", "colorspace",
                               "color_primaries", "color_trc"), values, strict=True))
        properties = {field: "unknown" if value == "unspecified" else value
                      for field, value in properties.items()}
        _still_params(properties)
    except (ValueError, KeyError, TypeError) as exc:
        raise PictureError(f"{photo.name}: decoded frame properties could not be read.") from exc
    return properties


def _frame_sha256(path: Path, photo: Path) -> str:
    """Read exactly one raw frame hash produced by ffmpeg."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        frames = [line.split(",") for line in lines if line and not line.startswith("#")]
        if ("#hash: SHA256" not in lines or len(frames) != 1 or len(frames[0]) != 6
                or not re.fullmatch(r"[0-9a-f]{64}", frames[0][-1].strip())):
            raise ValueError("Expected one frame hash")
    except (OSError, ValueError) as exc:
        raise PictureError(f"{photo.name}: decoded frame hash could not be read.") from exc
    return frames[0][-1].strip()


def _cached_still(folder: Path, key: str, source_sha256: str
                  ) -> tuple[Path, dict[str, str]] | None:
    """Validate the complete published entry before opening its frame."""
    if not folder.exists():
        return None
    try:
        record = json.loads((folder / "receipt.json").read_bytes())
        frame = folder / "frame.mkv"
        _still_params(record["properties"])
        valid = (record["schema"] == "yap-studio/still-cache@1" and record["key"] == key
                 and record["source_sha256"] == source_sha256
                 and isinstance(record["pixel_sha256"], str)
                 and re.fullmatch(r"[0-9a-f]{64}", record["pixel_sha256"]) is not None
                 and record["sha256"] == receipt.sha256_file(frame))
    except (OSError, ValueError, KeyError, TypeError):
        valid = False
    if not valid:
        raise PictureError(f"Incomplete still cache: {folder}. Delete it and run again.")
    return paths.validate_input(frame), record["properties"]


def _prepare_still(source: _Source, project_dir: Path, commands: list[tuple[str, ...]],
                   stills: list[dict[str, Any]], *, source_sha256: str,
                   owner: ResolvedElement, ffmpeg_version: str | None,
                   frames: tuple[int, int] | None = None) -> _Source:
    """Cache one upright lossless HEIF frame before any composite transforms."""
    photo = Path(owner.payload.get("media", {}).get("source_name") or source.path.name)
    key = projection_sha256({
        "source_sha256": source_sha256, "grid": source.grid,
        "size": [source.info.width, source.info.height], "rotation": source.info.rotation,
        "hdr": source.hdr.describe() if source.hdr else None,
        "ffmpeg": ffmpeg_version, "rule": STILL_RULE,
    })
    cache = paths.validate_output_dir(project_dir / ".yap/cache/stills")
    folder = cache / key
    cached = _cached_still(folder, key, source_sha256)
    reused = cached is not None
    if cached is None:
        with tempfile.TemporaryDirectory(prefix=".still-", dir=cache) as temporary:
            staging = Path(temporary)
            graph = _Graph()
            prefix = source.hdr.filter_prefix.split(",") if source.hdr else ["null"]
            decode = []
            for index, expression in enumerate(prefix):
                previous = prefix[index - 1] if index else ""
                if expression.startswith("format=") and not previous.startswith(
                    ("scale", "zscale", "hwdownload"),
                ):
                    decode.append("scale")
                decode.append(expression)
            branch = graph.filter(graph.input(source, element=owner),
                                  ",".join(decode))
            formats = re.findall(r"(?:^|,)format=([a-z0-9]+)", source.hdr.filter_prefix) \
                if source.hdr else []
            pixel_format = formats[-1] if formats else source.info.pix_fmt
            codec = (["-c:v", "libx264", "-qp", "0", "-preset", "ultrafast"]
                     if (pixel_format or "").startswith("yuvj")
                     else ["-c:v", "ffv1", "-level", "3"])
            storage = "setparams=range=tv" if "ffv1" in codec else "null"
            measured = staging / "reference.framehash"
            graph.filters.extend([f"[{branch}]split=2[store][measure]",
                                  f"[store]{storage}[stored]",
                                  "[measure]showinfo@still[measured]"])
            command = ["ffmpeg", "-nostdin", "-v", "info", "-filter_complex_threads", "1",
                       *graph.inputs, "-filter_complex", ";".join(graph.filters),
                       "-map", "[stored]", "-an", "-frames:v", "1", *codec,
                       "-threads", "1", "-pix_fmt", "+", "-map_metadata", "-1",
                       "-metadata:s:v:0", "rotate=0", str(staging / "frame.mkv"),
                       "-map", "[measured]", "-an", "-frames:v", "1", "-c:v", "rawvideo",
                       "-threads", "1", "-pix_fmt", "+", "-f", "framehash", str(measured)]
            try:
                result = ffmpeg.run(command, timeout_s=300)
            except ProcessingError as exc:
                raise PictureProcessingError(exc, list(graph.owners.values()), project_dir,
                                             frames=frames) from exc
            commands.append(tuple(result.args))
            properties = _decoded_properties(result.stderr, photo)
            pixel_sha256 = _frame_sha256(measured, photo)
            verified = staging / "verified.framehash"
            try:
                result = ffmpeg.run([
                    "ffmpeg", "-nostdin", "-v", "error", "-i", str(staging / "frame.mkv"),
                    "-map", "0:v:0", "-an", "-vf", _still_params(properties), "-frames:v", "1",
                    "-c:v", "rawvideo", "-threads", "1", "-pix_fmt", "+", "-f", "framehash",
                    str(verified),
                ], timeout_s=300)
            except ProcessingError as exc:
                raise PictureProcessingError(exc, list(graph.owners.values()), project_dir,
                                             frames=frames) from exc
            commands.append(tuple(result.args))
            if _frame_sha256(verified, photo) != pixel_sha256:
                raise PictureError(f"{photo.name}: photo could not be prepared exactly.")
            paths.atomic_write_text(staging / "receipt.json", json.dumps({
                "schema": "yap-studio/still-cache@1", "key": key,
                "sha256": receipt.sha256_file(staging / "frame.mkv"),
                "source_sha256": source_sha256,
                "properties": properties, "pixel_sha256": pixel_sha256,
            }) + "\n")
            try:
                os.rename(staging, folder)
            except OSError:
                if not folder.is_dir():
                    raise
                reused = True
            cached = _cached_still(folder, key, source_sha256)
            assert cached is not None
    frame, properties = cached
    info = _probe(frame, commands)
    if (info.width, info.height, info.rotation) != (source.info.width, source.info.height, 0):
        raise PictureError(f"Invalid still cache picture: {folder}. Delete it and run again.")
    stills.append({"input": owner.id, "cache": key, "reused": reused})
    # Match the properties restored by _still_loop after lossless storage.
    info = replace(info, color_range=properties["range"], color_space=properties["colorspace"])
    return _Source(frame, info, None, still=True, image=True, params=_still_params(properties),
                   hdr_converted=source.hdr is not None or source.hdr_converted)


def _prefix(source: _Source) -> str:
    """Convert tagged source colour before compositing on the BT.709 cut."""
    prefix = source.hdr.filter_prefix + "," if source.hdr else ""
    full = source.info.color_range in {"pc", "jpeg"} or (source.info.pix_fmt or "").startswith(
        "yuvj",
    )
    matrix = source.info.color_space
    if not source.hdr and not source.hdr_converted and full and matrix != "gbr":
        options = ["in_range=pc", "out_range=tv"]
        if matrix not in {None, "unknown", "unspecified"}:
            options.extend([f"in_color_matrix={matrix}", "out_color_matrix=bt709"])
        prefix += "scale=" + ":".join(options) + ","
    return prefix + "setpts=PTS-STARTPTS"


def _cover(size: tuple[int, int] = (WIDTH, HEIGHT)) -> str:
    """Fill the canvas without changing the source aspect ratio."""
    width, height = size
    return (f"scale={width}:{height}:force_original_aspect_ratio=increase:"
            f"force_divisible_by=2,crop={width}:{height}")


def _still_loop(source: _Source) -> str:
    """Restore measured properties before holding a decoded HEIF frame."""
    return ((source.params + "," if source.params else "")
            + ("loop=loop=-1:size=1," if source.still else ""))


def _insert(graph: _Graph, source: _Source, item: ResolvedElement, *,
            size: tuple[int, int] = (WIDTH, HEIGHT)) -> str:
    """Decode, retime and fit one montage part in its already resolved window."""
    media = item.payload["media"]
    width, height = size
    check_insert_bounds(item, size)
    frames = item.end_frame - item.start_frame
    image = source.image
    branch = graph.filter(
        graph.input(source, input_args=("-loop", "1", "-framerate", str(FPS))
                    if image and not source.still else (), element=item),
        f"{_still_loop(source)}{_prefix(source)},"
        f"trim=start={0 if image else media.get('in_s', 0):.12f},"
        f"setpts=(PTS-STARTPTS)/{1 if image else media.get('speed', 1):.12f},"
        f"fps={FPS}:start_time=0:round=near,trim=end_frame={frames},"
        f"tpad=stop_mode=clone:stop=1,trim=end_frame={frames},setsar=1",
    )
    fit = media.get("fit", "cover")
    if fit == "crop":
        x, y, w, h = media["crop"]
        branch = graph.filter(branch, f"crop={w}:{h}:{x}:{y}:exact=1,{_cover(size)}")
    elif fit == "contain":
        background, foreground = f"b{len(graph.filters)}", f"f{len(graph.filters)}"
        graph.filters.append(f"[{branch}]split=2[{background}][{foreground}]")
        background = graph.filter(background, f"{_cover(size)},boxblur=24:2")
        foreground = graph.filter(
            foreground, f"scale={width}:{height}:force_original_aspect_ratio=decrease:"
            "force_divisible_by=2",
        )
        branch = graph.filter(f"{background}][{foreground}",
                              "overlay=x=(W-w)/2:y=(H-h)/2:format=auto:shortest=1")
    else:
        branch = graph.filter(branch, _cover(size))
    # Full chroma resolution preserves odd privacy-box edges and sizes exactly.
    branch = graph.filter(branch, "format=yuv444p")
    for x, y, w, h in media.get("blur", []):
        base, box = f"b{len(graph.filters)}", f"f{len(graph.filters)}"
        graph.filters.append(f"[{branch}]split=2[{base}][{box}]")
        radius = min(16, int(min(w, h)) // 2 - 1)
        box = graph.filter(box, f"crop={w}:{h}:{x}:{y}:exact=1,boxblur={radius}:2")
        branch = graph.filter(f"{base}][{box}", f"overlay=x={x}:y={y}:format=auto:shortest=1")
    if item.payload.get("motion") == "push":
        base_w, base_h = 4 * width, 4 * height
        travel = item.payload["push_scale"] - 1
        progress = f"n/{max(frames - 1, 1)}"
        scale = (f"1+{travel:.12f}*({frames - 1}-n)/{max(frames - 1, 1)}"
                 if item.payload["push_direction"] == "out"
                 else f"1+{travel:.12f}*{progress}")
        scaled_w = f"trunc({base_w}*({scale})/2)*2"
        scaled_h = f"trunc({base_h}*({scale})/2)*2"
        if item.payload["push_direction"] == "out":
            # crop keeps its first input size, so pad the shrinking frame to a fixed size.
            pad_w, pad_h = round(base_w * MAX_ZOOM), round(base_h * MAX_ZOOM)
            crop = (f"pad={pad_w}:{pad_h}:x='(ow-iw)/2':y='(oh-ih)/2':eval=frame,"
                    f"crop={base_w}:{base_h}:x={(pad_w - base_w) // 2}:"
                    f"y={(pad_h - base_h) // 2}:exact=1")
        else:
            crop = (f"crop={base_w}:{base_h}:x='({scaled_w}-{base_w})/2':"
                    f"y='({scaled_h}-{base_h})/2':exact=1")
        branch = graph.filter(branch, f"scale={base_w}:{base_h}:flags=bicubic")
        branch = graph.filter(
            branch,
            f"scale=w='{scaled_w}':h='{scaled_h}':eval=frame,"
            f"{crop},"
            f"scale={width}:{height}:flags=bicubic:out_range=tv",
        )
    return graph.filter(branch, f"setsar=1,settb=1/{FPS},setpts=N+{item.start_frame}")


def _zoom(graph: _Graph, source: _Source, item: ResolvedElement, *, first: int,
          last: int, src_first: int, crop_x: float, from_cut: bool) -> str:
    """Scale directly from the take, retaining a canvas-relative fixed zoom centre."""
    assert source.info.width is not None and source.info.height is not None
    factor = max(WIDTH / source.info.width, HEIGHT / source.info.height)
    base_w = math.ceil(source.info.width * factor / 2) * 2
    base_h = math.ceil(source.info.height * factor / 2) * 2
    if from_cut:
        base_w, base_h = WIDTH, HEIGHT
    curve = ZoomCurve.from_payload(item.start_frame, item.end_frame, item.payload)
    scale, cx, cy = curve.expressions(first)
    # Keep constant crop centers identical for existing swell renders.
    center_x = (f"{WIDTH * float(cx):.12f}" if "(" not in cx else f"{WIDTH}*({cx})")
    center_y = (f"{HEIGHT * float(cy):.12f}" if "(" not in cy else f"{HEIGHT}*({cy})")
    if curve.peak is not None:
        center_x = f"{WIDTH * item.payload['center'][0]:.12f}"
        center_y = f"{HEIGHT * item.payload['center'][1]:.12f}"
    scaled_w, scaled_h = f"trunc({base_w}*{scale}/2)*2", f"trunc({base_h}*{scale}/2)*2"
    # Scale counts its first frame as 1; crop counts it as 0. Preserve the legacy clock.
    resize_scale = scale if curve.peak is not None else scale.replace("(n+", "(n-1+")
    resize_w = f"trunc({base_w}*{resize_scale}/2)*2"
    resize_h = f"trunc({base_h}*{resize_scale}/2)*2"
    resize_flags = ":flags=lanczos" if curve.peak is None else ""
    origin_x, origin_y = (base_w - WIDTH) * crop_x, (base_h - HEIGHT) / 2
    # crop's iw/ih retain initial link dimensions when upstream scale changes per frame.
    ratio_x, ratio_y = f"({scaled_w}/{base_w})", f"({scaled_h}/{base_h})"
    return graph.filter(
        graph.input(source, element=item),
        f"{_prefix(source)},fps={FPS}:start_time=0:round=near,"
        f"trim=start_frame={src_first}:end_frame={src_first + last - first},"
        f"setpts=PTS-STARTPTS,scale=w='{resize_w}':h='{resize_h}':eval=frame{resize_flags},"
        f"crop={WIDTH}:{HEIGHT}:x='{origin_x:.12f}*{ratio_x}+"
        f"{center_x}*({ratio_x}-1)':y='{origin_y:.12f}*{ratio_y}+"
        f"{center_y}*({ratio_y}-1)':exact=1,"
        f"setsar=1,settb=1/{FPS},setpts=N+{first}",
    )


def _overlay_size(info: ffmpeg.MediaInfo, item: ResolvedElement) -> tuple[int, int]:
    """Fit media to the destination box without changing its aspect ratio."""
    assert info.width is not None and info.height is not None
    _, _, width, height = item.payload["box_px"]
    contain = item.payload.get("fit", "cover") == "contain"
    factor = (min if contain else max)(width / info.width, height / info.height)
    rounding = round if contain else math.ceil
    return max(1, rounding(info.width * factor)), max(1, rounding(info.height * factor))


def _overlay(graph: _Graph, source: _Source, item: ResolvedElement,
             *, fade: bool = True, motion: bool = True) -> str:
    """Decode silent media through the shared overlay filter chain."""
    image = source.image
    page = item.payload.get("motion") == "scroll" or item.payload.get("placement") == "tall_behind"
    return graph.filter(
        graph.input(source, input_args=("-loop", "1", "-framerate", str(FPS))
                    if image and not source.still and not page else (), element=item),
        _overlay_filters(source, item, image=image, fade=fade, motion=motion),
    )


def _overlay_filters(source: _Source, item: ResolvedElement, *, image: bool,
                     fade: bool = True, motion: bool = True) -> str:
    """Fit media, drift about the box centre and fade on the local frame grid.

    ``motion=False`` holds the framing still whatever the payload says.
    """
    media = item.payload["media"]
    frames = item.end_frame - item.start_frame
    _, _, width, height = item.payload["box_px"]
    if item.payload.get("motion") == "scroll" or item.payload.get("placement") == "tall_behind":
        page_h = tall_box.page_height(source.info.width, source.info.height, width)
        padding = (f",pad={width}:{height}:0:0:color="
                   f"0x{item.payload['fill_color'].lstrip('#')}"
                   if page_h < height and tall_box.chosen_framing(item.payload) == "shrink"
                   else "")
        y = (tall_box.scroll_expression(frames, page_h, height)
             if motion and item.payload.get("motion") == "scroll" else "0")
        filters = [
            (source.params + "," if source.params else "") + _prefix(source),
            f"trim=end_frame=1,format=rgba,scale={width}:{page_h}{padding},"
            f"loop=loop=-1:size=1,trim=end_frame={frames},settb=1/{FPS},setpts=N,"
            f"crop={width}:{height}:x=0:y='{y}':exact=1,setsar=1",
        ]
        if motion and item.payload.get("motion") == "drift":
            factor = f"(1+0.04*n/{max(1, frames - 1)})"
            sw, sh = f"round({width}*{factor})", f"round({height}*{factor})"
            filters.append(f"scale=w='{sw}':h='{sh}':eval=frame,"
                           f"crop={width}:{height}:x='({sw}-{width})/2':"
                           f"y='({sh}-{height})/2':exact=1")
        if fade and item.payload.get("placement") != "tall_behind":
            fade_frames = min(6, max(1, frames // 2))
            filters.append(f"fade=t=in:start_frame=0:nb_frames={fade_frames}:alpha=1,"
                           f"fade=t=out:start_frame={frames - fade_frames}:"
                           f"nb_frames={fade_frames}:alpha=1")
        filters.append(f"settb=1/{FPS},setpts=N+{item.start_frame},format=rgba")
        return ",".join(filters)
    filters = [
        f"{_still_loop(source)}{_prefix(source)},"
        f"trim=start={0 if image else media.get('in_s', 0):.12f},"
        f"setpts=(PTS-STARTPTS)/{1 if image else media.get('speed', 1):.12f},"
        f"fps={FPS}:start_time=0:round=near,trim=end_frame={frames},"
        f"tpad=stop_mode=clone:stop={frames},trim=end_frame={frames},"
        "setsar=1,format=rgba",
    ]
    fitted_w, fitted_h = _overlay_size(source.info, item)
    fit = f"scale={fitted_w}:{fitted_h}"
    if item.payload.get("fit", "cover") == "contain":
        fit += f",pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black@{0 if fade else 1}"
    else:
        fit += f",crop={width}:{height}:exact=1"
    filters.append(fit)
    if motion and item.payload.get("motion", "none") == "drift":
        factor = f"(1+0.04*n/{max(1, frames - 1)})"
        scaled_w, scaled_h = f"round({width}*{factor})", f"round({height}*{factor})"
        filters.append(
            f"scale=w='{scaled_w}':h='{scaled_h}':eval=frame,"
            f"crop={width}:{height}:x='({scaled_w}-{width})/2':"
            f"y='({scaled_h}-{height})/2':exact=1",
        )
    if fade:
        fade_frames = min(6, max(1, frames // 2))
        filters.append(f"fade=t=in:start_frame=0:nb_frames={fade_frames}:alpha=1,"
                       f"fade=t=out:start_frame={frames - fade_frames}:"
                       f"nb_frames={fade_frames}:alpha=1")
    filters.append(f"settb=1/{FPS},setpts=N+{item.start_frame},"
                   f"format={'rgba' if fade else 'rgb24,format=rgba'}")
    return ",".join(filters)


def overlay_alpha_runs(path: Path, item: ResolvedElement) -> list[tuple[int, int, Box]]:
    """Measure consecutive overlay alpha bounds on the canvas and output frame clock."""
    source = _source(path, [], [], owner=item.id)
    graph = _Graph()
    alpha = graph.filter(_overlay(graph, source, item), "alphaextract")
    frames = item.end_frame - item.start_frame
    x, y, width, height = item.payload["box_px"]
    stream = ffmpeg.run_stream([
        "ffmpeg", "-nostdin", "-v", "error", "-filter_complex_threads", "1",
        *graph.inputs, "-filter_complex", ";".join(graph.filters), "-map", f"[{alpha}]",
        "-an", "-frames:v", str(frames), "-pix_fmt", "gray", "-fps_mode", "passthrough",
        "-f", "rawvideo", "pipe:1",
    ], timeout_s=_timeout(source.info.video_duration_s or frames / FPS), chunk_size=64 * 1024)
    runs: list[tuple[int, int, Box]] = []
    try:
        frame_size = width * height
        buffer = bytearray(frame_size)
        target = memoryview(buffer)
        used = local_frame = 0
        for chunk in stream:
            raw = memoryview(chunk)
            while raw:
                size = min(len(raw), frame_size - used)
                target[used:used + size] = raw[:size]
                used += size
                raw = raw[size:]
                if used != frame_size:
                    continue
                pixels = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width)
                rows, columns = np.nonzero(pixels >= 64)
                if rows.size:
                    left, top = int(columns.min()), int(rows.min())
                    box = Box(x + left, y + top,
                              int(columns.max()) - left + 1, int(rows.max()) - top + 1)
                    frame = item.start_frame + local_frame
                    if runs and runs[-1][1] == frame and runs[-1][2] == box:
                        runs[-1] = (runs[-1][0], frame + 1, box)
                    else:
                        runs.append((frame, frame + 1, box))
                local_frame += 1
                used = 0
            del raw, chunk
        if used:
            raise PictureError(f"{item.id}: overlay alpha decode ended with a partial frame.")
    finally:
        stream.close()
    return runs


def _frames(spans: Iterable[tuple[int, int]]) -> str:
    """Join half-open frame windows into one ffmpeg enable expression."""
    return "+".join(f"between(n,{first},{last - 1})" for first, last in spans)


def _insert_inputs(root: Path, timeline: ResolvedTimeline, commands: list[tuple[str, ...]],
                   conversions: list[dict[str, Any]]) -> dict[str, _Source]:
    """Recheck media identity and geometry before any layer or picture render."""
    sources = {}
    for item in timeline.elements:
        if item.kind not in {"insert", "overlay", "background"}:
            continue
        media = item.payload["media"]
        path = paths.validate_input(root / media["path"])
        if receipt.sha256_file(path) != media["sha256"]:
            raise PictureError(f"{item.id}: {item.kind} media hash changed.")
        source = _source(path, commands, conversions, owner=item.id)
        if hdr_kind(source.info) and media.get("hdr", "auto") != "auto":
            raise PictureError(f"{item.id}: HDR picture requires media.hdr: auto.")
        if media.get("fit", "cover") == "crop":
            x, y, w, h = media["crop"]
            if x + w > source.info.width or y + h > source.info.height:
                raise PictureError(f"{item.id}: source crop extends beyond the displayed picture.")
        for x, y, w, h in media.get("blur", []):
            if min(w, h) < 8 or x + w > WIDTH or y + h > HEIGHT:
                raise PictureError(f"{item.id}: blur boxes must be at least 8 px "
                                   "and inside canvas.")
        sources[item.id] = source
    return sources


def _zoom_inputs(root: Path, timeline: ResolvedTimeline, cut: _Source,
                 commands: list[tuple[str, ...]], conversions: list[dict[str, Any]],
                 warnings: list[str]) -> list[tuple[ResolvedElement, PlanSegment, _Source, float]]:
    """Map zoom windows across every retained source range, including repeated takes."""
    zooms = [item for item in timeline.elements if item.kind == "zoom"]
    if not zooms:
        return []
    spans = load_edit_list(paths.validate_input(root / "cut/spans.json"))
    plan = Plan.build(spans, fps=Fraction(FPS), sample_rate=48_000)
    if plan.total_frames != timeline.total_frames:
        raise PictureError("Cut spans and the resolved timeline have different frame counts.")
    project = load_project(paths.validate_input(root / "project.json"))
    crops = {take.id: take.crop_x for take in project.takes}
    sources: dict[str, _Source] = {}
    result = []
    for item in zooms:
        curve = ZoomCurve.from_payload(item.start_frame, item.end_frame, item.payload)
        if not 1 < curve.max_scale() <= SCALE_LIMIT:
            raise PictureError(f"{item.id}: zoom peak must be greater than 1 "
                               f"and at most {SCALE_LIMIT:g}.")
        for piece in plan.segments:
            first = max(item.start_frame, piece.out_f0)
            last = min(item.end_frame, piece.out_f0 + piece.n_frames)
            if first >= last:
                continue
            name = piece.source_id
            if name is None or spans.sources[name].path is None or name not in crops:
                raise PictureError(f"{item.id}: cut span is missing a project take.")
            if name not in sources:
                sources[name] = _source(spans.sources[name].path, commands, conversions,
                                        owner=name, convert_hdr=False)
            source = sources[name]
            assert source.info.width is not None and source.info.height is not None
            # The axis that limits the cover crop determines whether native pixels remain.
            enough = min(source.info.width / WIDTH, source.info.height / HEIGHT) > 1
            if not enough:
                warnings.append(f"{item.id}: take {name} is not larger than the canvas in the "
                                "cropped axis; rendering zoom from the cut.")
                source = cut
            elif hdr_kind(source.info) and source.hdr is None:
                selected = conversion(source.info, commands=commands)
                assert selected is not None
                conversions.append({"input": name, "kind": hdr_kind(source.info),
                                    **selected.describe()})
                source = sources[name] = _Source(source.path, source.info, selected)
            result.append((item, piece, source, crops[name]))
    return result


def _layer(graph: _Graph, root: Path, placement: dict[str, Any], timeline: ResolvedTimeline,
           commands: list[tuple[str, ...]], conversions: list[dict[str, Any]]) -> str:
    """Keep an alpha layer at its own box size and delay it onto the output grid."""
    source = _source(root / placement["path"], commands, conversions, owner=placement["path"])
    start, frames = placement["start_frame"], placement["frames"]
    if (source.info.width != placement["width"] or source.info.height != placement["height"]
            or start < 0 or frames <= 0 or start + frames > timeline.total_frames
            or source.info.video_frames != frames or source.info.fps != FPS):
        raise PictureError(f"{placement['path']}: alpha layer box or frame window is invalid.")
    return graph.filter(graph.input(source, owner=placement["path"]),
                        f"trim=end_frame={frames},setsar=1,settb=1/{FPS},setpts=N+{start}")


def _window_motion_valid(window: dict) -> bool:
    """Keep a moving window on the same canvas and output frame clock."""
    motion = window["motion"]
    if (not isinstance(motion, dict)
            or set(motion) != {"start_frame", "frames", "to", "ease"}
            or type(motion["start_frame"]) is not int or type(motion["frames"]) is not int
            or motion["frames"] < 2 or motion["start_frame"] < window["start_frame"]
            or motion["start_frame"] + motion["frames"] > window["start_frame"] + window["frames"]
            or motion["ease"] != "cubic_in_out"
            or not isinstance(motion["to"], list) or len(motion["to"]) != 4
            or any(type(value) is not int or value % 2 for value in motion["to"])):
        return False
    x, y, width, height = motion["to"]
    return min(x, y) >= 0 and min(width, height) > 0 and x + width <= WIDTH and y + height <= HEIGHT


def _person_edges(graph: _Graph, video: Path, matte: Path, work: Path,
                  behind: list[dict[str, Any]], backgrounds: list[ResolvedElement],
                  commands: list[tuple[str, ...]], conversions: list[dict[str, Any]], *,
                  frames: tuple[int, int] | None = None, cube: Path | None = None,
                  compares: Sequence[ResolvedElement] = (),
                  tall: Sequence[ResolvedElement] = (),
                  tall_shrink: Sequence[ResolvedElement] = ()) -> dict[str, str]:
    """Correct and grade person edges outside comparison windows."""
    person = {}
    windows = {
        "graphic": [(p["start_frame"], p["start_frame"] + p["frames"]) for p in behind],
        "background": [(item.start_frame, item.end_frame) for item in backgrounds],
    }
    if tall:
        windows["tall"] = [(item.start_frame, item.end_frame) for item in tall]
    if tall_shrink:
        windows["tall_shrink"] = [(item.start_frame, item.end_frame) for item in tall_shrink]
    for kind, spans in windows.items():
        if not spans:
            continue
        branch = f"person_{kind}"
        for compare in compares:
            spans = [(left, right) for first, last in spans
                     for left, right in ((first, min(last, compare.start_frame)),
                                         (max(first, compare.end_frame), last)) if left < right]
        for first, count, path in prepare_edge_layers(video, matte, spans, work,
                                                      commands=commands, frames=frames):
            layer = _source(path, commands, conversions, owner="person edge", convert_hdr=False)
            pad = graph.filter(graph.input(layer), f"settb=1/{FPS},setpts=N+{first},setsar=1")
            if cube is not None:
                pad = graph.filter(pad, graph_alpha_filter(cube))
            branch = graph.overlay(branch, pad, first, first + count)
        person[kind] = branch
    return person


def _tall_composite(graph: _Graph, room: str, person: str, page: str,
                    item: ResolvedElement) -> str:
    """Keep the real room or replace it with the page behind a smaller person."""
    start, end = item.start_frame, item.end_frame
    framing = tall_box.chosen_framing(item.payload)
    if framing not in {"keep", "shrink"}:
        raise PictureError("Resolve automatic tall framing in assembly before rendering.")
    if framing == "keep":
        room = graph.filter(room, f"trim=start_frame={start}:end_frame={end},"
                            f"settb=1/{FPS},setpts=N+{start}")
        px, py, _, _ = tall_box.TALL_BOX_PX
        room = graph.overlay(room, page, start, end, x=px, y=py, enable="1")
        # Restore the unscaled person only over the page, preserving the surrounding room.
        _, _, page_w, page_h = tall_box.TALL_BOX_PX
        person = graph.filter(person, f"trim=start_frame={start}:end_frame={end},"
                              f"settb=1/{FPS},setpts=N+{start},"
                              f"crop={page_w}:{page_h}:{px}:{py}:exact=1")
        return graph.overlay(room, person, start, end, x=px, y=py, enable="1")
    graph.filters.append(f"[{room}]nullsink")
    fill = f"tall_paper{len(graph.filters)}"
    color = item.payload["fill_color"].lstrip("#")
    graph.filters.append(f"color=c=0x{color}:s={WIDTH}x{HEIGHT}:r={FPS},"
                         f"trim=end_frame={end - start},settb=1/{FPS},"
                         f"setpts=N+{start},format=rgba[{fill}]")
    px, py, _, _ = item.payload["box_px"]
    room = graph.overlay(fill, page, start, end, x=px, y=py, enable="1")
    x, y, width, height = tall_box.SHRINK_BOX_PX
    person = graph.filter(person, f"trim=start_frame={start}:end_frame={end},"
                          f"settb=1/{FPS},setpts=N+{start},format=rgba")
    rgb, alpha = green_screen._split(graph, person)
    rgb = graph.filter(rgb, f"format=gbrp,scale={width}:{height}:flags=bicubic")
    alpha = graph.filter(alpha, f"alphaextract,scale={width}:{height}:flags=bicubic")
    mask = f"tall_edge{len(graph.filters)}"
    fade = tall_box.PERSON_EDGE_FADE_PX
    graph.filters.append(f"nullsrc=s={width}x{height}:r={FPS},format=gray,"
                         f"geq=lum='255*clip(X/{fade},0,1)*clip((W-1-X)/{fade},0,1)"
                         f"*clip(Y/{fade},0,1)':enable='lt(n,1)',"
                         f"trim=end_frame=1,loop=loop=-1:size=1,trim=end_frame={end - start},"
                         f"settb=1/{FPS},setpts=N+{start}[{mask}]")
    alpha = graph.filter(f"{alpha}][{mask}", "blend=all_mode=multiply")
    person = graph.filter(f"{rgb}][{alpha}", "alphamerge=ts_sync_mode=nearest")
    return graph.overlay(room, person, start, end, x=x, y=y, enable="1")


def _window_sources(root: Path, placements: list[dict], cut: _Source,
                    commands: list[tuple[str, ...]], conversions: list[dict[str, Any]]
                    ) -> dict[tuple[str, int], _Source]:
    """Validate window geometry and retained media before decoding any clips."""
    sources = {}
    for placement in placements:
        for index, window in enumerate(placement.get("windows", [])):
            owner = f"{placement['id']} window {index}"
            integers = ("x", "y", "width", "height", "start_frame", "frames",
                        "fade_in_frames", "fade_out_frames")
            if (any(type(window.get(key)) is not int for key in integers)
                    or any(window[key] % 2 for key in ("width", "height"))
                    or min(window["x"], window["y"], window["start_frame"]) < 0
                    or min(window["width"], window["height"], window["frames"],
                           window["fade_in_frames"]) < 1
                    or window["fade_out_frames"] < 0
                    or (window["fade_out_frames"] == 0 and "motion" not in window)
                    or ("motion" in window and not _window_motion_valid(window))
                    or window["x"] + window["width"] > WIDTH
                    or window["y"] + window["height"] > HEIGHT
                    or window["start_frame"] + window["frames"] > cut.info.video_frames
                    or isinstance(window.get("radius"), bool)
                    or not isinstance(window.get("radius"), (int, float))
                    or not math.isfinite(window["radius"]) or window["radius"] < 0
                    or window["fade_in_frames"] + window["fade_out_frames"] > window["frames"]):
                raise PictureError(f"{owner}: invalid window box or frame counts.")
            source = window["source"]
            if source["kind"] == "self":
                if "crop" in source:
                    raise PictureError(f"{owner}: self takes no crop.")
                sources[placement["id"], index] = cut
                continue
            if source["kind"] != "media":
                raise PictureError(f"{owner}: choose media or self.")
            path = paths.validate_input(root / source["path"])
            if not path.resolve().is_relative_to(root.resolve()):
                raise PictureError(f"{owner}: use a retained clip inside the rendering folder.")
            if receipt.sha256_file(path) != source["sha256"]:
                raise PictureError(f"{owner}: the clip changed; update its sha256.")
            decoded = _source(path, commands, conversions, owner=owner)
            duration = decoded.info.video_duration_s
            if decoded.image or duration is None or not math.isfinite(duration) or duration <= 0:
                raise PictureError(f"{owner}: use a moving clip with known duration.")
            speed, in_s = source.get("speed", 1), source.get("in_s", 0)
            if (isinstance(speed, bool) or not isinstance(speed, (int, float))
                    or not math.isfinite(speed) or not 0.5 <= speed <= 2
                    or isinstance(in_s, bool) or not isinstance(in_s, (int, float))
                    or not math.isfinite(in_s) or in_s < 0
                    or in_s + window["frames"] / FPS * speed > duration + 1e-9):
                raise PictureError(f"{owner}: use speed 0.5 to 2 and enough clip duration.")
            sources[placement["id"], index] = decoded
    return sources


def _window_rect(window: dict) -> tuple[int, int, int, int]:
    """Bleed two pixels under the frame with an even box inside the canvas."""
    x, y = max(0, window["x"] - 2), max(0, window["y"] - 2)
    width = (min(WIDTH, window["x"] + window["width"] + 2) - x) // 2 * 2
    height = (min(HEIGHT, window["y"] + window["height"] + 2) - y) // 2 * 2
    return x, y, width, height


def _window_coordinate(window: dict, column: int, clock: str) -> str:
    """Use the planner's exact rectangles for scale and overlay expressions."""
    motion = window["motion"]
    rects = [_window_rect({**window, **dict(zip(("x", "y", "width", "height"),
                                              window_rect(window, frame), strict=True))})
             for frame in range(motion["start_frame"], motion["start_frame"] + motion["frames"])]
    expression = str(rects[-1][column])
    for index in range(len(rects) - 2, -1, -1):
        expression = (f"if(lt({clock},{motion['start_frame'] + index + 1}),"
                      f"{rects[index][column]},{expression})")
    return expression


def _window_clip(graph: _Graph, source: _Source, window: dict, *, owner: str) -> str:
    """Cover and mask once, then scale a moving window on its exact frame clock."""
    start, frames = window["start_frame"], window["frames"]
    media = window["source"]
    if media["kind"] == "self":
        timing = (f"fps={FPS}:start_time=0:round=near,"
                  f"trim=start_frame={start}:end_frame={start + frames},setpts=PTS-STARTPTS")
    else:
        timing = (f"trim=start={media.get('in_s', 0):.12f},"
                  f"setpts=(PTS-STARTPTS)/{media.get('speed', 1):.12f},"
                  f"fps={FPS}:start_time=0:round=near,trim=end_frame={frames},"
                  f"tpad=stop_mode=clone:stop={frames},trim=end_frame={frames}")
    motion = window.get("motion")
    target = ({**window, **dict(zip(("x", "y", "width", "height"), motion["to"], strict=True))}
              if motion else window)
    _, _, width, height = _window_rect(target)
    focus_x, focus_y = media.get("focus", [0.5, 0.5])
    crop = ""
    if "crop" in media:
        x, y, w, h = media["crop"]
        crop = f"crop={w}:{h}:{x}:{y}:exact=1,"
    branch = graph.filter(
        graph.input(source, owner=owner),
        f"{_prefix(source)},{timing},{crop}"
        f"scale=w='ceil(max({width}/(iw*sar),{height}/ih)*iw*sar/2)*2':"
        f"h='ceil(max({width}/(iw*sar),{height}/ih)*ih/2)*2',setsar=1,"
        f"crop={width}:{height}:x='(iw-{width})*{focus_x:.12f}':"
        f"y='(ih-{height})*{focus_y:.12f}':exact=1,setsar=1,format=rgba",
    )
    radius = float(window["radius"]) + 2
    if motion:
        radius *= width / _window_rect(window)[2]
    radius = min(radius, width / 2, height / 2)
    mask = f"window_mask{len(graph.filters)}"
    dx = f"max(max({radius}-X,X-({width - 1}-{radius})),0)"
    dy = f"max(max({radius}-Y,Y-({height - 1}-{radius})),0)"
    edge = f",geq=lum='255*clip({radius}+0.5-sqrt({dx}*{dx}+{dy}*{dy}),0,1)'" if radius else ""
    graph.filters.append(
        f"color=c=white:s={width}x{height}:r={FPS},trim=end_frame=1,format=gray{edge},"
        f"loop=loop=-1:size=1,settb=1/{FPS},setpts=N[{mask}]",
    )
    branch = graph.filter(f"{branch}][{mask}", "alphamerge=shortest=1")
    fade_out = ("" if motion else
                f"fade=t=out:start_frame={frames - window['fade_out_frames']}:"
                f"nb_frames={window['fade_out_frames']}:alpha=1,")
    branch = graph.filter(
        branch,
        f"fade=t=in:start_frame=0:nb_frames={window['fade_in_frames']}:alpha=1,"
        f"{fade_out}settb=1/{FPS},setpts=N+{start}",
    )
    if motion:
        branch = graph.filter(branch, f"scale=w='{_window_coordinate(window, 2, f'n+{start}')}':"
                              f"h='{_window_coordinate(window, 3, f'n+{start}')}':"
                              "eval=frame,setsar=1")
    return branch


def _card_windows(graph: _Graph, base: str, placement: dict,
                  sources: dict[tuple[str, int], _Source], *,
                  cube: Path | None = None,
                  matches: Mapping[str | tuple[str, int], str] | None = None) -> tuple[str, float]:
    """Overlay each clip before its chrome and account for independent decode time."""
    seconds = 0.0
    for index, window in enumerate(placement.get("windows", [])):
        source = sources[placement["id"], index]
        start, frames = window["start_frame"], window["frames"]
        layer = _window_clip(graph, source, window, owner=f"{placement['id']} "
                             f"({window['source'].get('source_name', source.path.name)})")
        if matches and (placement["id"], index) in matches:
            layer = graph.filter(layer, matches[placement["id"], index])
        if cube is not None:
            layer = graph.filter(layer, rgba_filter(cube))
        if "motion" in window:
            base = graph.filter(f"{base}][{layer}",
                                f"overlay=x='{_window_coordinate(window, 0, 'n-1')}':"
                                f"y='{_window_coordinate(window, 1, 'n-1')}':eval=frame:"
                                "eof_action=pass:repeatlast=0:format=auto:"
                                f"enable='between(n,{start + 1},{start + frames})'")
        else:
            x, y, _, _ = _window_rect(window)
            base = graph.overlay(base, layer, start, start + frames, x=x, y=y)
        seconds += ((start + frames) / FPS if window["source"]["kind"] == "self"
                    else source.info.video_duration_s or source.info.duration_s or frames / FPS)
    return base, seconds


def _assert_picture(info: ffmpeg.MediaInfo, expected: int, frame: Canvas | None = None) -> None:
    """Refuse to hand audio assembly an incorrectly encoded delivery stream."""
    frame = frame or canvas()
    checks = {"frames": (info.video_frames, expected), "width": (info.width, frame.width),
              "height": (info.height, frame.height), "fps": (info.fps, FPS),
              "pixel format": (info.pix_fmt, PIX_FMT),
              "primaries": (info.color_primaries, COLOUR),
              "transfer": (info.color_transfer, COLOUR), "matrix": (info.color_space, COLOUR),
              "codec": (info.video_codec, "h264"), "audio": (info.has_audio, False)}
    failures = [f"{name} is {actual!r}, expected {wanted!r}"
                for name, (actual, wanted) in checks.items() if actual != wanted]
    if failures:
        raise PictureError("Encoded picture failed verification: " + "; ".join(failures))


def render_picture(
    project_dir: Path, out: Path, *, work_dir: Path,
    timeline: ResolvedTimeline | None = None,
    title_layers: list[dict[str, Any]] | None = None,
    caption_layer: dict[str, Any] | None = None,
    transition_layers: list[dict[str, Any]] | None = None,
    smooth_layers: list[dict[str, Any]] | None = None,
    letters_layers: list[dict[str, Any]] | None = None,
    matte_project_dir: Path | None = None,
    frames: tuple[int, int] | None = None,
    grade: GradeSpec | None = None,
    grade_style: dict[str, Any] | None = None,
    moment_style: dict[str, Any] | None = None,
    cutout: str | None = None,
    base_picture: Path | None = None,
) -> PictureResult:
    """Encode silent H.264 once; U15 owns publication and the final stage receipt.

    ``out`` must be a new staging file. Layer stages retain their own receipts;
    this result records picture probes, the encode and HDR conversions. The
    caller can serialize it with ``to_dict()`` and mux its stream by copy.
    Explicit behind graphics restore the base through the cut's person matte.
    Titles retain their stacking behavior and never trigger or use the cutout.
    Restoration is skipped on zoom, insert and transition frames because their
    picture no longer aligns with the cut matte. Front graphics and captions follow it.
    ``matte_project_dir`` keeps the cut matte and still cache in the real project.
    With ``frames``, the same graph is trimmed to that inclusive output window before
    encoding, so a pilot's frames are the full render's pixels.
    """
    with paths.operation():
        root = Path(project_dir).resolve()
        frame = canvas(load_project(root / "project.json").shape)
        cache_root = Path(matte_project_dir or root).resolve()
        timeline = timeline if timeline is not None else resolve_timeline(root).require_valid()
        check_split_voiceover(root, timeline.elements)
        if timeline.camera_3d and base_picture is None:
            camera_captions = captions(root)
            if camera_captions.get("picture"):
                base_picture = root / camera_captions["picture"]
            if caption_layer is None:
                caption_layer = camera_captions["layer"]
        commands: list[tuple[str, ...]] = []
        conversions: list[dict[str, Any]] = []
        warnings = [item.message for item in timeline.findings if item.severity == "warning"]
        backgrounds = [item for item in timeline.elements if item.kind == "background"]
        tall = [item for item in timeline.elements if item.kind == "overlay"
                and item.payload.get("placement") == "tall_behind"]
        moments = any(item.kind == "moment" for item in timeline.elements)
        green_measurements: tuple[dict, ...] = ()
        cut = _source(base_picture or root / "cut/cut.mp4", commands, conversions, owner="cut")
        matte_requested = moments or backgrounds or tall or any(
            item.kind == "graphic" and item.layer == "behind" for item in timeline.elements)
        if matte_requested and (cut.info.variable_frame_rate or cut.info.fps != FPS):
            raise CutMatteError(("The tall box behind your head needs a usable person cutout: "
                                 "use a constant-rate 30 fps cut." if tall else
                                 "Behind graphics, backgrounds and moments need a constant-rate "
                                 "30 fps cut."),
                                timing=True)
        if (cut.info.video_frames != timeline.total_frames or cut.info.width != frame.width
                or cut.info.height != frame.height or cut.info.fps != FPS):
            raise PictureError("Cut picture dimensions or frame count disagree with the timeline.")
        inserts = _insert_inputs(root, timeline, commands, conversions)
        layouts = split_layouts(cache_root, timeline)
        warnings.extend(warning for layout in layouts.values() for warning in layout.warnings)
        letters_by_id = letters.placements(timeline, letters_layers or [])
        ffmpeg_version = ffmpeg.tool_versions().get("ffmpeg")
        stills: list[dict[str, Any]] = []
        prepared: dict[str, tuple[_Source, dict[str, Any] | None]] = {}
        for item in timeline.elements:
            source = inserts.get(item.id)
            if source is None or not source.still:
                continue
            digest = item.payload["media"]["sha256"]
            if digest not in prepared:
                count = len(stills)
                decoded = _prepare_still(source, cache_root, commands, stills,
                                          source_sha256=digest, owner=item,
                                          ffmpeg_version=ffmpeg_version, frames=frames)
                prepared[digest] = (decoded, stills[-1] if len(stills) > count else None)
            elif (entry := prepared[digest][1]) is not None:
                stills.append({**entry, "input": item.id, "reused": True})
            inserts[item.id] = prepared[digest][0]
        zooms = _zoom_inputs(root, timeline, cut, commands, conversions, warnings)
        target = paths.validate_output(out, allowed_existing_suffixes=frozenset())
        work = paths.validate_output_dir(work_dir)
        cube = write_cube(grade, work) if grade is not None and grade.active else None
        target.parent.mkdir(parents=True, exist_ok=True)
        title_layers = graphics(root)["elements"] if title_layers is None else title_layers
        window_sources = _window_sources(root, title_layers, cut, commands, conversions)
        matches, match_records = match_broll(
            root, timeline, cut, inserts, work, prefix=_prefix, still_loop=_still_loop,
            commands=commands, placements=title_layers, window_sources=window_sources,
            window_rect=_window_rect,
            color_match=moment_style["broll"]["color_match"] == "on"
            if moment_style is not None else None)
        graphic_ids = {item.id for item in timeline.elements if item.kind == "graphic"}
        number_ids = {item.id for item in timeline.elements
                      if item.kind == "graphic" and item.payload.get("template") == "big_number"}
        behind_ids = {layer["id"] for layer in title_layers
                      if layer["id"] in graphic_ids and layer["layer"] == "behind"}
        if (behind_ids or backgrounds) and (cut.info.variable_frame_rate or cut.info.fps != FPS):
            raise CutMatteError("Behind graphics and backgrounds need a constant-rate 30 fps cut.",
                                timing=True)
        caption_layer = captions(root)["layer"] if caption_layer is None else caption_layer
        rendered = {placement.get("id") for placement in title_layers}
        missing = [item.id for item in timeline.elements
                   if item.kind in {"graphic", "title", "chapter"} and item.id not in rendered]
        if missing:
            raise PictureError("Missing rendered graphics: " + ", ".join(missing))
        try:
            matte_path = ensure_cut_matte(matte_project_dir or root, cut.path,
                                         cutout=cutout or chroma.project_request(cache_root)) \
                if behind_ids or backgrounds or number_ids or moments or tall else None
        except CutMatteError as exc:
            if tall:
                raise CutMatteError("The tall box behind your head needs a usable person cutout: "
                                    + str(exc)) from exc
            if backgrounds:
                raise CutMatteError("Background media needs a usable person matte: " + str(exc)) \
                    from exc
            raise
        cutout_info = chroma.cutout_metadata(matte_path)
        skipped_items = [item for item in timeline.elements
                         if item.kind in {"zoom", "insert", "transition"}]
        skipped = tuple((item.start_frame, item.end_frame) for item in skipped_items)
        skipped += tuple((item["start_frame"], item["start_frame"] + item["frames"])
                         for item in smooth_layers or [])
        graph = _Graph()
        base = graph.filter(graph.input(cut, owner="cut"),
                            f"trim=end_frame={timeline.total_frames},"
                            f"settb=1/{FPS},setpts=N,setsar=1")
        decode_seconds = (timeline.total_frames + timeline.end_hold_frames) / FPS
        for placement in smooth_layers or []:
            layer = _layer(graph, root, {**placement, "width": WIDTH, "height": HEIGHT},
                           timeline, commands, conversions)
            start = placement["start_frame"]
            base = graph.overlay(base, layer, start, start + placement["frames"])
            decode_seconds += placement["frames"] / FPS
        for item, piece, source, crop_x in zooms:
            first = max(item.start_frame, piece.out_f0)
            last = min(item.end_frame, piece.out_f0 + piece.n_frames)
            from_cut = source is cut
            src_first = first if from_cut else piece.src_f0 + first - piece.out_f0
            layer = _zoom(graph, source, item, first=first, last=last, src_first=src_first,
                          crop_x=crop_x, from_cut=from_cut)
            base = graph.overlay(base, layer, first, last)
            decode_seconds += (src_first + last - first) / FPS
        for item in timeline.elements:
            if item.kind == "insert":
                if item.id in layouts:
                    branch = split_frame(graph, inserts[item.id], item, layouts[item.id],
                                         insert=_insert, cut=cut,
                                         top_filter=matches.get(item.id))
                else:
                    branch = _insert(graph, inserts[item.id], item)
                    if item.id in matches:
                        branch = graph.filter(branch, matches[item.id])
                if item.id in letters_by_id:
                    plate = _layer(graph, root, {**letters_by_id[item.id],
                                   "width": WIDTH, "height": HEIGHT}, timeline,
                                   commands, conversions)
                    branch = letters.composite(graph, branch, plate, item)
                base = graph.overlay(base, branch, item.start_frame, item.end_frame)
                decode_seconds += item.end_frame / FPS if item.id in layouts else 0
                decode_seconds += inserts[item.id].info.video_duration_s or 0
        transitions = [item for item in timeline.elements if item.kind == "transition"]
        placements = transition_layers or []
        if ({placement["id"] for placement in placements} != {item.id for item in transitions}
                or len(placements) != len(transitions)):
            raise PictureError("Rendered transitions must match the resolved transition elements.")
        by_id = {item.id: item for item in transitions}
        for placement in placements:
            item = by_id[placement["id"]]
            if (placement["start_frame"] != item.start_frame
                    or placement["frames"] != item.end_frame - item.start_frame):
                raise PictureError(f"{item.id}: rendered transition window differs from timeline.")
            layer = _layer(graph, root, {**placement, "width": WIDTH, "height": HEIGHT},
                           timeline, commands, conversions)
            base = graph.overlay(base, layer, item.start_frame, item.end_frame)
            decode_seconds += placement["frames"] / FPS
        compares = [item for item in timeline.elements if item.kind == "grade_compare"]
        compare_records: tuple[dict[str, Any], ...] = ()
        if compares and cube is None:
            raise PictureError("Grade compare needs an active grade.")
        if cube is not None:
            if compares:
                from yap_studio.project.status import ProjectContext
                from yap_studio.stages.assemble_grade import grade_picture

                style = (grade_style if grade_style is not None
                         else ProjectContext(root).render_style().data)
                base, compare_records = grade_picture(graph, base, cube, compares, style, work)
            else:
                base = graph.filter(base, graph_filter(cube))
        moment_records: list[dict[str, Any]] = []
        base = overlay_moments(graph, base, cut.path, matte_path, timeline, work, cache_root,
                               commands, conversions, frames=frames, style=moment_style,
                               root=root, records=moment_records, cube=cube)
        ordered = sorted(title_layers, key=lambda item: (
            item.get("kind") == "title" and item.get("id") == TEXT_HOOK_ID,
            item["layer"] != "behind", item.get("kind") == "title",
            item["start_frame"], item.get("id", ""),
        ))
        behind = [placement for placement in ordered if placement.get("id") in behind_ids]
        if matte_path is not None and (behind or backgrounds or tall):
            tall_keep = [item for item in tall if tall_box.chosen_framing(item.payload) != "shrink"]
            tall_shrink = [item for item in tall
                           if tall_box.chosen_framing(item.payload) == "shrink"]
            # Split the base before graphics; no text pixels may enter the person branch.
            kinds = [kind for kind, used in (("graphic", behind), ("background", backgrounds),
                                             ("tall", tall_keep), ("tall_shrink", tall_shrink))
                     if used]
            graph.filters.append(f"[{base}]split={len(kinds) + 1}[behind_base]"
                                 + "".join(f"[person_{kind}]" for kind in kinds))
            base = "behind_base"
            matte_source = _source(matte_path, commands, conversions, owner="cut matte",
                                   convert_hdr=False)
            raw = graph.filter(graph.input(matte_source, owner="cut matte"),
                               f"trim=end_frame={timeline.total_frames},"
                               f"settb=1/{FPS},setpts=N,setsar=1,format=gray")
            masks = {kinds[0]: raw}
            if len(kinds) > 1:
                graph.filters.append(f"[{raw}]split={len(kinds)}"
                                     + "".join(f"[matte_{kind}]" for kind in kinds))
                masks = {kind: f"matte_{kind}" for kind in kinds}
            if (backgrounds or tall_shrink) and cutout_info is None:
                # Keep the soft outer rim while firming translucent hands inside the silhouette.
                # A screen key already has a firm interior.
                for kind in ("background", "tall_shrink"):
                    if kind in masks:
                        expression = background_matte_filter(WIDTH, HEIGHT)
                        if kind == "tall_shrink":
                            expression = expression.replace("[edge_", "[tall_shrink_edge_")
                        masks[kind] = graph.filter(masks[kind], expression)
            person = _person_edges(graph, cut.path, matte_path, work, behind, backgrounds,
                                   commands, conversions, frames=frames, cube=cube,
                                   compares=compares,
                                   **({"tall": tall_keep} if tall_keep else {}),
                                   **({"tall_shrink": tall_shrink} if tall_shrink else {}))
            people = {kind: graph.filter(f"{person[kind]}][{masks[kind]}",
                                         "alphamerge=ts_sync_mode=nearest")
                      for kind in kinds}
            people = chroma.despill_people(graph, people, cutout_info)
            if tall:
                rooms = green_screen._split(graph, base, len(tall) + 1)
                base = rooms[0]
                cutouts = {item.id: branch
                    for kind, group in (("tall", tall_keep), ("tall_shrink", tall_shrink)) if group
                    for item, branch in zip(group, green_screen._split(graph, people[kind],
                                                                      len(group)), strict=True)}
                for item, room in zip(tall, rooms[1:], strict=True):
                    page = _overlay(graph, inserts[item.id], item)
                    if item.id in matches:
                        page = graph.filter(page, matches[item.id])
                    if cube is not None:
                        page = graph.filter(page, rgba_filter(cube))
                    layer = _tall_composite(graph, room, cutouts[item.id], page, item)
                    base = graph.overlay(base, layer, item.start_frame, item.end_frame)
            legacy_backgrounds, look_layers, green_measurements = green_screen.prepare(
                graph, backgrounds, inserts, people, cut, matte_source, commands,
                layer_filter=rgba_filter(cube) if cube is not None else None)
            for item in legacy_backgrounds:
                # Legacy backgrounds keep their original framing.
                layer = _overlay(graph, inserts[item.id], item, fade=False, motion=False)
                if cube is not None:
                    layer = graph.filter(layer, rgba_filter(cube))
                base = graph.overlay(base, layer, item.start_frame, item.end_frame)
                decode_seconds += inserts[item.id].info.video_duration_s or 0
            for item, layer, _ in look_layers:
                base = graph.overlay(base, layer, item.start_frame, item.end_frame)
                decode_seconds += inserts[item.id].info.video_duration_s or 0
                decode_seconds += green_screen.decode_seconds(item)
            for placement in behind:
                base, window_seconds = _card_windows(graph, base, placement, window_sources,
                                                     cube=cube, matches=matches)
                decode_seconds += window_seconds
                layer = _layer(graph, root, placement, timeline, commands, conversions)
                base = graph.overlay(base, layer, placement["start_frame"],
                                     placement["start_frame"] + placement["frames"],
                                     x=placement["x"], y=placement["y"])
            excluded = _frames(skipped)
            background_frames = _frames((item.start_frame, item.end_frame)
                                        for item in backgrounds)
            graphic_frames = "0"
            if behind:
                people["graphic"], graphic_frames = number_redraw(
                    graph, people["graphic"], behind, timeline, width=WIDTH, height=HEIGHT, fps=FPS,
                )
            # Background frames take the background person, so the two enables never overlap.
            for kind, visible, hidden in (
                ("background", _frames((item.start_frame, item.end_frame)
                                       for item in legacy_backgrounds), excluded),
                ("graphic", graphic_frames, "+".join(filter(None, (background_frames, excluded)))),
            ):
                if kind in people:
                    enable = f"({visible})" + (f"*not({hidden})" if hidden else "")
                    base = graph.overlay(base, people[kind], 0, timeline.total_frames,
                                         enable=enable)
            for item, _, person in look_layers:
                base = graph.overlay(base, person, item.start_frame, item.end_frame)
            ordered = [placement for placement in ordered if placement.get("id") not in behind_ids]
            skipped_kinds = {skipped.kind for skipped in skipped_items for item in behind
                             if max(skipped.start_frame, item["start_frame"]) < min(
                                 skipped.end_frame, item["start_frame"] + item["frames"]
                             )}
            if {"zoom", "insert"} & skipped_kinds:
                warnings.append("Person restoration for behind graphics is skipped during "
                                "zoom and insert windows.")
            if "transition" in skipped_kinds:
                warnings.append("Person restoration for behind graphics is skipped during "
                                "transition windows.")
            if any(max(window["start_frame"], item["start_frame"]) < min(
                    window["start_frame"] + window["frames"], item["start_frame"] + item["frames"])
                   for window in smooth_layers or [] for item in behind):
                warnings.append("Person restoration for behind graphics is skipped during "
                                "smooth join windows.")
        for item in timeline.elements:
            if item.kind == "overlay" and item.payload.get("placement") != "tall_behind":
                x, y, _, _ = item.payload["box_px"]
                layer = _overlay(graph, inserts[item.id], item)
                if item.id in matches:
                    layer = graph.filter(layer, matches[item.id])
                if cube is not None:
                    layer = graph.filter(layer, rgba_filter(cube))
                base = graph.overlay(base, layer,
                                     item.start_frame, item.end_frame, x=x, y=y)
                decode_seconds += inserts[item.id].info.video_duration_s or 0
        for placement in [*ordered, caption_layer]:
            base, window_seconds = _card_windows(graph, base, placement, window_sources,
                                                 cube=cube, matches=matches)
            decode_seconds += window_seconds
            layer = _layer(graph, root, placement, timeline, commands, conversions)
            base = graph.overlay(base, layer, placement["start_frame"],
                                 placement["start_frame"] + placement["frames"],
                                 x=placement["x"], y=placement["y"])
        total = timeline.total_frames + timeline.end_hold_frames
        base = graph.filter(base, f"trim=end_frame={timeline.total_frames},"
                            f"tpad=stop_mode=clone:stop={timeline.end_hold_frames},"
                            f"trim=end_frame={total},settb=1/{FPS},setpts=N,format={PIX_FMT},"
                            f"setparams=color_primaries={COLOUR}:color_trc={COLOUR}:"
                            f"colorspace={COLOUR}")
        if frames is not None:
            base = graph.filter(base, f"trim=start_frame={frames[0]}:"
                                f"end_frame={frames[1] + 1},setpts=N")
        script = paths.validate_output(work / "picture.ffgraph",
                                       allowed_existing_suffixes=frozenset())
        script.write_text(";\n".join(graph.filters) + "\n", encoding="utf-8")
        command = ["ffmpeg", "-nostdin", "-v", "error", "-filter_complex_threads", "1",
                   *graph.inputs, "-filter_complex", ";".join(graph.filters), "-map", f"[{base}]",
                   "-an", "-frames:v", str(frames[1] - frames[0] + 1 if frames else total),
                   "-r", str(FPS), "-c:v", CODEC,
                   "-crf", str(CRF), "-preset", PRESET, "-pix_fmt", PIX_FMT,
                   "-color_primaries", COLOUR, "-color_trc", COLOUR, "-colorspace", COLOUR,
                   "-movflags", "+faststart", str(target)]
        try:
            result = ffmpeg.run(command, timeout_s=_timeout(decode_seconds))
        except ProcessingError as exc:
            indexes = dict.fromkeys(int(match) for match in re.findall(
                r"(?:\[(?:vist|in)#|\binput\s+#)(\d+)", exc.full_stderr, re.IGNORECASE))
            owners = [graph.owners[index] for index in indexes if index in graph.owners]
            raise PictureProcessingError(exc, owners, cache_root, frames=frames) from exc
        commands.append(tuple(result.args))
        _assert_picture(_probe(target, commands),
                        frames[1] - frames[0] + 1 if frames else total, frame)
        measurements = measure_behind_layers(
            root, matte_path, cut.info, [item for item in behind if item["id"] not in number_ids],
            skipped_windows=skipped,
        ) if matte_path is not None and behind else ()
        if number_ids:
            measurements += measure_number_layers(root, matte_path, cut.info, title_layers,
                                                   timeline, skipped)
        return PictureResult(total, timeline.end_hold_frames, tuple(commands),
                             tuple(dict.fromkeys(warnings)), tuple(conversions), measurements,
                             frames, tuple(stills), describe(grade, cube) if cube else None,
                             compare_records,
                             splits=tuple({"id": key, **value.to_dict()}
                                          for key, value in layouts.items()),
                             broll_match=tuple(match_records),
                             cutout=cutout_info,
                             green_screen=green_measurements,
                             moments=tuple(moment_records))
