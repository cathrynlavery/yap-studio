"""Deterministic BT.709 colour looks shared by assembly and verification."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from yap_studio import ffmpeg, receipt
from yap_studio.errors import ValidationError

GRADE_VERSION = 2
EYEDROPPER_DEFAULT_STRENGTH = 0.5
EYEDROPPER_MIN_CHROMA = 0.02
EYEDROPPER_PULL = 0.45
EYEDROPPER_CHROMA_CAP = 0.12
EYEDROPPER_LIGHTNESS = (0.08, 0.4)
EYEDROPPER_HIGHLIGHTS = (0.85, 1.0)
EYEDROPPER_HIGHLIGHT_REDUCTION = 0.6
EYEDROPPER_SKIN_PROTECTION = 0.92
COLOR_PATTERN = re.compile(r"#[0-9A-Fa-f]{6}\Z")
GAMMA = 2.4
TONE_X = np.array([0, 0.25, 0.55, 0.8, 1.0])
TONE_Y = np.array([0.04, 0.215, 0.55, 0.84, 0.97])
SKIN_HUE = 55.0
SKIN_HUE_INNER = 20.0
SKIN_HUE_OUTER = 35.0
SKIN_CHROMA_LOW = (0.015, 0.035)
SKIN_CHROMA_HIGH = (0.17, 0.22)
SKIN_LIGHTNESS = (0.2, 0.35)
CHROMA_REDUCTION = 0.15
SKIN_CHROMA_PROTECTION = 0.8
SKIN_TONE_PROTECTION = 0.92
SHADOW_RANGE = (0.15, 0.6)
HIGHLIGHT_RANGE = (0.55, 0.95)
SHADOW_TINT = np.array([-0.020, -0.022])
HIGHLIGHT_TINT = np.array([0.010, 0.026])
# Oklab matrices for linear sRGB, which shares BT.709 primaries.
RGB_TO_LMS = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
LMS_TO_LAB = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])
LAB_TO_LMS = np.linalg.inv(LMS_TO_LAB)
LMS_TO_RGB = np.linalg.inv(RGB_TO_LMS)


class GradeError(ValidationError):
    """Refuse a colour or sample with a concrete correction."""

    def __init__(self, detail: str, fix: str = "Pick a more colourful point or colour.") -> None:
        super().__init__("grade", detail)
        self.suggested_action = fix


def color_info(color: str) -> dict[str, Any]:
    """Describe a hex colour without refusing a valid neutral sample."""
    if not isinstance(color, str) or COLOR_PATTERN.fullmatch(color) is None:
        raise GradeError("Use a colour like #1E6BD6.", "Use a hex colour like #1E6BD6.")
    rgb = np.array([int(color[i:i+2], 16) for i in (1, 3, 5)]) / 255
    target = np.cbrt(rgb ** GAMMA @ RGB_TO_LMS.T) @ LMS_TO_LAB.T
    chroma = float(np.hypot(target[1], target[2]))
    return {"color": color.upper(), "chroma": chroma,
            "colorful": chroma >= EYEDROPPER_MIN_CHROMA}


@dataclass(frozen=True)
class GradeSpec:
    """One look selection and its origin, independent of a render window."""

    look: Literal["none", "film", "eyedropper"] = "none"
    strength: float = 1.0
    source: Literal["style", "project"] = "style"
    color: str | None = None

    def __post_init__(self) -> None:
        if self.look not in ("none", "film", "eyedropper"):
            raise ValidationError("grade.look", "choose none, film or eyedropper")
        if (type(self.strength) not in (int, float) or not math.isfinite(self.strength)
                or not 0 <= self.strength <= 1):
            raise ValidationError("grade.strength", "use a finite number from 0 to 1")
        if self.source not in ("style", "project"):
            raise ValidationError("grade.source", "choose style or project")
        if self.color is not None:
            info = color_info(self.color)
            object.__setattr__(self, "color", info["color"])
        if self.look == "eyedropper":
            if self.color is None:
                raise GradeError("Eyedropper needs a colour.")
            if not color_info(self.color)["colorful"]:
                raise GradeError("This colour is too grey. Pick a more colourful point or colour.")

    @property
    def active(self) -> bool:
        """Return whether picture pixels need grading."""
        return self.look != "none" and self.strength > 0

    def to_dict(self) -> dict[str, Any]:
        """Return the resolved selection without render evidence."""
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def from_dict(cls, data: dict) -> GradeSpec:
        """Validate a stored selection and any declared math version."""
        if not isinstance(data, dict):
            raise ValidationError("grade", "use an object with look and strength")
        if set(data) - {"look", "strength", "source", "version", "color"}:
            raise ValidationError("grade", "remove unknown fields")
        if "version" in data and (type(data["version"]) is not int
                                  or data["version"] != GRADE_VERSION):
            raise ValidationError("grade.version", "re-render with yap-studio assemble")
        values = {key: value for key, value in data.items() if key != "version"}
        if values.get("look") == "eyedropper":
            values.setdefault("strength", EYEDROPPER_DEFAULT_STRENGTH)
        return cls(**values)


def resolve_grade(style_data: dict, project_grade: dict | None, *,
                  cut: Path | None = None) -> GradeSpec:
    """Use an explicit project override, including an empty or inactive one."""
    data = style_data.get("grade", {}) if project_grade is None else project_grade
    if not isinstance(data, dict):
        raise GradeError("Use a grade object.", "Choose a grade look and strength in your style.")
    data = dict(data)
    if project_grade is None:
        data.pop("slider", None)
    sample = data.pop("sample", None)
    if sample is not None:
        if not isinstance(sample, dict) or set(sample) != {"time_s", "x", "y"}:
            raise GradeError("Use a sample with time_s, x and y.",
                             "Use --time T --x X --y Y to pick a point in the cut.")
        if data.get("look") != "eyedropper" or data.get("color") is not None:
            raise GradeError("Use exactly one eyedropper colour or sample.")
        if cut is None or not cut.is_file():
            raise GradeError("The cut picture is missing.", "Run yap-studio cut DIR first.")
        data["color"] = sample_color(cut, **sample)
    spec = GradeSpec.from_dict(data)
    return GradeSpec(spec.look, spec.strength, "style" if project_grade is None else "project",
                     spec.color)


def sample_color(video: Path, *, time_s: float, x: float, y: float,
                 radius_px: int = 6) -> str:
    """Average one clamped RGB patch from an exact frame of the 30 fps cut."""
    for name, value in (("time_s", time_s), ("x", x), ("y", y)):
        if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                or (name != "time_s" and value > 1)):
            raise GradeError(f"Invalid sample {name}.",
                             "Use a time inside the cut and x, y from 0 to 1.")
    if type(radius_px) is not int or radius_px < 0:
        raise GradeError("Invalid sample radius.", "Use a non-negative integer radius.")
    if not Path(video).is_file():
        raise GradeError("The cut picture is missing.", "Run yap-studio cut DIR first.")
    info = ffmpeg.probe(video)
    frame = round(time_s * 30)
    if info.fps != 30 or not info.video_frames or frame >= info.video_frames:
        raise GradeError("Sample time is outside the 30 fps cut.",
                         "Pick a time inside cut/cut.mp4; run yap-studio cut DIR first if needed.")
    cx = min(info.width - 1, round(x * info.width))
    cy = min(info.height - 1, round(y * info.height))
    left, top = max(0, cx - radius_px), max(0, cy - radius_px)
    right, bottom = min(info.width, cx + radius_px + 1), min(info.height, cy + radius_px + 1)
    data = ffmpeg.run([
        "ffmpeg", "-v", "error", "-i", str(video), "-vf",
        f"trim=start_frame={frame}:end_frame={frame + 1},format=rgb24,"
        f"crop={right-left}:{bottom-top}:{left}:{top}:exact=1",
        "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ], timeout_s=60, binary=True).stdout
    if len(data) != (right-left) * (bottom-top) * 3:
        raise GradeError("Cannot read the sample frame.", "Run yap-studio cut DIR first.")
    mean = np.rint(np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).mean(axis=0)).astype(int)
    return "#" + "".join(f"{value:02X}" for value in mean)


def _smoothstep(low: float, high: float, values: np.ndarray) -> np.ndarray:
    t = np.clip((values - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


def _tone(values: np.ndarray) -> np.ndarray:
    """Evaluate a Fritsch-Carlson monotone cubic Hermite curve."""
    widths = np.diff(TONE_X)
    slopes = np.diff(TONE_Y) / widths
    tangents = np.concatenate(([slopes[0]], (slopes[:-1] + slopes[1:]) / 2, [slopes[-1]]))
    for i, slope in enumerate(slopes):
        if slope == 0:
            tangents[i:i + 2] = 0
            continue
        alpha, beta = tangents[i:i + 2] / slope
        radius = math.hypot(alpha, beta)
        if radius > 3:
            tangents[i:i + 2] *= 3 / radius
    index = np.clip(np.searchsorted(TONE_X, values, side="right") - 1, 0, len(widths) - 1)
    t = (values - TONE_X[index]) / widths[index]
    return ((2*t**3 - 3*t**2 + 1) * TONE_Y[index]
            + (t**3 - 2*t**2 + t) * widths[index] * tangents[index]
            + (-2*t**3 + 3*t**2) * TONE_Y[index + 1]
            + (t**3 - t**2) * widths[index] * tangents[index + 1])


def apply_rgb(rgb: np.ndarray, spec: GradeSpec) -> np.ndarray:
    """Grade non-linear BT.709 float RGB in 0..1, preserving leading dimensions."""
    values = np.clip(np.asarray(rgb, dtype=np.float64), 0, 1)
    if values.ndim == 0 or values.shape[-1] != 3:
        raise ValidationError("rgb", "use an array with three colour channels")
    if not spec.active:
        return values
    lab = np.cbrt(values ** GAMMA @ RGB_TO_LMS.T) @ LMS_TO_LAB.T
    light, a, b = np.moveaxis(lab, -1, 0)
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a))
    distance = np.abs((hue - SKIN_HUE + 180) % 360 - 180)
    skin = ((1 - _smoothstep(SKIN_HUE_INNER, SKIN_HUE_OUTER, distance))
            * _smoothstep(*SKIN_CHROMA_LOW, chroma)
            * (1 - _smoothstep(*SKIN_CHROMA_HIGH, chroma))
            * _smoothstep(*SKIN_LIGHTNESS, light))
    graded = lab.copy()
    if spec.look == "eyedropper":
        target_rgb = np.array([int(spec.color[i:i+2], 16) for i in (1, 3, 5)]) / 255
        target = np.cbrt(target_rgb ** GAMMA @ RGB_TO_LMS.T) @ LMS_TO_LAB.T
        target_chroma = np.hypot(target[1], target[2])
        direction = target[1:] / target_chroma
        pull = EYEDROPPER_PULL * min(target_chroma, EYEDROPPER_CHROMA_CAP)
        weight = (_smoothstep(*EYEDROPPER_LIGHTNESS, light)
                  * (1 - EYEDROPPER_HIGHLIGHT_REDUCTION
                     * _smoothstep(*EYEDROPPER_HIGHLIGHTS, light)))
        graded[..., 1:] += (pull * weight * (1 - EYEDROPPER_SKIN_PROTECTION * skin))[..., None] \
            * direction
    else:
        graded[..., 0] = _tone(light)
        graded[..., 1:] *= (1 - CHROMA_REDUCTION
                           * (1 - SKIN_CHROMA_PROTECTION * skin))[..., None]
        tint = ((1 - _smoothstep(*SHADOW_RANGE, light))[..., None] * SHADOW_TINT
                + _smoothstep(*HIGHLIGHT_RANGE, light)[..., None] * HIGHLIGHT_TINT)
        graded[..., 1:] += tint * (1 - SKIN_TONE_PROTECTION * skin)[..., None]
    blended = lab + spec.strength * (graded - lab)
    linear = (blended @ LAB_TO_LMS.T) ** 3 @ LMS_TO_RGB.T
    return np.clip(linear, 0, 1) ** (1 / GAMMA)


def cube_text(spec: GradeSpec, size: int = 33) -> str:
    """Serialize a red-fastest RGB grid with stable six-decimal values."""
    if type(size) is not int or not 2 <= size <= 256:
        raise ValidationError("cube.size", "use an integer from 2 to 256")
    b, g, r = np.meshgrid(*[np.linspace(0, 1, size)] * 3, indexing="ij")
    values = apply_rgb(np.stack((r, g, b), axis=-1).reshape(-1, 3), spec)
    return f"LUT_3D_SIZE {size}\n" + "".join(
        f"{r:.6f} {g:.6f} {b:.6f}\n" for r, g, b in values)


def write_cube(spec: GradeSpec, folder: Path) -> Path:
    """Write the versioned cube into a caller-owned work directory."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cube = folder / f"grade-v{GRADE_VERSION}.cube"
    cube.write_text(cube_text(spec), encoding="ascii")
    return cube


def _option_path(path: Path) -> str:
    """Escape both the filter option parser and the enclosing graph parser."""
    value = str(path)
    value = "".join("\\" + c if c in "\\':" else c for c in value)
    return "".join("\\" + c if c in "\\'[],; " else c for c in value)


def rgba_filter(cube: Path) -> str:
    """Grade an RGB overlay without changing alpha."""
    return f"lut3d=file={_option_path(cube)}:interp=tetrahedral"


def _graph_filter(cube: Path, *, alpha: bool) -> str:
    """Share the BT.709 conversion around the cube for pictures and alpha layers."""
    rgb, yuv = ("gbrap16le", "yuva444p") if alpha else ("gbrp16le", "yuv444p")
    return (f"scale=in_color_matrix=bt709:in_range=tv,format={rgb},"
            f"{rgba_filter(cube)},scale=out_color_matrix=bt709:out_range=tv,format={yuv}")


def graph_filter(cube: Path) -> str:
    """Convert a limited-range BT.709 picture to RGB and back around the cube."""
    return _graph_filter(cube, alpha=False)


def graph_alpha_filter(cube: Path) -> str:
    """Grade limited-range BT.709 layer colours without changing their alpha."""
    return _graph_filter(cube, alpha=True)


def describe(spec: GradeSpec, cube: Path) -> dict:
    """Return the selection, math version and exact cube identity for receipts."""
    return {**spec.to_dict(), "version": GRADE_VERSION, "cube_sha256": receipt.sha256_file(cube)}


COMPARE_POINTS = ((0.0, 0.0), (0.30, 0.78), (0.52, 0.30),
                  (0.68, 0.50), (0.86, 0.50), (1.0, 0.0))
HANDLE_WIDTH = 96
HANDLE_HEIGHT = 1920
HANDLE_KNOB_Y = 0.66
HANDLE_FADE_FRAMES = 6


def compare_position(local_frame: int, frames: int) -> float:
    """Return the divider share, with fully graded first and last frames."""
    if frames < 2:
        raise GradeError("A compare needs at least two frames.", "Use a 2 to 8 second window.")
    share = min(1.0, max(0.0, local_frame / (frames - 1)))
    for (start, left), (end, right) in zip(COMPARE_POINTS, COMPARE_POINTS[1:], strict=False):
        if share <= end:
            eased = (1 - math.cos(math.pi * (share - start) / (end - start))) / 2
            return left + (right - left) * eased
    return 0.0


def compare_x_expr(start_frame: int, frames: int, width: int) -> str:
    """Build the same cosine legs in ffmpeg's absolute output frame clock."""
    if frames < 2:
        raise GradeError("A compare needs at least two frames.", "Use a 2 to 8 second window.")
    expression = "0"
    legs = list(zip(COMPARE_POINTS, COMPARE_POINTS[1:], strict=False))
    for (start, left), (end, right) in reversed(legs):
        progress = f"((n-{start_frame})/{frames-1}-{start:g})/{end-start:.12g}"
        leg = f"({left:g}+({right-left:.12g})*(1-cos(PI*({progress})))/2)"
        expression = f"if(lt(n,{start_frame + end * (frames-1):.12g}),{leg},{expression})"
    return f"if(lt(n,{start_frame}),0,round({width}*({expression})))"


def handle_rgba(style_data: dict) -> tuple[int, int, bytes]:
    """Draw an antialiased full-height divider and round knob with distance fields."""
    text = style_data.get("captions", {}).get("text_color", "#FFFFFF")
    ink = style_data.get("graphics", {}).get("ink_color", "#222222")
    colours = [np.array([int(value[i:i+2], 16) for i in (1, 3, 5)], dtype=float)
               for value in (text, ink)]
    yy, xx = np.mgrid[:HANDLE_HEIGHT, :HANDLE_WIDTH]
    x, y = xx + 0.5 - HANDLE_WIDTH / 2, yy + 0.5 - HANDLE_HEIGHT * HANDLE_KNOB_Y
    radius = np.hypot(x, y)
    alpha = np.maximum(0.35 * np.exp(-(x / 5) ** 2),
                       0.35 * np.exp(-(np.maximum(radius - 32, 0) / 4) ** 2))
    alpha[alpha < 1 / 255] = 0
    premultiplied = np.zeros((*alpha.shape, 3))

    def paint(coverage: np.ndarray, colour: np.ndarray) -> None:
        nonlocal alpha, premultiplied
        coverage = np.clip(coverage, 0, 1)
        premultiplied = colour * coverage[..., None] + premultiplied * (1-coverage[..., None])
        alpha = coverage + alpha * (1-coverage)

    paint(np.maximum(np.clip(2.5 - np.abs(x), 0, 1), np.clip(32.5-radius, 0, 1)), colours[0])
    paint(np.clip(1.5 - np.abs(radius - 29.5), 0, 1), colours[1])
    for side in (-1, 1):
        for tip_y in (-6, 6):
            ax, ay, bx, by = side * 14, 0, side * 8, tip_y
            t = np.clip(((x-ax)*(bx-ax) + (y-ay)*(by-ay)) / 72, 0, 1)
            distance = np.hypot(x-ax-t*(bx-ax), y-ay-t*(by-ay))
            paint(np.clip(1.5-distance, 0, 1), colours[1])
    rgb = np.divide(premultiplied, alpha[..., None], out=np.zeros_like(premultiplied),
                    where=alpha[..., None] > 0)
    pixels = np.dstack((rgb, alpha * 255))
    return HANDLE_WIDTH, HANDLE_HEIGHT, np.rint(pixels).astype(np.uint8).tobytes()


def _area_axis(pixels: np.ndarray, length: int, axis: int) -> np.ndarray:
    """Average fractional pixel boxes without a dense resampling matrix."""
    source = np.moveaxis(pixels, axis, 0)
    if len(source) == length:
        return pixels
    edges = np.linspace(0, len(source), length + 1)
    indices = np.floor(edges).astype(int)
    fractions = (edges - indices).reshape(-1, *([1] * (source.ndim-1)))
    integral = np.concatenate((np.zeros_like(source[:1]), source.cumsum(axis=0)))
    integral = integral[indices] + fractions * source[np.minimum(indices, len(source)-1)]
    return np.moveaxis(np.diff(integral, axis=0) / (len(source)/length), 0, axis)


def describe_compare(style_data: dict) -> dict:
    """Bind comparison freshness to its motion, fades and actual handle pixels."""
    return {"positions": COMPARE_POINTS, "fade_frames": HANDLE_FADE_FRAMES,
            "handle_sha256": hashlib.sha256(handle_rgba(style_data)[2]).hexdigest()}


def compare_still(original: np.ndarray, graded: np.ndarray, style_data: dict) -> np.ndarray:
    """Show a half-and-half wipe with the handle at its output-frame height."""
    height, width, _ = original.shape
    result = graded.copy()
    result[:, :round(width/2)] = original[:, :round(width/2)]
    strip_width, strip_height, data = handle_rgba(style_data)
    scale = min(width/1080, height/1920)
    target_width = max(1, round(strip_width*scale))
    target_height = max(1, round(strip_height*scale))
    handle = np.frombuffer(data, dtype=np.uint8).reshape(strip_height, strip_width, 4) / 255
    handle[..., :3] *= handle[..., 3:]
    handle = _area_axis(_area_axis(handle, target_height, 0), target_width, 1)
    top = round((height-target_height) * HANDLE_KNOB_Y)
    handle = np.pad(handle, ((top, height-target_height-top), (0, 0), (0, 0)), mode="edge")
    left = (width-target_width)//2
    region = result[:, left:left+target_width]
    region[:] = np.rint(handle[..., :3]*255 + region*(1-handle[..., 3:])).astype(np.uint8)
    return result
