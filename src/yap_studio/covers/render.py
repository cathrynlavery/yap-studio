"""Prepare cover pixels and render one checked still through HyperFrames."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from yap_studio import ffmpeg, paths
from yap_studio.covers.layout import (
    check_geometry,
    plan_blocks,
)
from yap_studio.errors import ValidationError
from yap_studio.matte import chroma
from yap_studio.matte.make import make_matte
from yap_studio.matte.model import load_matte
from yap_studio.render import fonts, hyperframes
from yap_studio.render.text_metrics import text_width_em
from yap_studio.sfx import bleep
from yap_studio.style import ResolvedStyle

_TEMPLATE = Path(__file__).parents[1] / "templates/cover/template.html"


@dataclass(frozen=True)
class CoverAssets:
    """Local, cover-sized picture assets."""

    frame: Path
    cutout: Path | None = None
    blur: Path | None = None
    dark: Path | None = None
    matte: Path | None = None
    inset: Path | None = None
    face_box: tuple[float, float, float, float] | None = None
    shape: str = "vertical"


@dataclass(frozen=True)
class RenderedCover:
    """Checked cover and renderer identity."""

    png: Path
    checks: dict
    renderer_version: str


def _rgba(path: Path, size: tuple[int, int] = (1080, 1920)) -> np.ndarray:
    raw = ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
                      "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1"],
                     timeout_s=90, binary=True).stdout
    width, height = size
    if len(raw) != width * height * 4:
        raise ValidationError("image", f"expected a {width}x{height} cover image")
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)


def _frame_png(clip: Path, index: int, out: Path) -> None:
    ffmpeg.run(["ffmpeg", "-y", "-v", "error", "-i", str(clip),
                "-vf", f"select=eq(n\\,{index}),scale=in_color_matrix=bt709:in_range=tv:"
                       "out_range=pc:flags=accurate_rnd+full_chroma_int,format=rgba",
                "-frames:v", "1", "-c:v", "png", "-pix_fmt", "rgba", out.as_posix()],
               timeout_s=120)


def _default_matte(frame: Path, work: Path) -> Path | None:
    clip = work / "one-frame.mov"
    ffmpeg.run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", "30",
                "-i", str(frame), "-frames:v", "1", "-c:v", "rawvideo", "-pix_fmt", "bgra",
                str(clip)], timeout_s=120)
    data = make_matte(clip, work / "matte", notice=None)
    if data.status != "measured" or data.matte is None:
        return None
    result = work / "matte.png"
    ffmpeg.run(["ffmpeg", "-y", "-v", "error", "-i", str(work / "matte" /
                data.matte["file"]), "-frames:v", "1", "-pix_fmt", "gray", str(result)],
               timeout_s=120)
    return result


def head_box_from_matte(gray: np.ndarray) -> tuple[float, float, float, float] | None:
    """Estimate the head from the upper edge of a person matte."""
    def runs(row: np.ndarray) -> list[int]:
        columns = np.flatnonzero(row)
        if not len(columns):
            return []
        groups = np.split(columns, np.flatnonzero(np.diff(columns) > 4) + 1)
        return [int(group[-1] - group[0] + 1) for group in groups]

    person = gray > 127
    if np.count_nonzero(person) < 1000:
        return None
    top = None
    for row in np.flatnonzero(np.count_nonzero(person, axis=1) >= 20):
        if np.max(np.convolve(person[row].astype(np.uint8), np.ones(20, dtype=np.uint8),
                              mode="valid")) == 20:
            top = int(row)
            break
    if top is None:
        return None
    band = person[top:min(top + 240, gray.shape[0])]
    if any(sum(width >= 12 for width in runs(row)) != 1 for row in band):
        return None
    _, columns = np.nonzero(band)
    if len(columns) < 1000:
        return None
    left, right = np.percentile(columns, (2, 98))
    x = max(0, int(left))
    width = min(gray.shape[1] - x, int(right) + 1 - x)
    if not 80 <= width <= gray.shape[1] * 0.9:
        return None
    shoulder_start = top + round(width * 1.2)
    shoulder_end = min(gray.shape[0], top + round(width * 2.5))
    shoulder_width = max(width + 1, min(width * 1.4, gray.shape[1] * 0.95))
    if not any(max(runs(row), default=0) >= shoulder_width
               for row in person[shoulder_start:shoulder_end]):
        return None
    height = min(gray.shape[0] - top, round(width * 1.3))
    return float(x), float(top), float(width), float(height)


def prepare_assets(
    cut: Path, frame: int, work: Path, look: str, *,
    matte_provider: Callable[[Path, Path], Path | None] | None = None,
    inset: Path | None = None,
    face_box: tuple[float, float, float, float] | None = None,
    shape: str = "vertical",
) -> CoverAssets:
    """Extract the overlay-free frame, lossless matte, and look backgrounds."""
    from yap_studio.covers.pick import extract_frame

    paths.validate_output_dir(work)
    if shape == "horizontal":
        from yap_studio.covers.wide import thumbnail_face
        from yap_studio.faces.model import Box

        picture = extract_frame(cut, frame, work / "frame.png", shape=shape)
        if face_box is None:
            raise ValidationError("face", "wide thumbnail needs a measured face; "
                                  "choose another frame")
        info = ffmpeg.probe(cut)
        face = thumbnail_face(Box(*face_box), info.width, info.height)
        inset_image = None
        if inset is not None:
            inset_image = work / "inset.png"
            ffmpeg.run(["ffmpeg", "-y", "-v", "error", "-i", str(inset), "-vf",
                        "scale=600:600:force_original_aspect_ratio=increase,crop=600:600",
                        "-frames:v", "1", str(inset_image)], timeout_s=120)
        return CoverAssets(picture, inset=inset_image, face_box=(face.x, face.y, face.w, face.h),
                           shape=shape)
    picture = extract_frame(cut, frame, work / "frame.png")
    matte = (matte_provider or _default_matte)(picture, work)
    cutout = None
    if matte is not None:
        gray = ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(matte), "-frames:v", "1",
                           "-vf", "scale=1080:1920", "-pix_fmt", "gray", "-f", "rawvideo",
                           "pipe:1"], timeout_s=90, binary=True).stdout
        gray_image = np.frombuffer(gray, dtype=np.uint8).reshape(1920, 1080)
        coverage = float(np.count_nonzero(gray_image > 32)) / gray_image.size
        if coverage >= 0.005:
            if face_box is None:
                face_box = head_box_from_matte(gray_image)
            cutout = work / "cutout.png"
            document = (work / "matte/matte.json" if matte_provider is None
                        else matte.parent / "matte.json")
            backend = load_matte(document).backend if document.is_file() else None
            filters = ["[0:v]format=rgb24[photo]", "[1:v]scale=1080:1920,format=gray[alpha]"]
            photo, alpha = chroma.append_despill(filters, "photo", "alpha", backend)
            filters.append(f"[{photo}][{alpha}]alphamerge,format=rgba[out]")
            ffmpeg.run(["ffmpeg", "-y", "-v", "error", "-i", str(picture), "-i", str(matte),
                        "-filter_complex", ";".join(filters),
                        "-map", "[out]", "-frames:v", "1", "-pix_fmt", "rgba",
                        str(cutout)], timeout_s=120)
    if face_box is None and (look in {"number", "before-after"} or inset is not None):
        raise ValidationError("face", "no face in the face cache or person matte at this "
                              "frame; choose another frame")
    if look in {"number", "hook"} and cutout is None:
        raise ValidationError("matte", "no person found; choose another frame or photo look")
    blur = work / "blur.png"
    dark = work / "dark.png"
    if look == "hook":
        ffmpeg.run(["ffmpeg", "-y", "-v", "error", "-i", str(picture),
                    "-vf", "gblur=sigma=30", "-frames:v", "1", str(blur)], timeout_s=120)
    if look == "before-after":
        ffmpeg.run(["ffmpeg", "-y", "-v", "error", "-i", str(picture),
                    "-vf", "format=rgb24,colorchannelmixer=rr=0.75:gg=0.75:bb=0.75",
                    "-frames:v", "1",
                    str(dark)], timeout_s=120)
    inset_image = None
    if inset is not None:
        inset_image = work / "inset.png"
        ffmpeg.run(["ffmpeg", "-y", "-v", "error", "-i", str(inset),
                    "-vf", "scale=600:600:force_original_aspect_ratio=increase,crop=600:600",
                    "-frames:v", "1", str(inset_image)], timeout_s=120)
    return CoverAssets(picture, cutout, blur if blur.is_file() else None,
                       dark if dark.is_file() else None, matte, inset_image, face_box)


def _veil_stops(look: str, strength: float, blocks: list[dict],
                face_box: tuple[float, float, float, float] | None) -> list[tuple[float, float]]:
    """Describe the browser's vertical black gradient in frame fractions."""
    if look in {"photo", "hook"}:
        return [(0, strength), (.35, .85 * strength), (.72, 0), (1, 0)]
    if face_box is None:
        raise ValidationError("face", "face box unavailable; choose another frame")
    top = min(block["y"] for block in blocks) / 1920
    bottom = max(block["y"] + block["height"] for block in blocks) / 1920
    face_top = max(0, min(1, face_box[1] / 1920))
    face_bottom = max(0, min(1, (face_box[1] + face_box[3]) / 1920))
    fade = min(120, face_box[3] * .25) / 1920
    if bottom < face_top:
        after_end = max(block["y"] + block["height"] for block in blocks
                        if block["role"] == "after") / 1920
        stops = [(0, strength), (after_end, .8 * strength)]
        if bottom > after_end:
            stops.append((bottom, .35 * strength))
        return [*stops, (min(1, face_top + fade), 0), (1, 0)]
    after_top = min(block["y"] for block in blocks if block["role"] == "after") / 1920
    return [(0, 0), (max(0, face_bottom - fade), 0),
            (face_bottom, .2 * strength), (top, .7 * strength),
            (after_top, .85 * strength), (1, strength)]


def _veil_css(stops: list[tuple[float, float]], direction: str = "bottom") -> str:
    content = ", ".join(f"rgba(0,0,0,{alpha:.4f}) {position * 100:.5f}%"
                        for position, alpha in stops)
    return f"linear-gradient(to {direction}, {content})"


def _linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def _predicted_contrast(frame: np.ndarray, blocks: list[dict],
                        stops: list[tuple[float, float]], *,
                        horizontal: bool = False) -> dict[str, float]:
    """Predict glyph contrast from measured text spans and the actual veil profile."""
    positions, alphas = zip(*stops, strict=True)
    weights = np.array([.2126, .7152, .0722])
    results = {}
    for index, block in enumerate(blocks):
        size = block["font_size"]
        y0 = block["y"] + round(size * .10)
        y1 = min(block["y"] + block["height"], block["y"] + round(size * 1.08))
        rows = np.arange(y0, max(y0 + 1, y1), 3)
        veil = np.interp(rows / 1920, positions, alphas)[:, None, None]
        text = block["text"]
        start = block.get("accent_start", -1)
        end = block.get("accent_end", -1)
        spans = ([(0, start, block["color"]),
                  (start, end, block["accent_color"]),
                  (end, len(text), block["color"])] if start >= 0
                 else [(0, len(text), block["color"])])
        ratios = []
        small_caps = block["small_caps"]
        font = block["font"]
        tracking = .12 if small_caps else 0
        for first, last, color in spans:
            if first >= last:
                continue
            prefix = text[:first].upper() if small_caps else text[:first]
            through = text[:last].upper() if small_caps else text[:last]
            x0 = block["x"] + block.get("text_inset", 0) + round(
                text_width_em(prefix, font, 400, tracking_em=tracking) * size)
            x1 = block["x"] + block.get("text_inset", 0) + round(
                text_width_em(through, font, 400, tracking_em=tracking) * size)
            columns = np.arange(max(block["x"], x0),
                                min(block["x"] + block["width"], max(x0 + 1, x1)), 3)
            if not len(columns):
                continue
            if horizontal:
                veil = np.interp(columns / 1280, positions, alphas)[None, :, None]
            rgb = frame[np.ix_(rows, columns)][:, :, :3].astype(np.float64) / 255
            back = rgb * (1 - veil)
            foreground = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)]) / 255
            glyph_opacity = block["opacity"] * (
                .8 if block["role"] in {"kicker", "label", "proof"} else 1)
            visible = foreground * glyph_opacity + back * (1 - glyph_opacity)
            a = _linear(visible) @ weights
            b = _linear(back) @ weights
            ratios.append(((np.maximum(a, b) + .05) / (np.minimum(a, b) + .05)).ravel())
        if not ratios:
            raise ValidationError("text", f"{block['role']} has no measurable glyph area")
        results[str(index)] = round(float(np.percentile(np.concatenate(ratios), 10)), 3)
    return results


def _vignette(frame: np.ndarray, blocks: list[dict], look: str,
              face_box: tuple[float, float, float, float] | None,
              *, side: str | None = None,
              ) -> tuple[float, str, dict[str, float]]:
    """Choose the lightest veil whose predicted block contrasts pass."""
    # Narrow number strokes need more margin between box samples and measured glyphs.
    margin = .4 if side is not None and any(b["role"] == "number" for b in blocks) else .2
    for strength in np.arange(0, 1.001, .02):
        stops = _veil_stops(look, float(strength), blocks, face_box)
        if side is not None:
            stops = [(0, 0), (.38, 0), (.52, float(strength)), (1, float(strength))]
            if side == "right":
                stops = [(1 - position, alpha) for position, alpha in reversed(stops)]
        predicted = _predicted_contrast(frame, blocks, stops, horizontal=side is not None)
        if all(value >= (3 if blocks[int(index)]["role"] in {"headline", "number", "after"}
                         else 4.5) + margin for index, value in predicted.items()):
            return (round(float(strength), 2),
                    _veil_css(stops, "right" if side is not None else "bottom"), predicted)
    raise ValidationError("contrast", "text contrast cannot reach its minimum; "
                          "choose another frame or look")


def _config(assets: CoverAssets, style: ResolvedStyle, look: str,
            values: dict[str, str]) -> dict:
    if assets.shape == "horizontal":
        plan = plan_blocks(look, values, style.data, face_box=assets.face_box,
                           shape=assets.shape)
        blocks = [{**block, "font_stack": fonts.font_stack(block["font"])}
                  for block in plan.blocks]
        vignette, veil_css, predicted = _vignette(
            _rgba(assets.frame, (1280, 720)), blocks, "photo", assets.face_box,
            side=plan.speaker_side)
        return {**asdict(plan), "blocks": blocks, "look": look, "wide": True,
                "inset": assets.inset is not None, "vignette": vignette,
                "veil_css": veil_css, "predicted_contrast": predicted, "darkening": None}
    plan = plan_blocks(look, values, style.data, face_box=assets.face_box)
    blocks = [{**block, "font_stack": fonts.font_stack(block["font"])}
              for block in plan.blocks]
    if bleep.enabled(style.data):
        vocabulary = bleep.listed(style.data["music"])
        if any(bleep.mask_text(block["text"], vocabulary) != block["text"] for block in blocks):
            raise ValidationError("bleep.words", "Bleep check failed: mask the cover text.")
    if look == "hook":
        if assets.blur is None or assets.cutout is None:
            raise ValidationError("assets", "blurred frame and person cutout required")
        base = _rgba(assets.blur).astype(np.float64)
        person = _rgba(assets.cutout).astype(np.float64)
        alpha = person[:, :, 3:4] / 255
        base[:, :, :3] = person[:, :, :3] * alpha + base[:, :, :3] * (1 - alpha)
    else:
        picture = assets.dark if look == "before-after" else assets.frame
        if picture is None:
            raise ValidationError("assets", "darkened frame required")
        base = _rgba(picture)
    if look in {"photo", "hook", "before-after"}:
        vignette, veil_css, predicted = _vignette(base, blocks, look, assets.face_box)
    else:
        vignette, veil_css, predicted = 0, "", {}
    return {
        "look": look, "paper": style.data["graphics"]["paper_color"],
        "blocks": blocks, "speaker_side": plan.speaker_side,
        "speaker_shift_px": plan.speaker_shift_px,
        "inset": assets.inset is not None, "inset_x": plan.inset_x, "inset_y": plan.inset_y,
        "inset_side": plan.inset_side, "inset_scale": plan.inset_scale,
        "inset_width": plan.inset_width, "inset_height": plan.inset_height,
        "inset_padding": plan.inset_padding,
        "inset_bottom_padding": plan.inset_bottom_padding,
        "inset_box": plan.inset_box,
        "vignette": vignette, "veil_css": veil_css, "predicted_contrast": predicted,
        "darkening": ({"base_rgb_scale": .75, "gradient_strength": vignette,
                       "gradient": veil_css} if look == "before-after" else None),
    }


def speaker_overlap(text_alpha: np.ndarray, person_alpha: np.ndarray, shift_px: int,
                    blocks: list[dict]) -> int:
    """Count number and label glyph pixels over the shifted person cutout."""
    person = np.zeros_like(person_alpha, dtype=bool)
    canvas_width = person_alpha.shape[1]
    left, right = max(0, shift_px), min(canvas_width, canvas_width + shift_px)
    if left < right:
        person[:, left:right] = person_alpha[:, left - shift_px:right - shift_px] > 127
    overlap = 0
    for block in blocks:
        if block["role"] not in {"number", "label"}:
            continue
        x, y, width, height = (block[key] for key in ("x", "y", "width", "height"))
        threshold = max(32, round(block["opacity"] * 255 * .72))
        glyph = text_alpha[y:y + height, x:x + width] >= threshold
        overlap += int(np.count_nonzero(glyph & person[y:y + height, x:x + width]))
    return overlap


def render_cover(work: Path, assets: CoverAssets, style: ResolvedStyle,
                 look: str, values: dict[str, str]) -> RenderedCover:
    """Render three cover states, check their pixels, and retain only the full PNG."""
    paths.validate_output_dir(work)
    config = _config(assets, style, look, values)
    composition = paths.validate_output_dir(work / "composition")
    source = _TEMPLATE.read_text(encoding="utf-8")
    wide = assets.shape == "horizontal"
    if wide:
        source = source.replace("1080", "1280").replace("1920", "720")
    names = (style.data["hook"]["font"], style.data["graphics"]["font"])
    source = source.replace("/* BUNDLED_FONT_FACES */", fonts.font_face_css(names))
    fonts.copy_fonts(names, composition / "fonts")
    for name, asset in (("frame.png", assets.frame), ("cutout.png", assets.cutout),
                        ("blur.png", assets.blur), ("dark.png", assets.dark),
                        ("inset.png", assets.inset)):
        if asset is not None:
            shutil.copyfile(asset, composition / name)
    paths.atomic_write_text(composition / "index.html", source)
    clip = work / "cover.mov"
    hyperframes.render_layer(composition, clip, frames=3,
                             variables={"config": json.dumps(config)}, fps=30)
    full = work / "cover.png"
    backdrop = work / "backdrop.png"
    text = work / "text.png"
    try:
        for index, out in enumerate((full, backdrop, text)):
            _frame_png(clip, index, out)
        only_text = _rgba(text, (1280, 720)) if wide else _rgba(text)
        behind = _rgba(backdrop, (1280, 720)) if wide else _rgba(backdrop)
        blocks = config["blocks"]
        if look == "number" and not wide:
            overlap = speaker_overlap(only_text[:, :, 3], _rgba(assets.cutout)[:, :, 3],
                                      config["speaker_shift_px"], blocks)
            if overlap > 0:
                raise ValidationError("speaker", "number text would cover the speaker; "
                                      "choose another frame with --frame or another look")
        band = (style.data["internal"]["key_text_top_px"],
                style.data["internal"]["key_text_bottom_px"])
        geometry = check_geometry(only_text[:, :, 3], blocks=blocks, band=band,
                                  shape=assets.shape)
        checked = []
        accent_pixels = 0
        for block in blocks:
            x, y, width, height = (block[key] for key in ("x", "y", "width", "height"))
            pixels = only_text[y:y + height, x:x + width]
            ground = behind[y:y + height, x:x + width, :3]
            threshold = max(32, round(block["opacity"] * 255 * .72))
            glyph = pixels[:, :, 3] >= threshold
            yy, xx = np.nonzero(glyph)
            if not len(xx):
                raise ValidationError("text", f"{block['role']} did not render; shorten the text")
            measured_box = [x + int(xx.min()), y + int(yy.min()),
                            x + int(xx.max()) + 1, y + int(yy.max()) + 1]
            color = pixels[:, :, :3][glyph].astype(np.float64) / 255
            back = ground[glyph].astype(np.float64) / 255
            alpha = pixels[:, :, 3][glyph, None].astype(np.float64) / 255
            if block["opacity"] < 1:
                color = np.array([int(block["color"][i:i + 2], 16)
                                  for i in (1, 3, 5)], dtype=np.float64) / 255
            visible = color * alpha + back * (1 - alpha)
            linear_text = np.where(visible <= .04045, visible / 12.92,
                                   ((visible + .055) / 1.055) ** 2.4)
            linear_back = np.where(back <= .04045, back / 12.92,
                                   ((back + .055) / 1.055) ** 2.4)
            weights = np.array([.2126, .7152, .0722])
            a, b = linear_text @ weights, linear_back @ weights
            ratios = (np.maximum(a, b) + .05) / (np.minimum(a, b) + .05)
            p10 = float(np.percentile(ratios, 10))
            minimum = 3 if block["role"] in {"headline", "number", "after"} else 4.5
            if p10 < minimum:
                raise ValidationError("contrast", f"{block['role']} contrast {p10:.2f}:1 "
                                      f"is below {minimum}:1; choose another frame or look")
            checked.append({"role": block["role"], "box": [x, y, width, height],
                            "measured_box": measured_box, "font_size": block["font_size"],
                            "color": block["color"], "contrast_p10": round(p10, 3),
                            "predicted_contrast_p10": config["predicted_contrast"].get(
                                str(len(checked))), "minimum": minimum})
            accent = (block.get("accent_color") if block.get("accent_start", -1) >= 0
                      else block["color"] if block["role"] in {"after", "number"} else None)
            if accent:
                target = np.array([int(accent[i:i + 2], 16) for i in (1, 3, 5)])
                difference = np.max(np.abs(pixels[:, :, :3].astype(np.int16) - target), axis=2)
                accent_pixels += int(np.count_nonzero(glyph & (difference <= 20)))
        checks = {"geometry": geometry, "blocks": checked,
                  "contrast": {"passed": True,
                               "p10": min(item["contrast_p10"] for item in checked),
                               "accent_pixels": accent_pixels}}
        if wide:
            from yap_studio.covers.wide import check_face, check_legibility

            checks["legibility"] = check_legibility(checked, look=look)
            checks["face_clearance"] = check_face(blocks, assets.face_box)
        if look == "number" and not wide:
            checks["speaker_clearance"] = {"shift_px": config["speaker_shift_px"],
                                           "overlap_px": 0}
        if config["inset"]:
            checks["inset"] = {"box": config["inset_box"],
                               "scale": config["inset_scale"],
                               "side": config["inset_side"]}
        if config["darkening"] is not None:
            checks["darkening"] = config["darkening"]
    except BaseException:
        full.unlink(missing_ok=True)
        raise
    finally:
        backdrop.unlink(missing_ok=True)
        text.unlink(missing_ok=True)
        clip.unlink(missing_ok=True)
    return RenderedCover(full, checks, hyperframes.HYPERFRAMES_VERSION)
