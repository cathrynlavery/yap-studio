"""Measure caption ink against its picture in the final render."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from contextlib import closing
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from yap_studio import ffmpeg, paths
from yap_studio.checks.model import Finding, Thresholds, format_time
from yap_studio.errors import InputError
from yap_studio.faces.model import Box
from yap_studio.receipt import sha256_file
from yap_studio.taste.frames import frame_selection

CONTRAST_SUFFIX = ".contrast.json"
CONTRAST_SCHEMA = "yap-studio/caption-contrast@1"
SAMPLE_EVERY = 3
HELD_SAMPLES = 2
OPAQUE_ALPHA = 242
SURROUND_PERCENTILE = 75

_SRGB = np.arange(256, dtype=np.float64) / 255
_LINEAR = np.where(_SRGB <= 0.04045, _SRGB / 12.92, ((_SRGB + 0.055) / 1.055) ** 2.4)
_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return _LINEAR[rgb] @ _WEIGHTS


def _ratio(first: float, second: float) -> float:
    return (max(first, second) + 0.05) / (min(first, second) + 0.05)


def _color_luminance(color: str) -> float:
    return float(_luminance(np.frombuffer(bytes.fromhex(color[1:]), dtype=np.uint8)))


def _readable_ink(color: str) -> str:
    luminance = _color_luminance(color)
    dark = (luminance + 0.05) / 0.05560539162420272
    light = 1.05 / (luminance + 0.05)
    return "#000000" if max(dark, light) < 4.5 else "#111111" if dark >= light else "#FFFFFF"


def _ink_colors(style: Mapping[str, str]) -> dict[str, str]:
    """Match the solid fills in the caption template, including readable cores."""
    text, highlight = style["text_color"], style["highlight_color"]
    mode = style["highlight_style"]
    if mode in {"neon_glow", "neon_accent", "gradient_fill", "glitch_rgb",
                "particle_burst", "texture"}:
        if _ratio(_color_luminance(text), _color_luminance(highlight)) < 4.5:
            text = _readable_ink(highlight)
        highlight = text
    elif mode in {"moving_block", "pill", "highlight", "pill_karaoke"}:
        highlight = style.get("highlight_text_color", "#111111")
    elif mode in {"none", "underline", "weight_shift", "blend_difference", "parallax_layers"}:
        highlight = text
    return {"text": text, "highlight": highlight}


def _display_ratio(ratio: float) -> str:
    return f"{math.floor(ratio * 10) / 10:.1f}"


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Apply a square maximum in two one-dimensional passes."""
    result = mask
    for axis in (1, 0):
        padding = [(0, 0), (0, 0)]
        padding[axis] = (radius, radius)
        padded = np.pad(result, padding, constant_values=False)
        result = np.logical_or.reduce([
            padded[:, offset:offset + mask.shape[1]] if axis == 1 else
            padded[offset:offset + mask.shape[0], :]
            for offset in range(2 * radius + 1)
        ])
    return result


def _ink_groups(values: np.ndarray) -> list[float]:
    """Split ink and an opaque ground or outline with Otsu's threshold."""
    counts, edges = np.histogram(values, bins=64, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    weight = counts.cumsum()
    total = (counts * centers).cumsum()
    other = len(values) - weight
    score = np.zeros(64)
    valid = (weight > 0) & (other > 0)
    score[valid] = (total[-1] * weight[valid] - total[valid] * len(values)) ** 2 / (
        weight[valid] * other[valid]
    )
    threshold = edges[int(score.argmax()) + 1]
    low, high = values[values < threshold], values[values >= threshold]
    if min(len(low), len(high)) >= 0.08 * len(values):
        medians = [float(np.median(low)), float(np.median(high))]
        if _ratio(*medians) >= 3:
            return medians
    return [float(np.median(values))]


def word_contrast(final_rgb: np.ndarray, layer_rgba: np.ndarray,
                  box: tuple[float, float, float, float], *,
                  halo_ground: bool = True) -> float | None:
    """Compare a word's opaque ink with its own ground and busy picture surround."""
    return _word_contrast(final_rgb, layer_rgba, box, halo_ground=halo_ground)[0]


def _word_contrast(final_rgb: np.ndarray, layer_rgba: np.ndarray,
                   box: tuple[float, float, float, float], *,
                   halo_ground: bool) -> tuple[float | None, float | None]:
    """Keep the sampled ink luminance beside its contrast score."""
    x, y, width, height = box
    rows, columns = layer_rgba.shape[:2]
    left, top = max(0, math.floor(x - 12)), max(0, math.floor(y - 12))
    right = min(columns, math.ceil(x + width + 12))
    bottom = min(rows, math.ceil(y + height + 12))
    if right <= left or bottom <= top:
        return None, None
    layer = layer_rgba[top:bottom, left:right]
    opaque = layer[..., 3] >= OPAQUE_ALPHA
    if np.count_nonzero(opaque) < 40:
        return None, None
    values = _luminance(layer[..., :3][opaque])
    luminance = float(np.median(values))
    groups = _ink_groups(values)
    inner = _ratio(*groups) if len(groups) == 2 else 1
    radius = max(3, round(0.04 * height))
    ring = _dilate(opaque, radius) & ~_dilate(opaque, 1)
    if np.count_nonzero(ring) < 40:
        frame = np.zeros(opaque.shape, dtype=bool)
        frame[:2] = frame[-2:] = True
        frame[:, :2] = frame[:, -2:] = True
        opaque_frame = frame & opaque
        if np.count_nonzero(opaque_frame) < 0.9 * np.count_nonzero(frame):
            return None, None
        ground = float(np.median(_luminance(layer[..., :3][opaque_frame])))
        contrasts = (np.maximum(values, ground) + 0.05) / (np.minimum(values, ground) + 0.05)
        ink = values[contrasts >= 1.25]
        if len(ink) < 40:
            return 1.0, luminance
        luminance = float(np.median(ink))
        return round(max(_ratio(luminance, ground), inner), 2), luminance
    surround = _luminance(final_rgb[top:bottom, left:right][ring])
    median = float(np.median(surround))
    dark, light = np.percentile(surround, [100 - SURROUND_PERCENTILE, SURROUND_PERCENTILE])
    outer = max(_ratio(ink, float(light if ink >= median else dark)) for ink in groups)
    if halo_ground:
        # The halo core is a drawn ground; its feathered edge is still the picture.
        alpha = layer[..., 3][ring]
        paint = _luminance(layer[..., :3][ring])
        for ink in groups:
            contrast = (np.maximum(ink, paint) + 0.05) / (np.minimum(ink, paint) + 0.05)
            core = (alpha >= 128) & (alpha < OPAQUE_ALPHA) & (contrast >= 3)
            if np.count_nonzero(core) >= max(40, 0.08 * len(surround)):
                ground = surround[core]
                percentile = SURROUND_PERCENTILE if ink >= np.median(paint[core]) else \
                    100 - SURROUND_PERCENTILE
                inner = max(inner, _ratio(ink, float(np.percentile(ground, percentile))))
    return round(max(outer, inner), 2), luminance


def _document(*, measured: bool = True, reason: str | None = None) -> dict:
    return {
        "schema": CONTRAST_SCHEMA, "measured": measured, "reason": reason,
        "method": {"sample_every": SAMPLE_EVERY, "held_samples": HELD_SAMPLES,
                   "surround_percentile": SURROUND_PERCENTILE, "opaque_alpha": OPAQUE_ALPHA},
        "captions": [],
    }


def _frames(path: Path, shape: tuple[int, int, int], filters: str,
            count: int) -> Iterator[np.ndarray]:
    """Stream exact selected frames without resampling their timestamps."""
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0",
               "-an", "-vf", filters, "-fps_mode", "passthrough", "-frames:v", str(count),
               "-f", "rawvideo", "pipe:1"]
    picture = np.empty(shape, dtype=np.uint8)
    target = memoryview(picture).cast("B")
    used = decoded = 0
    with closing(ffmpeg.run_stream(command, timeout_s=ffmpeg.DEFAULT_TIMEOUT_S)) as stream:
        for chunk in stream:
            raw = memoryview(chunk)
            while raw:
                size = min(len(raw), len(target) - used)
                target[used:used + size] = raw[:size]
                raw = raw[size:]
                used += size
                if used == len(target):
                    decoded += 1
                    if decoded > count:
                        raise ValueError(f"{path}: decoded {decoded} frames; requested {count}")
                    yield picture
                    used = 0
        if used or decoded != count:
            detail = " and a partial frame" if used else ""
            raise ValueError(f"{path}: decoded {decoded} frames{detail}; requested {count}")


def measure_caption_contrast(
    final: Path, layer: Path, placement: Mapping[str, Any], layout: Mapping[str, Any],
    *, fps: float | Fraction = 30,
    halo_ground: bool = True,
    caption_style: Mapping[str, str] | None = None,
) -> dict:
    """Measure held word contrast on the layout's output frame grid."""
    if not layout["measured"]:
        return _document(measured=False,
                         reason=f"caption layout was not measured: {layout.get('reason')}")
    document = _document()
    colors = _ink_colors(caption_style) if caption_style is not None else {}
    luminances = {ink: _color_luminance(color) for ink, color in colors.items()}
    if colors:
        document["colors"] = colors
    captions = layout["captions"]
    if not captions:
        return document
    if not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError("caption contrast needs a positive frame rate")
    width, height = placement["width"], placement["height"]
    x, y, start = placement["x"], placement["y"], placement["start_frame"]
    end = start + placement["frames"]
    samples: dict[int, list[int]] = {}
    ratios: list[list[list[tuple[float, int, str | None]]]] = []
    boxes = []
    for index, caption in enumerate(captions):
        ratios.append([[] for _ in caption["words"]])
        boxes.append([(box["x"] - x, box["y"] - y, box["w"], box["h"])
                      for box in caption["boxes"]])
        if len(caption["words"]) != len(caption["boxes"]):
            continue
        frames = [frame for frame in range(caption["start_frame"] + 1, caption["end_frame"],
                                            SAMPLE_EVERY) if start <= frame < end]
        if not frames and start <= caption["start_frame"] < min(end, caption["end_frame"]):
            frames = [caption["start_frame"]]
        for frame in frames:
            samples.setdefault(frame, []).append(index)
    chosen = sorted(samples)
    if chosen:
        final_filter = (f"select='{frame_selection(chosen)}',format=rgb24,"
                        f"crop={width}:{height}:{x}:{y}")
        layer_selection = frame_selection([frame - start for frame in chosen])
        layer_filter = f"select='{layer_selection}',format=rgba"
        with (closing(_frames(final, (height, width, 3), final_filter, len(chosen))) as picture,
              closing(_frames(layer, (height, width, 4), layer_filter, len(chosen))) as ink):
            for frame, (rgb, rgba) in zip(chosen, zip(picture, ink, strict=True), strict=True):
                for index in samples[frame]:
                    for word, box in enumerate(boxes[index]):
                        if not Box.from_dict(captions[index]["boxes"][word]).overlaps_frames(
                            frame, frame + 1,
                        ):
                            continue
                        ratio, luminance = _word_contrast(rgb, rgba, box, halo_ground=halo_ground)
                        if ratio is not None:
                            role = min(luminances, key=lambda ink: abs(
                                luminances[ink] - luminance,
                            )) if luminances else None
                            ratios[index][word].append((ratio, frame, role))
    for caption, words in zip(captions, ratios, strict=True):
        measured_words = []
        for index, (word, values) in enumerate(zip(caption["words"], words, strict=True)):
            ratio, frame, ink = sorted(values)[min(HELD_SAMPLES, len(values)) - 1] if values else (
                None, None, None,
            )
            measured_words.append({"index": index, "text": word["text"], "ratio": ratio,
                                   "frame": frame, "samples": len(values),
                                   **({"ink": ink} if ink is not None else {})})
        worst = min((word for word in measured_words if word["ratio"] is not None),
                    key=lambda word: word["ratio"], default=None)
        document["captions"].append({
            **{key: caption[key] for key in ("id", "text", "start_frame", "end_frame")},
            "ratio": worst["ratio"] if worst else None,
            "word": worst["index"] if worst else None,
            "frame": worst["frame"] if worst else None, "words": measured_words,
            **({"ink": worst["ink"]} if worst and "ink" in worst else {}),
        })
    return document


def write_contrast_record(
    media: Path, document: Mapping[str, Any], out: Path | None = None,
) -> Path:
    """Atomically bind contrast measurements to the rendered bytes."""
    target = out if out is not None else media.with_suffix(CONTRAST_SUFFIX)
    source = paths.validate_input(media)
    paths.validate_output(target)
    try:
        record = {**document, "media_sha256": sha256_file(source)}
        return paths.atomic_write_text(
            target, json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        )
    except (OSError, ValueError) as exc:
        raise InputError(str(target), f"cannot write contrast record: {exc}") from exc


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate contrast record field")
        result[key] = value
    return result


def _validate_record(document: Any) -> None:
    if not isinstance(document, dict) or document.get("schema") != CONTRAST_SCHEMA:
        raise ValueError("unknown contrast record schema")
    if type(document.get("measured")) is not bool:
        raise ValueError("contrast record measured must be a boolean")
    if document.get("reason") is not None and not isinstance(document["reason"], str):
        raise ValueError("contrast record reason must be a string or null")
    if not isinstance(document.get("captions"), list):
        raise ValueError("contrast record captions must be a list")
    if "colors" in document:
        colors = document["colors"]
        if not isinstance(colors, dict) or colors.keys() != {"text", "highlight"}:
            raise ValueError("contrast colors must name text and highlight ink")
        for color in colors.values():
            if not isinstance(color, str) or len(color) != 7 or not color.startswith("#"):
                raise ValueError("contrast ink color must be a hex color")
            _color_luminance(color)
    for caption in document["captions"]:
        if not isinstance(caption, dict):
            raise ValueError("contrast caption must be an object")
        if caption.get("backing") not in (None, "shadow", "box"):
            raise ValueError("invalid contrast caption backing")
        if "auto" in caption and type(caption["auto"]) is not bool:
            raise ValueError("contrast caption auto must be a boolean")
        if not {"id", "text", "start_frame", "end_frame", "ratio", "word", "frame", "words"} \
                <= caption.keys():
            raise ValueError("contrast caption is missing measurement fields")
        for key in ("id", "text"):
            if not isinstance(caption.get(key), str):
                raise ValueError(f"contrast caption {key} must be text")
        for key in ("start_frame", "end_frame"):
            if type(caption.get(key)) is not int or caption[key] < 0:
                raise ValueError(f"contrast caption {key} must be a nonnegative integer")
        if caption["end_frame"] <= caption["start_frame"]:
            raise ValueError("contrast caption window must have positive length")
        if not isinstance(caption.get("words"), list):
            raise ValueError("contrast caption words must be a list")
        for item in [caption, *caption["words"]]:
            if not isinstance(item, dict):
                raise ValueError("contrast word must be an object")
            if "ink" in item and item["ink"] not in ("text", "highlight"):
                raise ValueError("contrast ink must be text or highlight")
            if not {"ratio", "frame"} <= item.keys():
                raise ValueError("contrast word is missing measurement fields")
            ratio = item.get("ratio")
            if ratio is not None and (type(ratio) not in (int, float)
                                      or not math.isfinite(ratio) or not 1 <= ratio <= 21):
                raise ValueError("contrast ratio must be finite and between 1 and 21")
            frame = item.get("frame")
            if ratio is None and frame is not None:
                raise ValueError("unmeasured contrast must have a null frame")
            if ratio is not None and (
                type(frame) is not int
                or not caption["start_frame"] <= frame < caption["end_frame"]
            ):
                raise ValueError("contrast frame must be inside its caption")
        for word in caption["words"]:
            if (type(word.get("index")) is not int or word["index"] < 0
                    or not isinstance(word.get("text"), str)
                    or type(word.get("samples")) is not int or word["samples"] < 0):
                raise ValueError("invalid contrast word")
            if (word["ratio"] is None) != (word["samples"] == 0):
                raise ValueError("contrast word samples must match its measurement")
        if caption["ratio"] is None and caption["word"] is not None:
            raise ValueError("unmeasured contrast must have a null word")
        if caption.get("ratio") is not None and not any(
            word["index"] == caption.get("word") and word["ratio"] == caption["ratio"]
            and word["frame"] == caption["frame"] for word in caption["words"]
        ):
            raise ValueError("contrast caption must name its measured word")


def read_contrast_record(media: Path) -> dict | None:
    """Read an optional contrast record, refusing corrupt or stale measurements."""
    target = media.with_suffix(CONTRAST_SUFFIX)
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        if target.is_symlink():
            raise InputError(str(target), "contrast record points to a missing file") from exc
        return None
    except (OSError, UnicodeError) as exc:
        raise InputError(str(target), "cannot read contrast record") from exc
    try:
        document = json.loads(text, object_pairs_hook=_json_object)
        _validate_record(document)
    except (ValueError, OverflowError, RecursionError) as exc:
        raise InputError(str(target), f"corrupt contrast record: {exc}") from exc
    source = paths.validate_input(media)
    try:
        digest = sha256_file(source)
    except OSError as exc:
        raise InputError(str(media), "cannot hash media for contrast record") from exc
    if digest != document.get("media_sha256"):
        raise InputError(str(target), "contrast record media hash does not match the media")
    return document


def check_caption_contrast(
    record: Mapping[str, Any], *, fps: float | Fraction, thresholds: Thresholds,
) -> tuple[list[Finding], dict]:
    """Flag phrases whose held word contrast loses to the picture."""
    captions = record["captions"]
    unmeasured = [caption for caption in captions if caption["ratio"] is None]
    coverage = {"ran": False, "reason": record.get("reason"), "unmeasured": len(unmeasured)}
    if unmeasured:
        first = min(caption["start_frame"] for caption in unmeasured)
        noun = "caption" if len(unmeasured) == 1 else "captions"
        coverage["note"] = (f"{len(unmeasured)} {noun} had no measurable word "
                            f"(first at {format_time(first / float(fps))})")
    if not record["measured"]:
        return [], coverage
    if not captions:
        return [], {**coverage, "reason": "no captions in the layout"}
    warn, fail = thresholds.caption_contrast_warn, thresholds.caption_contrast_fail
    flagged = [caption for caption in captions
               if caption["ratio"] is not None and caption["ratio"] < warn]
    coverage.update(ran=len(unmeasured) < len(captions),
                    reason="no caption word could be measured" if len(unmeasured) == len(captions)
                    else None, captions=len(captions), flagged=len(flagged))
    limits = {"warn_below": warn, "fail_below": fail}
    highlight_color = record.get("colors", {}).get("highlight")
    highlight_action = (
        "Choose a lighter or darker highlight colour on the style page, or a highlight "
        "style with its own ground such as Word background. If needed, put a box behind "
        "the weak phrases: yap-studio captions DIR --back box, then yap-studio run DIR again."
    )

    def word_text(caption: dict) -> str:
        return next(word["text"] for word in caption["words"] if word["index"] == caption["word"])

    if len(flagged) > max(10, len(captions) // 3):
        worst = sorted(flagged, key=lambda caption: caption["ratio"])[:10]
        failing = sum(caption["ratio"] < fail for caption in flagged)
        entries = [{"caption_id": caption["id"], "text": caption["text"],
                    "word": word_text(caption), "time_s": caption["frame"] / float(fps),
                    "ratio": caption["ratio"],
                    **({"ink": caption["ink"]} if "ink" in caption else {})} for caption in worst]
        examples = "; ".join(f'{format_time(item["time_s"])} "{item["word"]}" '
                             f'{_display_ratio(item["ratio"])}:1' for item in entries[:5])
        message = (f"{len(flagged)} of {len(captions)} captions are hard to read against the "
                   f"picture behind them ({failing} nearly vanish). Worst: {examples}")
        action = (
            "Most captions are weak on this footage, even with the halo. Put a box behind "
            "the weak phrases with yap-studio captions DIR --back box, or change the "
            "caption style for every phrase: a box ground "
            "(captions.background_mode box) or a stronger shadow (captions.shadow_opacity)."
        )
        if highlight_color and all(caption.get("ink") == "highlight" for caption in flagged):
            message = (
                "The highlighted word is hard to read on this footage: "
                f"in {len(flagged)} of {len(captions)} captions it drops to "
                f'{_display_ratio(worst[0]["ratio"])}:1 '
                f"({highlight_color} against the picture behind it)."
            )
            other_words = [word for caption in captions for word in caption["words"]
                           if word.get("ink") != "highlight"]
            if other_words and all(word.get("ink") == "text" and word["ratio"] is not None
                                   and word["ratio"] >= warn for word in other_words):
                message += f" The other words read at {warn:g}:1 or better."
            action = highlight_action
        return [Finding(
            "captions.contrast", "warn", message,
            time_s=entries[0]["time_s"],
            suggested_action=action,
            evidence={"flagged": len(flagged), "failing": failing, "captions": len(captions),
                      **limits, "worst": entries},
        )], coverage
    findings = []
    for caption in flagged:
        ratio, frame = caption["ratio"], caption["frame"]
        word = word_text(caption)
        time_s = frame / float(fps)
        display = _display_ratio(ratio)
        detail = (f"nearly vanishes ({display}:1 against the picture behind it)" if ratio < fail
                  else f"has {display}:1 contrast with the picture behind it")
        action = (
            "Back this phrase: yap-studio captions DIR --back shadow (a dark halo) or "
            "--back box (a box behind it), then yap-studio run DIR again."
        )
        if caption.get("backing") == "shadow":
            action = ("The halo is not enough here. Put a box behind this phrase: "
                      "yap-studio captions DIR --back box, then yap-studio run DIR again.")
        elif caption.get("backing") == "box":
            action = "Even the box is not enough here; check this moment by eye."
        message = (f'Caption "{caption["text"]}" at {format_time(time_s)} is hard to read: '
                   f'"{word}" {detail}; spoken words need at least {warn:g}:1.')
        if highlight_color and caption.get("ink") == "highlight":
            message = (
                f'The highlighted word "{word}" in caption "{caption["text"]}" at '
                f"{format_time(time_s)} is hard to read: {display}:1 "
                f"({highlight_color} against the picture behind it); "
                f"spoken words need at least {warn:g}:1."
            )
            action = highlight_action
        findings.append(Finding(
            "captions.contrast", "warn", message, time_s=time_s,
            suggested_action=action,
            evidence={"caption_id": caption["id"], "text": caption["text"], "word": word,
                      "word_index": caption["word"], "ratio": ratio, "frame": frame,
                      "start_frame": caption["start_frame"], "end_frame": caption["end_frame"],
                      **limits, **({"ink": caption["ink"]} if "ink" in caption else {})},
        ))
    return findings, coverage
