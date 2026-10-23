"""Project caption words, refusals and layout for the camera look."""

import json
import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from yap_studio.captions import CaptionWord, build_phrases
from yap_studio.errors import YapError
from yap_studio.faces.clearance import LAYOUT_SCHEMA, Layout

PLAN_VERSION = "camera3d-look-12"
ELEMENT_FIX = "Remove those from the timeline, or pick another caption look."


class Camera3DRefusal(YapError):
    """Explain why this take cannot use the selected camera look."""

    def __init__(self, message: str, fix: str) -> None:
        super().__init__(message, error_type="refusal", code="camera3d_refused",
                         suggested_action=fix)


def require_cut(info, cut=None) -> None:
    error = Camera3DRefusal(
        "3D camera captions work on vertical 1080 by 1920 videos at 30 fps.",
        "Pick another caption look for this video.")
    if ((info.width, info.height, info.fps) != (1080, 1920, 30)
            or info.variable_frame_rate):
        raise error
    if cut is not None:
        from yap_studio.matte.timing import require_constant_rate

        try:
            require_constant_rate(cut, info)
        except YapError as exc:
            raise error from exc


def person_refusal(reason: str) -> Camera3DRefusal:
    return Camera3DRefusal(
        "3D camera captions need you on camera in every shot: " + reason.rstrip(".") + ".",
        "Pick another caption look, or keep only shots that show you.")


def require_still(motion: dict) -> None:
    if motion["moving"]:
        shot = next((shot for shot in motion.get("shots", []) if shot["moving"]), None)
        location = ""
        if shot:
            minutes, seconds = divmod(shot["start_frame"] // 30, 60)
            location = f" in the shot at {minutes}:{seconds:02}"
        shift = (shot or motion)["shift_p75_px"]
        raise Camera3DRefusal(
            f"This take looks handheld or moving{location}: the room shifts about "
            f"{shift:.1f} px every 0.2 s. "
            "3D camera captions need a camera that stays still.",
            "Record on a tripod, or pick another caption look.")


def incompatible(item) -> bool:
    payload = item.payload
    return (item.kind in {"zoom", "moment", "background"}
            or item.kind == "graphic" and (
                item.layer == "behind" or payload.get("template") == "big_number")
            or item.kind == "overlay" and payload.get("placement") == "tall_behind"
            or item.kind == "insert" and (
                payload.get("media", {}).get("fit") == "split"
                or str(payload.get("layout", "")).startswith("split")))


def element_message(items: Sequence) -> str:
    kinds = ", ".join(sorted({item.kind for item in items}))
    ids = ", ".join(sorted(item.id for item in items))
    return ("3D camera captions move the camera over your take, so they can't share a video "
            f"with {kinds}: {ids}.")


def refuse_elements(elements: Sequence) -> None:
    blocked = [item for item in elements if incompatible(item)]
    if blocked:
        raise Camera3DRefusal(element_message(blocked), ELEMENT_FIX)


def display_words(words: list[dict], captions: dict, *, word_fixes: dict,
                  vocabulary: Sequence[str], gaps: Sequence[tuple[float, float]],
                  word_text_fixes: dict | None = None) -> list[dict]:
    """Flatten the same corrected display words the caption engine produces."""
    phrases = build_phrases([CaptionWord(w["id"], w["text"], w["start"], w["end"],
                                        segment=w.get("segment"))
                             for w in words], captions, word_fixes=word_fixes,
                            word_text_fixes=word_text_fixes,
                            text_rules=True, vocabulary=vocabulary)
    return [{"id": word.id, "source_ids": list(word.source_ids), "text": word.text,
             "start": word.start, "end": word.end}
            for phrase in phrases for word in phrase.words
            if not any(word.start < end and word.end > start for start, end in gaps)]


def _title_envelope(plan, style: dict) -> list[float]:
    bottom = max(plan.y + plan.height, style["internal"]["key_text_bottom_px"])
    if plan.variables.get("look") == "kinetic":
        bottom = max(bottom, math.ceil((1920 + plan.height) / 2))
    return [plan.x, 0, plan.x + plan.width, bottom]


def _title_box(item, style: dict, vocabulary: Sequence[str], faces) -> list[float] | None:
    from yap_studio.stages.captions import title_layer_plan
    from yap_studio.stages.graphics import place_title
    from yap_studio.style import ResolvedStyle
    from yap_studio.text_rules import apply_title_rules

    if not item.payload.get("lines"):
        return None
    try:
        lines, accent = apply_title_rules(item.payload["lines"],
                                          item.payload.get("accent") or "", vocabulary)
        selected = ResolvedStyle(style["name"], "camera3d", style)
        plan = title_layer_plan("", accent, selected,
                                lines=lines, frames=item.end_frame - item.start_frame,
                                start_frame=item.start_frame)
    except YapError:
        return None
    if plan.template == "letters":
        from yap_studio.letters import CENTER_Y_PX, LINE_HEIGHT_EM, TEXT_WIDTH_PX

        height = plan.variables["size_px"] * (
            1 + (len(json.loads(plan.variables["lines"])) - 1) * LINE_HEIGHT_EM)
        return [(1080 - TEXT_WIDTH_PX) / 2, CENTER_Y_PX - height / 2,
                (1080 + TEXT_WIDTH_PX) / 2, CENTER_Y_PX + height / 2]
    if faces is None or faces.status != "measured":
        return _title_envelope(plan, style)
    try:
        plan, _ = place_title(plan, selected, faces)
    except YapError:
        return _title_envelope(plan, style)
    return [plan.x, plan.y, plan.x + plan.width, plan.y + plan.height]


def planner_windows(
    elements: Sequence, style: dict, *, vocabulary: Sequence[str] = (), faces=None,
) -> tuple[list[dict], list[tuple[int, int]]]:
    obstacles, hero_free = [], []
    gap = style["graphics"]["caption_gap_px"]
    for item in elements:
        box = None
        if item.kind == "title":
            planned = _title_box(item, style, vocabulary, faces)
            if planned:
                left, top, right, bottom = planned
                box = [left - gap, top - gap, right + gap, bottom + gap]
            else:
                box = [0, 0, 1080, style["internal"]["key_text_bottom_px"] + gap]
        elif item.kind == "graphic" and item.layer == "front":
            requested = item.payload.get("box")
            bottom = (requested["y"] + requested["height"] if requested
                      else style["internal"]["key_text_bottom_px"])
            box = [0, 0, 1080, bottom + gap]
        elif item.kind == "overlay" and "box_px" in item.payload:
            x, y, width, height = item.payload["box_px"]
            box = [x - gap, y - gap, x + width + gap, y + height + gap]
        elif item.kind == "chapter":
            box = [0, 240 - gap, 1080, 1560 + gap]
        if box:
            obstacles.append({"start_frame": item.start_frame, "end_frame": item.end_frame,
                              "box": box})
        if (item.kind in {"insert", "transition", "chapter"}
                or item.kind == "graphic" and item.layer == "front"
                and item.payload.get("template") == "reel_feed"):
            hero_free.append((item.start_frame, item.end_frame))
    return obstacles, hero_free


def picture_windows(elements: Sequence) -> list[tuple[int, int]]:
    """Keep hidden speakers out of caption head clearance."""
    return [(item.start_frame, item.end_frame) for item in elements
            if item.kind == "transition" or item.kind == "insert" and not (
                item.payload.get("media", {}).get("fit") == "split"
                or str(item.payload.get("layout", "")).startswith("split"))]


def split_at_gaps(clauses: list[list[dict]], gaps: Sequence[tuple[int, int]]) -> list[list[dict]]:
    """Keep words on opposite sides of a caption gap in separate groups."""
    result = []
    for clause in clauses:
        current = []
        previous = None
        for word in clause:
            section = sum(word["start"] * 30 >= end for _, end in gaps)
            if current and section != previous:
                result.append(current)
                current = []
            current.append(word)
            previous = section
        if current:
            result.append(current)
    return result


def clamp_gaps(item: dict, gaps: Sequence[tuple[int, int]], *, frames: int | None = None) -> bool:
    """Finish fades before a gap and resume on the first clear painter step."""
    words = item.get("words", [])
    for first, last in sorted(gaps):
        if item["off_frame"] <= first or item["on_frame"] >= last:
            continue
        off = item["off_frame"]
        resumed = 2 * math.ceil(last / 2)
        if (words and words[0]["start"] * 30 >= last
                or not words and first <= item["on_frame"] < last):
            if frames is not None and resumed >= frames:
                return False
            item["on_frame"] = resumed
        else:
            item["off_frame"] = first // 2 * 2
            item["fade_start_frame"] = min(item["fade_start_frame"], item["off_frame"] - 6)
        if item["off_frame"] <= item["on_frame"]:
            if not words or frames is not None and resumed >= frames:
                return False
            item["on_frame"] = resumed
            item["off_frame"] = max(off, item["on_frame"] + 2)
    if item["off_frame"] <= item["on_frame"]:
        if not words:
            return False
        item["off_frame"] = item["on_frame"] + 2
    if frames is not None:
        if item["on_frame"] >= frames:
            return False
        item["on_frame"] = min(item["on_frame"], (frames - 1) // 2 * 2)
        item["off_frame"] = min(item["off_frame"], frames)
        if item["off_frame"] <= item["on_frame"]:
            return False
    item["fade_start_frame"] = max(item["on_frame"], min(
        item["fade_start_frame"], item["off_frame"] - 1))
    return True


def _union(boxes: list[list[float]]) -> dict:
    left = math.floor(min(box[0] for box in boxes) * 100) / 100
    top = math.floor(min(box[1] for box in boxes) * 100) / 100
    right = math.ceil(max(box[2] for box in boxes) * 100) / 100
    bottom = math.ceil(max(box[3] for box in boxes) * 100) / 100
    return {"x": left, "y": top, "w": round(right - left, 2), "h": round(bottom - top, 2)}


def add_halos(plan: dict) -> None:
    """Give each word the lighter or darker halo with the stronger contrast."""
    from yap_studio.render.camera3d import _linear_luma

    style = plan["style"]["data"]
    accent = style.get("brand", {}).get("accent_color") or style["hook"].get(
        "accent_color", style["captions"]["highlight_color"])
    for item, color in [*( (group, style["captions"]["text_color"]) for group in plan["groups"]),
                        *((s["hero"], accent) for s in plan["sentences"] if s["hero"])]:
        rgb = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        luma = sum(weight * _linear_luma(value) for weight, value in
                   zip((.2126, .7152, .0722), rgb, strict=True))
        dark = (luma + .05) / (_linear_luma(16 / 255) + .05)
        light = 1.05 / (luma + .05)
        item["halo"] = {"color": "#101010" if dark >= light else "#FFFFFF",
                        "contrast_ratio": round(max(dark, light), 2)}


def check_display_words(plan: dict, words: list[dict]) -> None:
    """Require each display word to appear once in a group or hero."""
    expected = Counter(word["id"] for word in words)
    items = [*plan["groups"], *(s["hero"] for s in plan["sentences"] if s["hero"])]
    drawn = Counter(word["id"] for item in items for word in item["words"])
    missing, duplicate = expected - drawn, drawn - expected
    if missing or duplicate:
        raise ValueError("camera caption words missing: " + ", ".join(sorted(missing))
                         + "; duplicate: " + ", ".join(sorted(duplicate)))


def group_screen_box(group: dict, sample: dict, frame: int) -> list[float]:
    from yap_studio.render.camera3d import _HERO_ENTRY

    pose = group["poses"][frame - group["on_frame"]]
    step = max(0, min(len(_HERO_ENTRY) - 1, frame // 2 - group["on_frame"] // 2))
    rise = (0 if group["flat"] else 30 * _HERO_ENTRY[step]
            * group["depth"] / (group["depth"] - sample["cz"]))
    left, top, right, bottom = pose["box"]
    return [left, top + rise, right, bottom + rise]


def pose_boxes(boxes: list[list[float]], first: int) -> list[dict]:
    """Merge consecutive poses while every edge stays within eight pixels."""
    result = []
    start = first
    low, high = boxes[0][:], boxes[0][:]
    for frame, box in enumerate(boxes[1:], first + 1):
        next_low = [min(a, b) for a, b in zip(low, box, strict=True)]
        next_high = [max(a, b) for a, b in zip(high, box, strict=True)]
        if max(b - a for a, b in zip(next_low, next_high, strict=True)) > 8:
            result.append({**_union([low, high]), "start_frame": start, "end_frame": frame})
            start, low, high = frame, box[:], box[:]
        else:
            low, high = next_low, next_high
    result.append({**_union([low, high]), "start_frame": start, "end_frame": first + len(boxes)})
    return result


def plan_layout(plan: dict) -> dict:
    """Retain timed runs of drawn poses on the output frame grid."""
    from yap_studio.render.camera3d_geometry import painted_box

    captions = []
    items = [(f"caption-{index}", group) for index, group in enumerate(plan["groups"])]
    items += [(f"hero-{index}", sentence["hero"])
              for index, sentence in enumerate(plan["sentences"]) if sentence["hero"]]
    for name, item in items:
        first, last = item["on_frame"], min(plan["frames"], item["off_frame"])
        boxes = [painted_box(item, plan["camera"]["samples"], f,
                              hero=name.startswith("hero-"), metrics=plan.get("font_metrics"))
                 for f in range(first, last)]
        words = [{"id": word["id"], "text": word["text"],
                  "source_ids": word.get("source_ids", [word["id"]]),
                  "start_s": word["start"], "end_s": word["end"]}
                 for word in item["words"]]
        captions.append({"id": name, "text": item["text"], "start_frame": first,
                         "end_frame": last, "boxes": pose_boxes(boxes, first), "words": words,
                         **({"face_windows": []} if item.get("layer") == "behind" else {})})
    layout = {"schema": LAYOUT_SCHEMA, "width": 1080, "height": 1920,
              "measured": True, "reason": None, "elements": [], "captions": captions}
    Layout.from_dict(layout)
    return layout


def _halo_contrast(pixels: np.ndarray, box: list[float], color: str, halo: str) -> float:
    from yap_studio.checks.caption_contrast import _dilate, _luminance, _ratio
    from yap_studio.render.camera3d import caption_contrast

    ratio = caption_contrast(pixels, box, color)
    left, top = max(0, math.floor(box[0]) - 7), max(0, math.floor(box[1]) - 7)
    right, bottom = min(1080, math.ceil(box[2]) + 7), min(1920, math.ceil(box[3]) + 7)
    rgb = pixels[top:bottom, left:right, :3].astype(float)
    if not rgb.size:
        return max(1, ratio)
    target = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)])
    ground = np.array([int(halo[i:i + 2], 16) for i in (1, 3, 5)])
    ink = np.max(np.abs(rgb - target), axis=2) <= 35
    ring = _dilate(ink, 5) & ~ink
    core = ring & (np.max(np.abs(rgb - ground), axis=2) <= 45)
    if np.count_nonzero(ink) >= 40 and np.count_nonzero(core) >= max(40, .08 * ring.sum()):
        luma = _luminance(rgb.astype(np.uint8))
        ratio = max(ratio, _ratio(float(np.median(luma[ink])), float(np.median(luma[core]))))
    return round(max(1, min(21, ratio)), 3)


def measure_contrast(picture, plan: dict, layout: dict, *, grade=None) -> dict:
    """Measure every camera caption's rendered ink, including baked behind heroes."""
    from yap_studio import ffmpeg
    from yap_studio.checks.caption_contrast import _document
    from yap_studio.grade import apply_rgb
    from yap_studio.matte.check import _window_frames
    from yap_studio.render.camera3d_geometry import painted_box, sharp_frame

    style = plan["style"]["data"]
    accent = style.get("brand", {}).get("accent_color") or style["hook"].get(
        "accent_color", style["captions"]["highlight_color"])
    document = _document()
    probes = {}
    for caption in layout["captions"]:
        index = int(caption["id"].split("-")[1])
        hero = caption["id"].startswith("hero-")
        item = plan["sentences"][index]["hero"] if hero else plan["groups"][index]
        frame = sharp_frame(item, plan["camera"]["samples"], hero=hero)
        probes.setdefault(frame, []).append(caption)
    if not probes:
        return document
    for frame, pixels in enumerate(_window_frames(picture, ffmpeg.probe(picture), 0,
                                                   max(probes) + 1, rgba=True)):
        for caption in probes.get(frame, ()):
            hero = caption["id"].startswith("hero-")
            index = int(caption["id"].split("-")[1])
            item = plan["sentences"][index]["hero"] if hero else plan["groups"][index]
            color = accent if hero else style["captions"]["text_color"]
            halo = item["halo"]["color"]
            if item.get("layer") == "behind" and grade is not None and grade.active:
                colors = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)]
                                   for c in (color, halo)]) / 255
                color, halo = ["#" + "".join(f"{v:02X}" for v in row)
                               for row in np.rint(apply_rgb(colors, grade) * 255).astype(int)]
            bounds = painted_box(item, plan["camera"]["samples"], frame, hero=hero,
                                  metrics=plan.get("font_metrics"))
            ratio = _halo_contrast(pixels, bounds, color, halo)
            document["captions"].append({
                **{key: caption[key] for key in ("id", "text", "start_frame", "end_frame")},
                "ratio": ratio, "word": 0, "frame": frame,
                "words": [{"index": i, "text": word["text"], "ratio": ratio,
                           "frame": frame, "samples": 1}
                          for i, word in enumerate(caption["words"])],
            })
    return document
