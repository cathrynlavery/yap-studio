"""Deterministic vertical camera caption planning and painting."""

from __future__ import annotations

import json
import math
import re
import shutil
import sys
import wave
from collections import Counter
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from functools import lru_cache, partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from yap_studio import ffmpeg
from yap_studio.captions import PhraseWord
from yap_studio.checks.model import Finding
from yap_studio.errors import YapError
from yap_studio.faces.model import FacesData
from yap_studio.matte.check import PERSON_ALPHA_THRESHOLD, _window_frames, measure_behind_layers
from yap_studio.matte.make import make_matte
from yap_studio.matte.model import Cleanup
from yap_studio.receipt import sha256_file
from yap_studio.render import hyperframes
from yap_studio.render.fonts import copy_fonts, font_face_css, font_stack
from yap_studio.render.text_metrics import _metrics as bundled_metrics
from yap_studio.render.text_metrics import text_width_em
from yap_studio.sfx import bleep
from yap_studio.style.loader import ResolvedStyle
from yap_studio.style.loader import resolve as resolve_style

TEMPLATE = Path(__file__).parents[1] / "templates" / "camera3d"
WIDTH, HEIGHT, FPS = 1080, 1920, 30
PLATE_DEPTH = 3000
BASE_PLATE_SCALE = 1.08
PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM = 240, 240, 480, 240
SAFE = (60, 220, 1020, 1500)
DEPTHS = (1250, 1500, 2200)
SIZES = (84, 72, 60)
LUMA_WIDTH, LUMA_HEIGHT, LUMA_FPS = 270, 480, 2
LIGHT_LUMA = 0.55
_HERO_ENTRY = (1, 0.564, 0.342, 0.188, 0.139, 0.09, 0.06, 0.035, 0.02, 0.002, 0)
_STOPS = frozenset(
    ("about after again along also another because before could every from have into just "
     "more only other people really should their there these those through today "
     "would your").split()
)
_HERO_EXCLUDE = _STOPS | frozenset(
    ("all already although anywhere anything anybody anyone absolutely because basically could "
     "different every exact heavy plain technical welcome where "
     "everybody everyone everything generally gonna have having here herself himself "
     "itself literally myself never nothing ourselves really somebody someone something "
     "suddenly themselves there these those whatever whenever wherever whether which "
     "whose yourself yourselves actually essentially want wanna with would "
     "i me my mine you your yours he him his she her hers it its we us our ours "
     "they them their theirs who whom what this that think thinking thought know knowing "
     "make making made need needed use using used work working worked connect connecting "
     "connected talk talking tell telling told say saying said see seeing seen do doing "
     "done get getting got give giving gave take taking took go going went explain "
     "explained understand remember wish wishing buy buying place placed deliver "
     "delivering log logging put share sharing stand stands touch end thing stuff").split()
)
# Losing more than a tenth of the ink removes most of a letter in a 6 to 12 letter word.
MIN_BEHIND_VISIBLE_SHARE = 0.90
CAPTION_MAX_SIZE = max(SIZES)
CAPTION_MIN_SIZE = min(SIZES)
CAPTION_MIN_PAINTED_SIZE = CAPTION_MIN_SIZE * .9
_ACTIVE_STYLE: ContextVar[dict | None] = ContextVar("camera3d_style", default=None)
_SAFE_TOP: ContextVar[int] = ContextVar("camera3d_safe_top", default=SAFE[1])


@lru_cache(maxsize=1)
def _metrics() -> dict:
    return json.loads((TEMPLATE / "metrics.json").read_text(encoding="utf-8"))


def _advance(text: str, font: str) -> float:
    style = _ACTIVE_STYLE.get()
    if style is None:
        table = _metrics()["adv"][font]
        return sum(table.get(char, table.get("_max", 0.65)) for char in text)
    family = style["hook" if font == "hero" else "captions"]["font"]
    weight = 400 if font == "hero" else _caption_weight(style)
    return text_width_em(text, family, weight)


def _caption_weight(style: dict) -> int:
    return 800 if style["captions"]["highlight_style"] in {"slam", "clip_wipe"} else 400


def measure_silhouette(alpha: np.ndarray) -> dict[str, Any]:
    """Measure a head box and shoulder line from one portrait person mask."""
    if alpha.ndim != 2 or alpha.shape != (HEIGHT, WIDTH):
        raise ValueError("expected a 1920 by 1080 alpha array")
    mask = alpha >= 128
    occupied = np.flatnonzero(mask.any(axis=1))
    if len(occupied) == 0:
        raise ValueError("person matte has no visible person")
    top = int(occupied[0])
    widths = mask.sum(axis=1)
    head_span = widths[top : min(HEIGHT, top + 500)]
    head_width = int(np.percentile(head_span[head_span > 0], 85))
    shoulder = next(
        (
            int(y)
            for y in range(top + 100, HEIGHT)
            if widths[y] > max(head_width * 1.55, WIDTH * 0.38)
        ),
        min(HEIGHT - 1, top + 520),
    )
    head_pixels = np.argwhere(mask[top:shoulder])
    if head_pixels.size == 0:
        raise ValueError("person matte has no head region")
    head = [
        int(head_pixels[:, 1].min()),
        top + int(head_pixels[:, 0].min()),
        int(head_pixels[:, 1].max()) + 1,
        top + int(head_pixels[:, 0].max()) + 1,
    ]
    body_pixels = np.argwhere(mask)
    body = [
        int(body_pixels[:, 1].min()),
        int(body_pixels[:, 0].min()),
        int(body_pixels[:, 1].max()) + 1,
        int(body_pixels[:, 0].max()) + 1,
    ]
    rows = []
    for y in range(0, HEIGHT, 16):
        block = np.flatnonzero(mask[y : min(y + 16, HEIGHT)].any(axis=0))
        if block.size:
            rows.append([y, int(block[0]), int(block[-1]) + 1])
    return {"head_box": head, "shoulder_y": shoulder, "body_box": body, "rows": rows}


def _sentences(words: list[dict]) -> list[list[dict]]:
    out: list[list[dict]] = []
    current: list[dict] = []
    for word in words:
        if current and word["start"] - current[-1]["end"] > 0.6:
            out.append(current)
            current = []
        current.append(word)
        if re.search(r"[.!?][\"'’”]*$", word["text"]):
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out


def _clauses(words: list[dict], max_words: int = 4,
             max_characters: int | None = None) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    for word in words:
        connective = word["text"].lower().strip(",.;:!?\"'’”") in {
            "and",
            "but",
            "so",
            "which",
            "because",
        }
        length = len(" ".join(_screen_word(w["text"]) for w in [*current, word]))
        if current and (len(current) >= max_words or connective
                        or max_characters is not None and length > max_characters):
            groups.append(current)
            current = []
        current.append(word)
        if word["text"].endswith(","):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _screen_word(text: str) -> str:
    style = _ACTIVE_STYLE.get() or {}
    if bleep.enabled(style):
        text = bleep.mask_text(text, bleep.listed(style["music"]))
    return re.sub(r"\.+(?=[\"'’”]*$)", "", text)


def _term_key(text: str) -> str:
    clean = re.sub(r"[^\w-]", "", text).casefold()
    if len(clean) > 3 and clean.endswith("s") and not clean.endswith(("ss", "us")):
        return clean[:-1]
    return clean


def _hero_word(words: list[dict], frequencies: Counter[str],
               forbidden: set[str] | None = None) -> str | None:
    candidates = []
    listed = bleep.listed((_ACTIVE_STYLE.get() or {}).get("music", {}))
    for index, word in enumerate(words):
        text = word["text"]
        if "'" in text or "’" in text:
            continue
        clean = re.sub(r"[^\w-]", "", text)
        key = _term_key(clean)
        acronym = clean.isupper() and len(clean) >= 2
        if ((len(clean) < 5 and not acronym and not (len(clean) >= 4 and
                                                    frequencies[key] >= 3))
                or key in _HERO_EXCLUDE or clean.casefold() in _HERO_EXCLUDE
                or key in (forbidden or ())
                or word.get("_bleep", False)
                or bleep.is_listed(text, listed)):
            continue
        capital_term = index > 0 and clean[0].isupper()
        noun_hint = not clean.casefold().endswith(("ly", "ing", "ed"))
        score = (min(frequencies[key], 5), capital_term, noun_hint, len(clean), -index)
        candidates.append((score, clean))
    return max(candidates)[1] if candidates else None


def _word_onset(word: dict, lead: float = .2) -> int:
    if lead == .2:
        return max(0, 2 * round((word["start"] - lead) * 15))
    return max(0, min(math.floor(word["start"] * FPS),
                      2 * math.ceil(round((word["start"] - lead) * 15, 9))))


def _box_luma(box: list[float], sample: np.ndarray, plate: dict, camera: dict) -> float:
    """Mean cut-picture luma below a screen box at one camera pose."""
    real = project_real_picture_box(plate, camera, WIDTH, HEIGHT)
    x0 = max(0, min(LUMA_WIDTH - 1, int((box[0] - real[0]) / (real[2] - real[0])
                                        * LUMA_WIDTH)))
    x1 = max(x0 + 1, min(LUMA_WIDTH, math.ceil((box[2] - real[0])
                                                / (real[2] - real[0]) * LUMA_WIDTH)))
    y0 = max(0, min(LUMA_HEIGHT - 1, int((box[1] - real[1]) / (real[3] - real[1])
                                         * LUMA_HEIGHT)))
    y1 = max(y0 + 1, min(LUMA_HEIGHT, math.ceil((box[3] - real[1])
                                                 / (real[3] - real[1]) * LUMA_HEIGHT)))
    return float(np.clip((sample[y0:y1, x0:x1].mean() - 16) / 219, 0, 1))


def _linear_luma(value: float) -> float:
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _label_luminance(groups: list[dict], samples: np.ndarray, plate: dict,
                     camera: list[dict], frames: int) -> None:
    """Mark light backgrounds for a tight dark text halo."""
    halo_luma = _linear_luma(16 / 255)
    for group in groups:
        first = max(0, math.ceil(group["on_frame"] * LUMA_FPS / FPS))
        last = min(len(samples), math.ceil(min(group["off_frame"], frames)
                                           * LUMA_FPS / FPS))
        indices = list(range(first, last))
        if not indices:
            midpoint = (group["on_frame"] + min(group["off_frame"], frames)) / 2
            indices = [min(len(samples) - 1, round(midpoint * LUMA_FPS / FPS))]
        values = []
        for index in indices:
            frame = min(frames - 1, round(index * FPS / LUMA_FPS))
            pose = group["poses"][max(0, min(len(group["poses"]) - 1,
                                            frame - group["on_frame"]))]
            values.append(_box_luma(pose["box"], samples[index], plate, camera[frame]))
        group["background_luma"] = round(float(np.mean(values)), 4)
        group["background_luma_peak"] = round(max(values), 4)
        if max(values) > LIGHT_LUMA:
            contrast = (_linear_luma(max(values)) + 0.05) / (halo_luma + 0.05)
            group["halo"] = {"color": "#101010", "contrast_ratio": round(contrast, 2)}
        else:
            group["halo"] = None


def check_hero_intervals(sentences: list[dict], onset_lead: float = .2) -> None:
    """Reject heroes that outlive or overlap their own sentence window."""
    previous_off = 0
    for index, sentence in enumerate(sentences):
        hero = sentence["hero"]
        if hero is None:
            continue
        if not (previous_off <= hero["on_frame"] <= hero["fade_start_frame"]
                < hero["off_frame"]):
            raise ValueError(f"hero timing overlaps or reverses at sentence {index}")
        next_start = (sentences[index + 1]["start"] - onset_lead
                      if index + 1 < len(sentences) else float("inf"))
        if onset_lead != .2 and index + 1 < len(sentences):
            next_start = _word_onset({"start": sentences[index + 1]["start"]}, onset_lead) / FPS
        limit = min(sentence["end"], next_start)
        if hero["off_frame"] / FPS > limit + 1 / FPS:
            raise ValueError(f"hero outlives sentence {index}")
        previous_off = hero["off_frame"]


def check_hero_repetition(sentences: list[dict]) -> None:
    """Keep each folded hero term to two uses and separate its appearances."""
    counts: Counter[str] = Counter()
    previous: str | None = None
    for index, sentence in enumerate(sentences):
        hero = sentence["hero"]
        if hero is None:
            continue
        key = _term_key(hero["text"])
        if key == previous or counts[key] >= 2:
            raise ValueError(f"hero repeats too often at sentence {index}: {hero['text']}")
        counts[key] += 1
        previous = key


def _ease(name: str, u: float) -> float:
    u = min(1.0, max(0.0, u))
    if name == "linear":
        return u
    if name == "push":
        return u**1.9
    if name == "arrive":
        arr = (
            1,
            0.740,
            0.575,
            0.449,
            0.358,
            0.289,
            0.227,
            0.175,
            0.133,
            0.110,
            0.088,
            0.070,
            0.056,
            0.044,
            0.033,
            0.024,
            0.016,
            0.009,
            0.004,
            0,
        )
        x = u * 19
        i = min(18, int(x))
        return 1 - (arr[i] + (arr[i + 1] - arr[i]) * (x - i)) if u < 1 else 1
    if u < 0.25:
        return 0.354 * (u / 0.25) ** 3
    return 0.354 + 0.646 * _ease("arrive", (u - 0.25) / 0.75)


def _table_value(values: list[float], frame: float) -> float:
    i = min(len(values) - 2, max(0, int(frame)))
    return values[i] + (values[i + 1] - values[i]) * (frame - i)


def _camera_samples(sentences: list[list[dict]], frames: int) -> tuple[list[dict], list[dict]]:
    source = (TEMPLATE / "tables.js").read_text(encoding="utf-8")
    table = {}
    for key in ("m", "px", "py"):
        match = re.search(r"\b" + key + r": \[([^]]+)\]", source)
        if match is None:
            raise ValueError(f"missing camera table {key}")
        table[key] = [float(value) for value in match[1].split(",") if value.strip()]
    segments = []
    whips = [{"kind": "opening", "start_frame": 0, "end_frame": min(8, frames),
              "padding_frames": []}]
    truck = 0
    for index, sentence in enumerate(sentences[1:], 1):
        onset = max(0.0, float(sentence[0]["start"]) - 0.2)
        target = 40 if index % 2 else -40
        segments.append(
            {"t0": max(8 / FPS, onset - 0.1), "dur": 0.65,
             "prof": "swing", "tx": target - truck}
        )
        truck = target
        if (index + 1) % 3 == 0:
            start = max(8, round((onset - 0.1) * FPS))
            if start < frames:
                whips.append({"kind": "sentence", "sentence": index,
                              "start_frame": start, "end_frame": min(start + 8, frames),
                              "padding_frames": []})
    if len(sentences) > 1:
        start = max(8, round((float(sentences[-1][0]["start"]) - 0.2) * FPS))
        if start < frames:
            whips.append({"kind": "finale", "start_frame": start,
                          "end_frame": min(start + 8, frames), "padding_frames": []})
    first_truck = min((segment["t0"] for segment in segments), default=1.2)
    wide_hold = min(0.8, max(0.2, first_truck - 0.2))
    wide_end = min(1.2, max(wide_hold + 1 / FPS, first_truck))
    samples = []
    for frame in range(frames + 1):
        t = frame / FPS
        creep = min(1.0, t / max(1 / FPS, frames / FPS))
        opening = math.sin(math.pi * frame / 8) if 0 <= frame < 8 else 0.0
        sentence_pulse = 0.0
        finale_pulse = 0.0
        for whip in whips[1:]:
            u = (frame - whip["start_frame"]) / 8
            if 0 < u < 1:
                pulse = math.sin(math.pi * u)
                if whip["kind"] == "finale":
                    finale_pulse = max(finale_pulse, pulse)
                else:
                    sentence_pulse = max(sentence_pulse, pulse)
        if t < wide_hold:
            base_m = 1 / BASE_PLATE_SCALE
        elif t < wide_end:
            base_m = 1 / BASE_PLATE_SCALE + (1 - 1 / BASE_PLATE_SCALE) * _ease(
                "arrive", (t - wide_hold) / (wide_end - wide_hold)
            )
        else:
            base_m = 1.0
        m = base_m + 0.008 * creep + max(0.15 * opening, 0.12 * sentence_pulse,
                                        0.14 * finale_pulse)
        px = _table_value(table["px"], min(19, frame + 8)) * HEIGHT / 1080 * 0.12 * opening
        py = (_table_value(table["py"], min(19, frame + 8)) * HEIGHT / 1080
              * 0.45 * opening - 8 * creep
              - 24 * max(sentence_pulse, finale_pulse))
        tx = 0.0
        for segment in segments:
            p = _ease(segment["prof"], (t - segment["t0"]) / segment["dur"])
            tx += segment["tx"] * p
        samples.append(
            {
                "frame": frame,
                "t": round(t, 6),
                "m": round(m, 7),
                "cz": round(PLATE_DEPTH * (1 - 1 / m), 6),
                "px": round(px, 5),
                "py": round(py, 5),
                "tx": round(tx, 5),
                "ty": 0,
            }
        )
    return samples, whips


def _project(camera: dict, x: float, y: float, depth: float) -> tuple[float, float, float]:
    distance = depth - camera["cz"]
    scale = depth / distance
    lens = 1326.1 * HEIGHT / 1080
    return (
        WIDTH / 2 + (x - WIDTH / 2) * scale + camera["px"] - camera["tx"] * lens / distance,
        HEIGHT / 2 + (y - HEIGHT / 2) * scale + camera["py"] - camera["ty"] * lens / distance,
        scale,
    )


def _unproject(camera: dict, x: float, y: float, depth: float) -> tuple[float, float]:
    distance = depth - camera["cz"]
    scale = depth / distance
    lens = 1326.1 * HEIGHT / 1080
    return (
        WIDTH / 2 + (x - camera["px"] + camera["tx"] * lens / distance - WIDTH / 2) / scale,
        HEIGHT / 2 + (y - camera["py"] + camera["ty"] * lens / distance - HEIGHT / 2) / scale,
    )


def project_plate_corners(plate: dict, camera: dict, width: int, height: int) -> list[list[float]]:
    """Project the four padded plate corners using the painter's plate transform."""
    depth = plate["depth"]
    scale = plate["scale"] * depth / (depth - camera["cz"])
    lens = 1326.1 * height / 1080
    x0 = (
        width / 2
        + (plate["ref_x"] - plate["ref_u"] * plate["scale"] - width / 2)
        * depth
        / (depth - camera["cz"])
        + camera["px"]
        - camera["tx"] * lens / (depth - camera["cz"])
    )
    y0 = (
        height / 2
        + (plate["ref_y"] - plate["ref_v"] * plate["scale"] - height / 2)
        * depth
        / (depth - camera["cz"])
        + camera["py"]
        - camera["ty"] * lens / (depth - camera["cz"])
    )
    return [
        [x0, y0],
        [x0 + plate["width"] * scale, y0],
        [x0 + plate["width"] * scale, y0 + plate["height"] * scale],
        [x0, y0 + plate["height"] * scale],
    ]


def project_real_picture_box(plate: dict, camera: dict, width: int,
                             height: int) -> list[float]:
    """Project the unpadded cut rectangle into the output frame."""
    corner = project_plate_corners(plate, camera, width, height)[0]
    scale = plate["scale"] * plate["depth"] / (plate["depth"] - camera["cz"])
    left = corner[0] + plate["pad_left"] * scale
    top = corner[1] + plate["pad_top"] * scale
    return [left, top, left + width * scale, top + height * scale]


def check_camera_coverage(plate: dict, samples: list[dict], whips: list[dict],
                          width: int = WIDTH, height: int = HEIGHT) -> dict:
    """Reject visible plate edges, excessive padding, or persistent punches."""
    if plate["scale"] > 1.12:
        raise ValueError("base plate scale exceeds 1.12")
    allowed = {frame for whip in whips
               for frame in range(whip["start_frame"], whip["end_frame"])}
    for whip in whips:
        if whip["end_frame"] - whip["start_frame"] > 8:
            raise ValueError("whip window exceeds eight frames")
        whip["padding_frames"] = []
    padding_frames = []
    run = 0
    maximum = 0.0
    for frame, sample in enumerate(samples[:-1]):
        corners = project_plate_corners(plate, sample, width, height)
        if (min(point[0] for point in corners) > 0
                or max(point[0] for point in corners) < width
                or min(point[1] for point in corners) > 0
                or max(point[1] for point in corners) < height):
            raise ValueError(f"padded plate edge visible at frame {frame}")
        left, top, right, bottom = project_real_picture_box(plate, sample, width, height)
        real_width = max(0.0, min(width, right) - max(0.0, left))
        real_height = max(0.0, min(height, bottom) - max(0.0, top))
        fraction = max(0.0, 1 - real_width * real_height / (width * height))
        maximum = max(maximum, fraction)
        scale = plate["scale"] * sample["m"]
        if scale > (1.25 if frame in allowed else 1.12) + 1e-6:
            raise ValueError(f"camera scale exceeds framing limit at frame {frame}")
        if fraction > 0.2 + 1e-6:
            raise ValueError(f"padding exceeds 20 percent at frame {frame}")
        if fraction > 1e-6:
            if frame == 0:
                raise ValueError("padding visible in the first frame")
            if frame not in allowed:
                raise ValueError(f"padding visible outside a whip at frame {frame}")
            padding_frames.append({"frame": frame, "fraction": round(fraction, 6)})
            for whip in whips:
                if whip["start_frame"] <= frame < whip["end_frame"]:
                    whip["padding_frames"].append(frame)
            run += 1
            if run > 8:
                raise ValueError("padding visible for more than eight consecutive frames")
        else:
            run = 0
    return {"max_fraction": round(maximum, 6), "frames": padding_frames}


def _overlap(a: list[float], b: list[float], gap: float = 0) -> bool:
    return a[0] < b[2] + gap and b[0] < a[2] + gap and a[1] < b[3] + gap and b[1] < a[3] + gap


def _hero_visible(box: list[float], silhouette: dict) -> float:
    area = (box[2] - box[0]) * (box[3] - box[1])
    covered = 0.0
    for y, left, right in silhouette["rows"]:
        overlap_y = max(0, min(box[3], y + 16) - max(box[1], y))
        overlap_x = max(0, min(box[2], right) - max(box[0], left))
        covered += overlap_x * overlap_y
    return max(0.0, 1 - covered / area)


def _hero_screen_box(hero: dict, camera: dict, frame: int) -> list[float]:
    """Project the visible hero, including its posterized entrance rise."""
    step = max(0, min(len(_HERO_ENTRY) - 1, frame // 2 - hero["on_frame"] // 2))
    x, baseline, scale = _project(
        camera, hero["ref_x"], hero["ref_y"] + 140 * _HERO_ENTRY[step], hero["depth"]
    )
    size = hero["ref_size"] * scale
    top = baseline - _metrics()["base"]["hero"] * size
    return [x, top, x + sum(hero.get("advances", [])) * size, top + size]


def _glyph_visible_share(hero: dict, camera: dict, frame: int, alpha: np.ndarray) -> float:
    """Count projected per-letter advance boxes inside the frame against person alpha."""
    box = _hero_screen_box(hero, camera, frame)
    size = box[3] - box[1]
    origin_x, origin_y, person_scale = _person_transform(camera)
    cursor = box[0]
    total = covered = 0
    for advance in hero["advances"]:
        x0, x1 = max(0, math.floor(cursor)), min(WIDTH, math.ceil(cursor + advance * size))
        y0, y1 = max(0, math.floor(box[1] + size * .1)), min(HEIGHT, math.ceil(box[3] - size * .1))
        if x1 > x0 and y1 > y0:
            total += (x1 - x0) * (y1 - y0)
            sx0 = max(0, math.floor((x0 - origin_x) / person_scale))
            sx1 = min(WIDTH, math.ceil((x1 - origin_x) / person_scale))
            sy0 = max(0, math.floor((y0 - origin_y) / person_scale))
            sy1 = min(HEIGHT, math.ceil((y1 - origin_y) / person_scale))
            if sx1 > sx0 and sy1 > sy0:
                covered += min((x1 - x0) * (y1 - y0),
                               int(np.count_nonzero(alpha[sy0:sy1, sx0:sx1] >=
                                                    PERSON_ALPHA_THRESHOLD)
                                   * person_scale ** 2))
        cursor += advance * size
    return 1 - covered / total if total else 0.0


def _person_transform(camera: dict) -> tuple[float, float, float]:
    """Match the painter's person transform at plate depth."""
    k = PLATE_DEPTH / (PLATE_DEPTH - camera["cz"])
    person_scale = BASE_PLATE_SCALE * k
    lens = 1326.1 * HEIGHT / 1080
    shift_x = camera["px"] - camera["tx"] * lens / (PLATE_DEPTH - camera["cz"])
    shift_y = camera["py"] - camera["ty"] * lens / (PLATE_DEPTH - camera["cz"])
    origin_x = WIDTH / 2 * (1 - person_scale) + shift_x
    origin_y = HEIGHT / 2 * (1 - person_scale) + shift_y
    return origin_x, origin_y, person_scale


def face_transforms(plan: dict) -> list[tuple[float, float, float]]:
    """Return each output frame's scale and source pixel origin."""
    return [(scale, x, y) for x, y, scale in
            (_person_transform(sample) for sample in plan["camera"]["samples"][:plan["frames"]])]


def _render_person_covered(ink: np.ndarray, matte: np.ndarray, frame: int,
                           x: int, y: int, *, camera: list[dict]) -> int:
    """Map rendered glyph pixels into that frame's source person matte."""
    rows, columns = np.nonzero(ink)
    origin_x, origin_y, scale = _person_transform(camera[frame])
    source_x = np.floor((columns + x + .5 - origin_x) / scale).astype(np.int32)
    source_y = np.floor((rows + y + .5 - origin_y) / scale).astype(np.int32)
    valid = ((source_x >= 0) & (source_x < matte.shape[1]) &
             (source_y >= 0) & (source_y < matte.shape[0]))
    return int(np.count_nonzero(matte[source_y[valid], source_x[valid]] >=
                                PERSON_ALPHA_THRESHOLD))


def _place_hero(hero: dict, section: dict, camera: list[dict],
                matte: Path | None, matte_info: ffmpeg.MediaInfo | None,
                obstacles: list[list[float]] = (), *, behind_only: bool = False) -> dict | None:
    """Try readable behind positions in order, streaming only this word's matte window."""
    initial = hero["box"]
    head = section["head_box"]
    width = initial[2] - initial[0]
    size = hero["size"]
    choices = [("behind", initial)]
    for x in (head[0] - width - 24, head[2] + 24):
        if SAFE[0] <= x and x + width <= SAFE[2]:
            choices.append(("beside", [x, initial[1], x + width, initial[3]]))
    higher_top = head[1] - size - 24
    if higher_top >= _SAFE_TOP.get():
        choices.append(("above", [(WIDTH - width) / 2, higher_top,
                                   (WIDTH + width) / 2, higher_top + size]))
    small = min(size, CAPTION_MAX_SIZE)
    if small < size:
        small_width = sum(hero["advances"]) * small
        top = max(_SAFE_TOP.get(), head[1] - small - 140 - 24)
        choices.append(("behind", [(WIDTH - small_width) / 2, top,
                                   (WIDTH + small_width) / 2, top + small]))
        for x in (head[0] - small_width - 24, head[2] + 24):
            if SAFE[0] <= x and x + small_width <= SAFE[2]:
                choices.append(("beside", [x, max(_SAFE_TOP.get(), head[1] + 80),
                                           x + small_width,
                                           max(_SAFE_TOP.get(), head[1] + 80) + small]))
    if obstacles and not behind_only:
        small = min(size, CAPTION_MAX_SIZE)
        small_width = sum(hero["advances"]) * small
        choices += [("below", [(WIDTH - small_width) / 2, y,
                                (WIDTH + small_width) / 2, y + small])
                    for y in range(SAFE[1], SAFE[3] - math.ceil(small) + 1, 24)]
    candidates = []
    landing = camera[min(len(camera) - 1, hero["on_frame"] + 12)]
    for placement, box in choices:
        candidate = dict(hero)
        candidate["placement"] = placement
        candidate["box"] = [round(value, 2) for value in box]
        candidate["size"] = round(box[3] - box[1], 4)
        candidate["size_fallback"] = candidate["size"] < size
        x, baseline = _unproject(
            landing, box[0], box[1] + candidate["size"] * _metrics()["base"]["hero"],
            PLATE_DEPTH,
        )
        candidate["ref_x"] = round(x, 5)
        candidate["ref_y"] = round(baseline, 5)
        candidate["ref_size"] = round(
            candidate["size"] / (PLATE_DEPTH / (PLATE_DEPTH - landing["cz"])), 5
        )
        candidate["min_visible_share"] = 1.0
        if behind_only:
            from yap_studio.render.camera3d_geometry import painted_box

            top = min(painted_box(candidate, camera, f, hero=True)[1]
                      for f in range(hero["on_frame"], hero["off_frame"]))
            if top < _SAFE_TOP.get():
                window = camera[max(0, hero["on_frame"] - 1):hero["off_frame"] + 1]
                scale = min(PLATE_DEPTH / (PLATE_DEPTH - sample["cz"])
                            for sample in window)
                shift = (_SAFE_TOP.get() - top + .01) / scale
                candidate["ref_y"] = round(candidate["ref_y"] + shift, 5)
                shift *= PLATE_DEPTH / (PLATE_DEPTH - landing["cz"])
                candidate["box"][1] += shift
                candidate["box"][3] += shift
        if any(_overlap(painted_box(candidate, camera, f, hero=True) if behind_only else
                        _hero_screen_box(candidate, camera[f], f), box)
               for f in range(hero["on_frame"], hero["off_frame"]) for box in obstacles):
            continue
        candidates.append(candidate)
    if not candidates:
        return None
    if matte is not None:
        assert matte_info is not None
        mattes = _window_frames(matte, matte_info, hero["on_frame"], hero["off_frame"], matte=True)
    else:
        synthetic = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        for y, left, right in section["rows"]:
            synthetic[y:min(HEIGHT, y + 16), left:right] = 255
        mattes = (synthetic for _ in range(hero["on_frame"], hero["off_frame"]))
    for frame, alpha in enumerate(mattes, start=hero["on_frame"]):
        for candidate in candidates:
            candidate["min_visible_share"] = min(
                candidate["min_visible_share"],
                _glyph_visible_share(candidate, camera[frame], frame, alpha),
            )
    for candidate in candidates:
        passes = candidate["min_visible_share"] >= MIN_BEHIND_VISIBLE_SHARE
        candidate["min_visible_share"] = round(candidate["min_visible_share"], 4)
        if passes:
            candidate["visible_fraction"] = candidate["min_visible_share"]
            return candidate
    if behind_only:
        return None
    front = dict(candidates[-1])
    front.update(placement="front", layer="front", depth=DEPTHS[0],
                 min_visible_share=1.0, visible_fraction=1.0)
    x, baseline = _unproject(
        landing, front["box"][0],
        front["box"][1] + front["size"] * _metrics()["base"]["hero"], front["depth"],
    )
    front["ref_x"], front["ref_y"] = round(x, 5), round(baseline, 5)
    front["ref_size"] = round(
        front["size"] / (front["depth"] / (front["depth"] - landing["cz"])), 5
    )
    if any(_overlap(_hero_screen_box(front, camera[f], f), box)
           for f in range(hero["on_frame"], hero["off_frame"]) for box in obstacles):
        return None
    return front


def _hero_boxes_by_frame(sentences: list[dict], camera: list[dict],
                         frames: int, *, painted: bool = False) -> list[list[list[float]]]:
    from yap_studio.render.camera3d_geometry import painted_box

    boxes: list[list[list[float]]] = [[] for _ in range(frames)]
    for sentence in sentences:
        hero = sentence["hero"]
        if hero is None:
            continue
        for frame in range(hero["on_frame"], min(frames, hero["off_frame"])):
            boxes[frame].append(painted_box(hero, camera, frame, hero=True) if painted else
                                _hero_screen_box(hero, camera[frame], frame))
    return boxes


def _hero_obstacles(boxes: list[list[list[float]]], on: int,
                    off: int) -> list[list[float]]:
    return [box for frame in boxes[max(0, on):min(len(boxes), off)] for box in frame]


def check_hero_caption_clearance(plan: dict) -> None:
    """Reject a caption pose that touches any visible hero at the same frame."""
    from yap_studio.render.camera3d_geometry import painted_box

    painted = "safe_text_top" in plan
    samples = plan["camera"]["samples"]
    boxes = _hero_boxes_by_frame(plan["sentences"], samples, plan["frames"], painted=painted)
    for group in plan["groups"]:
        for pose in group["poses"]:
            box = (painted_box(group, samples, pose["frame"], metrics=plan.get("font_metrics"))
                   if painted else pose["box"])
            if any(_overlap(box, hero, 24) for hero in boxes[pose["frame"]]):
                raise ValueError(
                    f"caption overlaps hero at frame {pose['frame']}: {group['text']}"
                )


def check_caption_clearance(plan: dict) -> None:
    """Reject overlapping caption boxes at any rendered frame."""
    from yap_studio.render.camera3d_geometry import painted_box

    for frame in range(plan["frames"]):
        boxes: list[list[float]] = []
        for group in plan["groups"]:
            if group["on_frame"] <= frame < group["off_frame"]:
                box = (painted_box(group, plan["camera"]["samples"], frame,
                                   metrics=plan.get("font_metrics"))
                       if "safe_text_top" in plan else
                       group["poses"][frame - group["on_frame"]]["box"])
                if any(_overlap(box, other, 24) for other in boxes):
                    raise ValueError(f"caption groups overlap at frame {frame}")
                boxes.append(box)


def _replaced_groups(groups: list[dict], onset: int) -> list[dict]:
    return [group for group in groups if group["on_frame"] < onset
            and group["fade_start_frame"] <= onset < group["off_frame"]]


def _end_replaced_groups(groups: list[dict], onset: int) -> None:
    for group in groups:
        group["fade_start_frame"] = max(group["on_frame"], onset - 6)
        group["off_frame"] = onset
        if "poses" in group:
            del group["poses"][onset - group["on_frame"]:]


def _unplace_group(group: dict, fallbacks: list[dict], reason: str) -> None:
    if group.get("unplaced"):
        return
    group["unplaced"] = True
    fallbacks.append({"sentence": group["sentence"], "word_ids": group["word_ids"],
                      "reason": reason, "action": "unplaced"})


def _head_at(plate: dict, camera: dict, head: list[int]) -> list[float]:
    corner = project_plate_corners(plate, camera, WIDTH, HEIGHT)[0]
    scale = plate["scale"] * plate["depth"] / (plate["depth"] - camera["cz"])
    left = corner[0] + (plate["pad_left"] + head[0]) * scale
    top = corner[1] + (plate["pad_top"] + head[1]) * scale
    return [left, top, left + (head[2] - head[0]) * scale,
            top + (head[3] - head[1]) * scale]


def _section_at(sections: list[dict], frame: int) -> dict:
    time_s = frame / FPS
    return next((section for section in sections
                 if section["start"] <= time_s < section["end"]), sections[-1])


def _stabilize_groups(groups: list[dict], sections: list[dict], plate: dict,
                      camera: list[dict], frames: int, fallbacks: list[dict],
                      hero_boxes: list[list[list[float]]], obstacles: list[dict] = (),
                      heads: list | None = None) -> None:
    """Keep depth text in clear portrait zones as the virtual camera moves."""
    base = _metrics()["base"]["caption"]
    for group in groups:
        group["poses"] = []
    prior: dict[int, list[float]] = {}
    for frame in range(frames):
        active = [(index, group) for index, group in enumerate(groups)
                  if not group.get("unplaced")
                  and group["on_frame"] <= frame < group["off_frame"]]
        occupied: list[list[float]] = []
        occupied_groups: list[dict] = []
        for index, group in active:
            cam = camera[frame]
            hero_obstacles = [*hero_boxes[frame], *(item["box"] for item in obstacles
                              if item["start_frame"] < group["off_frame"]
                              and item["end_frame"] > group["on_frame"])]
            section = _section_at(sections, frame)
            head = heads[frame] if heads is not None else _head_at(plate, cam, section["head_box"])
            margin = 0 if heads is not None else 24
            if group["flat"]:
                box = group["box"]
                conflicts = [_overlap(box, other, 24) for other in occupied]
                if any(conflicts):
                    replaced = _replaced_groups(occupied_groups, group["on_frame"])
                    remaining = [(prior_group, prior_box) for prior_group, prior_box
                                 in zip(occupied_groups, occupied, strict=True)
                                 if all(prior_group is not item for item in replaced)]
                    if replaced and all(not _overlap(box, other, 24)
                                        for _, other in remaining):
                        _end_replaced_groups(replaced, group["on_frame"])
                        occupied_groups = [item for item, _ in remaining]
                        occupied = [other for _, other in remaining]
                    else:
                        _unplace_group(group, fallbacks, f"flat box overlaps at frame {frame}")
                        continue
                if (_overlap(box, head, margin)
                        or any(_overlap(box, hero, 24) for hero in hero_obstacles)):
                    _unplace_group(group, fallbacks, f"flat box has no safe pose at frame {frame}")
                    continue
                occupied.append(box)
                occupied_groups.append(group)
                group["poses"].append({"frame": frame, "box": box, "ref_x": box[0],
                                       "ref_y": box[1] + base * group["size"], "scale": 1})
                continue
            x, baseline, scale = _project(cam, group["ref_x"], group["ref_y"],
                                          group["depth"])
            natural_size = group["ref_size"] * scale
            if heads is not None and min(natural_size, (SAFE[2] - SAFE[0]) / max(
                    .01, _advance(group["text"], "caption"))) < CAPTION_MIN_PAINTED_SIZE - .0001:
                _unplace_group(group, fallbacks,
                               f"depth size below {CAPTION_MIN_PAINTED_SIZE:g} px at frame {frame}")
                continue
            minimum = 44 if group["size"] < 60 else 60
            size = max(minimum, min(natural_size, 84,
                       (SAFE[2] - SAFE[0]) / max(0.01, _advance(group["text"], "caption"))))
            width = _advance(group["text"], "caption") * size
            ideal = [x, baseline - base * size]
            previous = prior.get(index)
            xs = [ideal[0], group["box"][0], (WIDTH - width) / 2,
                  SAFE[0], SAFE[2] - width]
            ys = [ideal[1], group["box"][1]]
            if previous is not None:
                xs.insert(0, previous[0])
                ys.insert(0, previous[1])
            ys += [float(y) for y in range(_SAFE_TOP.get(), SAFE[3] - int(size) + 1, 24)]
            choices = []
            for cy in ys:
                cy = min(max(cy, _SAFE_TOP.get()), SAFE[3] - size)
                for cx in xs:
                    cx = min(max(cx, SAFE[0]), SAFE[2] - width)
                    box = [cx, cy, cx + width, cy + size]
                    if (width > SAFE[2] - SAFE[0] or size > SAFE[3] - _SAFE_TOP.get()
                            or (size < 64 and box[3] <= head[1])
                            or _overlap(box, head, margin)
                            or any(_overlap(box, hero, 24) for hero in hero_obstacles)
                            or any(_overlap(box, other, 24) for other in occupied)):
                        continue
                    distance = (cx - ideal[0]) ** 2 + (cy - ideal[1]) ** 2
                    if previous is not None:
                        distance += .7 * ((cx - previous[0]) ** 2 +
                                          (cy - previous[1]) ** 2)
                    choices.append((distance, cy, cx, box))
            if not choices:
                box = _flat_box(group["text"], section,
                                [{"box": other, "on_frame": frame, "off_frame": frame + 1}
                                 for other in occupied], frame, frame + 1,
                                _hero_obstacles(hero_boxes, group["on_frame"],
                                                group["off_frame"]), head, look=heads is not None)
                if box is None:
                    replaced = _replaced_groups(occupied_groups, group["on_frame"])
                    remaining = [(prior_group, prior_box) for prior_group, prior_box
                                 in zip(occupied_groups, occupied, strict=True)
                                 if all(prior_group is not item for item in replaced)]
                    if replaced:
                        box = _flat_box(
                            group["text"], section,
                            [{"box": other, "on_frame": frame, "off_frame": frame + 1}
                             for _, other in remaining], frame, frame + 1,
                            _hero_obstacles(hero_boxes, group["on_frame"],
                                            group["off_frame"]), head, look=heads is not None
                        )
                        if box is not None:
                            _end_replaced_groups(replaced, group["on_frame"])
                            occupied_groups = [item for item, _ in remaining]
                            occupied = [other for _, other in remaining]
                if box is None:
                    _unplace_group(group, fallbacks, f"no safe camera pose at frame {frame}")
                    continue
                size = box[3] - box[1]
                group.update({"flat": True, "box": box, "size": size, "ref_size": size,
                              "depth": DEPTHS[-1], "ref_x": box[0],
                              "ref_y": box[1] + base * size,
                              "tracking": round(min(0.0, (SAFE[2] - SAFE[0]
                                  - _advance(group["text"], "caption") * size)
                                  / max(1, len(group["text"]) - 1)), 5)})
                group["poses"] = [
                    {"frame": pose["frame"], "box": box, "ref_x": box[0],
                     "ref_y": box[1] + base * size, "scale": 1}
                    for pose in group["poses"]
                ]
                group["poses"].append({"frame": frame, "box": box, "ref_x": box[0],
                                       "ref_y": box[1] + base * size, "scale": 1})
                fallbacks.append({"sentence": group["sentence"],
                                  "word_ids": group["word_ids"],
                                  "reason": f"no safe camera pose at frame {frame}",
                                  "action": "flat torso"})
                occupied.append(box)
                occupied_groups.append(group)
                continue
            _, _, _, box = min(choices)
            occupied.append(box)
            occupied_groups.append(group)
            prior[index] = box
            ref_x, ref_y = _unproject(cam, box[0], box[1] + base * size,
                                      group["depth"])
            group["poses"].append({"frame": frame, "box": [round(v, 3) for v in box],
                                   "ref_x": round(ref_x, 5), "ref_y": round(ref_y, 5),
                                   "scale": round(size / natural_size, 5)})


def _resolve_caption_conflicts(groups: list[dict], fallbacks: list[dict],
                               frames: int) -> None:
    """End replaced fades or omit a group whose final poses collide."""
    for frame in range(frames):
        visible: list[tuple[dict, list[float]]] = []
        for group in groups:
            if (group.get("unplaced") or not group["on_frame"] <= frame < group["off_frame"]):
                continue
            box = group["poses"][frame - group["on_frame"]]["box"]
            conflicts = [prior for prior, other in visible if _overlap(box, other, 24)]
            if conflicts:
                replaceable = _replaced_groups(conflicts, group["on_frame"])
                if len(replaceable) == len(conflicts):
                    _end_replaced_groups(replaceable, group["on_frame"])
                    visible = [(prior, other) for prior, other in visible
                               if all(prior is not item for item in replaceable)]
                else:
                    _unplace_group(group, fallbacks, f"caption overlap at frame {frame}")
                    continue
            visible.append((group, box))


def _choose_box(
    text: str, size: int, silhouette: dict, occupied: list[dict], on: int, off: int,
    preferred: tuple[float, float] | None = None,
    *, preferred_only: bool = False,
    luma_probe: Callable[[list[float]], float] | None = None,
    hero_obstacles: list[list[float]] | None = None,
) -> list[float] | None:
    width = _advance(text, "caption") * size
    head = silhouette["head_box"]
    shoulder = silhouette["shoulder_y"]
    candidates = []
    if preferred is not None:
        candidates.append(list(preferred))
    if not preferred_only:
        if head[1] - _SAFE_TOP.get() >= size + 24:
            candidates.append([WIDTH / 2 - width / 2, max(_SAFE_TOP.get(), head[1] - size - 20)])
        if head[0] >= 260 + SAFE[0] and width <= head[0] - SAFE[0] - 24:
            candidates.append([SAFE[0], max(_SAFE_TOP.get(), head[1] + 80)])
        if WIDTH - head[2] >= 260 + SAFE[0] and width <= WIDTH - SAFE[0] - head[2] - 24:
            candidates.append([WIDTH - SAFE[0] - width, max(_SAFE_TOP.get(), head[1] + 80)])
        for y in range(max(shoulder + 30, 820), SAFE[3] - size + 1, 48):
            candidates.append([WIDTH / 2 - width / 2, y])
        if shoulder + 30 < SAFE[3] - size:
            candidates.append([WIDTH / 2 - width / 2, SAFE[3] - size])
    valid = []
    for index, (x, y) in enumerate(candidates):
        box = [round(x, 2), round(y, 2), round(x + width, 2), round(y + size, 2)]
        if (
            box[0] < SAFE[0]
            or box[2] > SAFE[2]
            or box[1] < _SAFE_TOP.get()
            or box[3] > SAFE[3]
            or (size < 64 and box[3] <= head[1])
            or _overlap(box, head)
            or any(_overlap(box, hero, 24) for hero in (hero_obstacles or ()))
        ):
            continue
        if any(
            on < prior["off_frame"] and prior["on_frame"] < off and _overlap(box, prior["box"], 24)
            for prior in occupied
        ):
            continue
        if luma_probe is None:
            return box
        valid.append((luma_probe(box) + (0.08 if index else 0), index, box))
    return min(valid)[2] if valid else None


def _flat_box(
    text: str, silhouette: dict, occupied: list[dict], on: int,
    off: int, hero_obstacles: list[list[float]] | None = None,
    head_obstacle: list[float] | None = None, *, look: bool = False,
) -> list[float] | None:
    """Place a last-resort caption in the safe torso band."""
    width = min(SAFE[2] - SAFE[0], _advance(text, "caption") * 44)
    top = min(SAFE[3] - 44, max(_SAFE_TOP.get(), silhouette["shoulder_y"] + 30))
    options = range(int(top), SAFE[3] - 43, 12)
    size = 44
    if look:
        from yap_studio.render.camera3d_fallback import band_positions, lowest_box

        preferred = lowest_box(text)
        width, size = preferred[2] - preferred[0], preferred[3] - preferred[1]
        blocked = [head_obstacle or silhouette["head_box"], *(hero_obstacles or ()),
                   *(prior["box"] for prior in occupied
                     if on < prior["off_frame"] and prior["on_frame"] < off)]
        options = band_positions(preferred, blocked)
    head = head_obstacle or silhouette["head_box"]
    def cost(y: float) -> tuple[int, int, float]:
        box = [(WIDTH - width) / 2, y, (WIDTH + width) / 2, y + size]
        hero_conflict = any(_overlap(box, hero, 24) for hero in (hero_obstacles or ()))
        conflicts = sum(
            on < prior["off_frame"] and prior["on_frame"] < off
            and _overlap(box, prior["box"], 24) for prior in occupied
        )
        return int(hero_conflict or _overlap(box, head, 24)), conflicts, y
    available = [y for y in options if cost(y)[:2] == (0, 0)]
    if not available:
        return None
    y = available[0]
    return [round((WIDTH - width) / 2, 2), float(y),
            round((WIDTH + width) / 2, 2), float(y + size)]


def _plan_hero(sentence: list[dict], section: dict, next_sentence: float,
               has_next: bool, camera: list[dict], frames: int,
               width: int, frequencies: Counter[str],
               forbidden: set[str], matte: Path | None,
               matte_info: ffmpeg.MediaInfo | None, *, onset_lead: float = .2,
               obstacles: list[dict] = (), hero_free: list[tuple[int, int]] = (),
) -> tuple[dict | None, int | None]:
    """Choose the visible hero and the one source word it replaces."""
    unavailable = set(forbidden)
    candidates = ([word for word in sentence if not re.search(r"\s", word["text"])]
                  if onset_lead != .2 else sentence)
    while hero_text := _hero_word(candidates, frequencies, unavailable):
        unavailable.add(_term_key(hero_text))
        hero, word_id = _plan_hero_candidate(
            hero_text, sentence, section, next_sentence, has_next, camera, frames, width,
            onset_lead=onset_lead,
        )
        if hero is not None:
            if any(start < hero["off_frame"] and end > hero["on_frame"]
                   for start, end in hero_free):
                continue
            boxes = [item["box"] for item in obstacles
                     if item["start_frame"] < hero["off_frame"]
                     and item["end_frame"] > hero["on_frame"]]
            placed = _place_hero(hero, section, camera, matte, matte_info, boxes,
                                 behind_only=onset_lead != .2)
            if placed is not None:
                return placed, word_id
    return None, None


def _plan_hero_candidate(hero_text: str, sentence: list[dict], section: dict,
                         next_sentence: float, has_next: bool, camera: list[dict],
                         frames: int, width: int, *,
                         onset_lead: float = .2) -> tuple[dict | None, int | None]:
    target_width = 0.88 * width
    size = target_width / _advance(hero_text, "hero")
    masthead = section["head_box"][1] - _SAFE_TOP.get() >= size
    zone = "masthead" if masthead else "eye_level"
    y = (section["head_box"][1] - size * 0.8 if masthead
         else max(_SAFE_TOP.get(), section["head_box"][1] - size * 0.18))
    box = [
        round((width - target_width) / 2, 2),
        round(y, 2),
        round((width + target_width) / 2, 2),
        round(y + size, 2),
    ]
    visible = _hero_visible(box, section)
    if box[3] > SAFE[3]:
        return None, None
    hero_word = next(
        word for word in sentence
        if re.sub(r"[^\w’'-]", "", word["text"]) == hero_text
    )
    on = _word_onset(hero_word, onset_lead)
    sentence_end = 2 * round(sentence[-1]["end"] * 15)
    next_on = (_word_onset({"start": next_sentence}, onset_lead) if has_next else frames)
    off = min(frames, sentence_end, next_on, on + 3 * FPS)
    if off - on < 6:
        return None, None
    landing = camera[min(frames, on + 12)]
    x, baseline = _unproject(
        landing, box[0], box[1] + size * _metrics()["base"]["hero"], PLATE_DEPTH
    )
    hero = {
        "text": _screen_word(hero_text),
        "words": [dict(hero_word)],
        "advances": [text_width_em(char, _ACTIVE_STYLE.get()["hook"]["font"], 400)
                     if _ACTIVE_STYLE.get() else _advance(char, "hero")
                     for char in _screen_word(hero_text)],
        "zone": zone,
        "layer": "behind",
        "depth": PLATE_DEPTH,
        "box": box,
        "visible_fraction": round(visible, 4),
        "on_frame": on,
        "fade_start_frame": off - 6,
        "off_frame": off,
        "size": round(size, 4),
        "ref_size": round(size / (PLATE_DEPTH / (PLATE_DEPTH - landing["cz"])), 5),
        "ref_x": round(x, 5),
        "ref_y": round(baseline, 5),
    }
    return hero, hero_word["_occurrence"]


def _plan_camera3d(
    cut_words: dict,
    duration: float,
    silhouette: dict | list[dict],
    *,
    luminance: np.ndarray | None = None,
    matte: Path | None = None,
    width: int = WIDTH,
    height: int = HEIGHT,
    fps: int = FPS,
    onset_lead: float = .2,
    max_words: int = 4,
    max_characters: int | None = None,
    obstacles: list[dict] = (),
    hero_free: list[tuple[int, int]] = (),
    skip_heroes: set[int] = frozenset(),
    display_words: list[dict] | None = None,
    caption_gaps: list[tuple[int, int]] = (),
    faces: FacesData | None = None,
    replacements: list[tuple[int, int]] = (),
) -> dict:
    """Plan grouped captions, heroes, and one frame sampled moving camera."""
    if (width, height, fps) != (WIDTH, HEIGHT, FPS):
        raise ValueError("portrait pilot requires 1080x1920 at 30 fps")
    if cut_words.get("time_base") != "output":
        raise ValueError("cut words must use output time")
    if luminance is not None and (luminance.ndim != 3 or
                                   luminance.shape[1:] != (LUMA_HEIGHT, LUMA_WIDTH) or
                                   len(luminance) == 0):
        raise ValueError("luminance needs 480 by 270 sampled cut frames")
    frames = round(duration * fps)
    matte_info = ffmpeg.probe(matte) if matte is not None else None
    if frames < 1 or abs(frames / fps - duration) > 0.02:
        raise ValueError("duration must match a positive 30 fps cut")
    source_words = (display_words if display_words is not None else sorted(
        cut_words["words"], key=lambda word: (word["start"], word["end"], word["id"])))
    words = [dict(word, _occurrence=index) for index, word in enumerate(source_words)]
    frequencies = Counter(_term_key(word["text"]) for word in words
                          if "'" not in word["text"] and "’" not in word["text"])
    sentences = _sentences(words)
    if not sentences:
        raise ValueError("cut has no words")
    sections = (
        silhouette
        if isinstance(silhouette, list)
        else [{"start": 0, "end": duration, **silhouette}]
    )
    camera, whips = _camera_samples(sentences, frames)
    plate = {
        "width": width + PAD_LEFT + PAD_RIGHT,
        "height": height + PAD_TOP + PAD_BOTTOM,
        "pad_left": PAD_LEFT,
        "pad_top": PAD_TOP,
        "depth": PLATE_DEPTH,
        "scale": BASE_PLATE_SCALE,
        "ref_u": PAD_LEFT + width / 2,
        "ref_v": PAD_TOP + height / 2,
        "ref_x": width / 2,
        "ref_y": height / 2,
    }
    padding = check_camera_coverage(plate, camera, whips, width, height)
    heads = None
    if onset_lead != .2:
        from yap_studio.render.camera3d_clearance import head_regions

        heads = head_regions(sections, plate, camera, frames, faces, _ACTIVE_STYLE.get(),
                             replacements=replacements)
    groups = []
    fallbacks = []
    past_cut_words = set()
    planned_sentences = []
    clause_lists = [_clauses(sentence, min(4, max_words), max_characters) for sentence in sentences]
    if caption_gaps:
        from yap_studio.render.camera3d_look import split_at_gaps

        clause_lists = [split_at_gaps(clauses, caption_gaps) for clauses in clause_lists]
    if len(sentences[-1]) > 1:
        clause_lists[-1] = [[word] for word in sentences[-1]]
    hero_choices = []
    hero_counts: Counter[str] = Counter()
    previous_hero_key: str | None = None
    for si, sentence in enumerate(sentences):
        section_index = next(
            (index for index, item in enumerate(sections)
             if item["start"] <= sentence[0]["start"] < item["end"]),
            len(sections) - 1,
        )
        next_sentence = sentences[si + 1][0]["start"] if si + 1 < len(sentences) else duration
        forbidden = {key for key, count in hero_counts.items() if count >= 2}
        if previous_hero_key is not None:
            forbidden.add(previous_hero_key)
        hero, hero_occurrence = (None, None) if si in skip_heroes else _plan_hero(
            sentence, sections[section_index], next_sentence, si + 1 < len(sentences),
            camera, frames, width, frequencies, forbidden, matte, matte_info,
            onset_lead=onset_lead, obstacles=obstacles, hero_free=hero_free,
        )
        if hero is not None and caption_gaps:
            from yap_studio.render.camera3d_look import clamp_gaps

            if not clamp_gaps(hero, caption_gaps, frames=frames if onset_lead != .2 else None):
                hero, hero_occurrence = None, None
        if si in skip_heroes:
            fallbacks.append({"sentence": si, "reason": "rendered hero ink below 90 percent",
                              "action": "hero removed"})
        if hero is not None and (hero["placement"] != "behind" or hero["size_fallback"]):
            fallbacks.append({"sentence": si, "reason": "hero ink below 90 percent",
                              "action": ("front" if hero["placement"] == "front" else
                                         "smaller" if hero["size_fallback"] else
                                         hero["placement"]), "word": hero["text"]})
        if hero is not None:
            previous_hero_key = _term_key(hero["text"])
            hero_counts[previous_hero_key] += 1
        hero_choices.append((section_index, hero, hero_occurrence))
    hero_boxes = _hero_boxes_by_frame(
        [{"hero": hero} for _, hero, _ in hero_choices], camera, frames, painted=onset_lead != .2
    )
    for si, (sentence, clauses) in enumerate(zip(sentences, clause_lists, strict=True)):
        section_index, hero, hero_occurrence = hero_choices[si]
        section = sections[section_index]
        sentence_groups = []
        stair_x = float(SAFE[0])
        stair_y = float(max(section["shoulder_y"] + 30, 820))
        next_sentence = sentences[si + 1][0]["start"] if si + 1 < len(sentences) else duration
        clauses = [
            visible for clause in clauses
            if (visible := [word for word in clause
                            if word["_occurrence"] != hero_occurrence])
        ]
        gi = 0
        while gi < len(clauses):
            clause = clauses[gi]
            on = _word_onset(clause[0], onset_lead)
            replacement = (
                _word_onset(clauses[gi + 1][0], onset_lead)
                if gi + 1 < len(clauses)
                else _word_onset({"start": next_sentence}, onset_lead)
            )
            if si == len(sentences) - 1 and gi == len(clauses) - 1:
                replacement = frames
            text = " ".join(_screen_word(word["text"]) for word in clause)
            finale = si == len(sentences) - 1 and len(clauses) > 1
            depth = min(2400, 1500 + gi * 300) if finale else DEPTHS[gi % len(DEPTHS)]
            nominal = (84, 72, 66, 60)[min(gi, 3)] if finale else SIZES[gi % len(SIZES)]
            size = min(nominal, max(60, int(880 / max(1, _advance(text, "caption")))))
            off = min(frames if onset_lead != .2 else frames + 6, replacement + 6)
            if caption_gaps:
                from yap_studio.render.camera3d_look import clamp_gaps

                window = {"on_frame": on, "off_frame": off, "fade_start_frame": replacement,
                          "words": clause}
                if not clamp_gaps(window, caption_gaps,
                                   frames=frames if onset_lead != .2 else None):
                    past_cut_words.update(word["_occurrence"] for word in clause)
                    fallbacks.append({"sentence": si, "word_ids": [w["id"] for w in clause],
                                      "reason": "no painter frame before cut end after caption gap",
                                      "action": "drop"})
                    gi += 1
                    continue
                on, off, replacement = (window[key] for key in (
                    "on_frame", "off_frame", "fade_start_frame"))
            hero_obstacles = _hero_obstacles(hero_boxes, on, off)
            hero_obstacles += [item["box"] for item in obstacles
                               if item["start_frame"] < off and item["end_frame"] > on]
            if finale:
                preferred = (stair_x, stair_y + gi * 44)
            elif gi % 2 == 0:
                preferred = (WIDTH / 2 - _advance(text, "caption") * size / 2,
                             max(_SAFE_TOP.get(), section["head_box"][1] - size - 20))
            else:
                preferred = (WIDTH / 2 - _advance(text, "caption") * size / 2,
                             max(section["shoulder_y"] + 30, 820))
            landing_frame = min(frames - 1, max(on, replacement - 1))
            luma_probe = None
            if luminance is not None:
                sample = luminance[min(len(luminance) - 1,
                                       round(landing_frame * LUMA_FPS / FPS))]
                luma_probe = partial(_box_luma, sample=sample, plate=plate,
                                     camera=camera[landing_frame])
            box = None
            if luma_probe is None:
                box = _choose_box(text, size, section, groups, on, off, preferred,
                                  preferred_only=True, hero_obstacles=hero_obstacles)
            if box is None:
                box = _choose_box(text, size, section, groups, on, off, preferred,
                                  luma_probe=luma_probe, hero_obstacles=hero_obstacles)
            if box is None:
                minimum = 44 if onset_lead == .2 else CAPTION_MIN_SIZE
                for smaller in dict.fromkeys([*range(size - 2, minimum - 1, -2), minimum]):
                    box = _choose_box(text, smaller, section, groups, on, off,
                                      luma_probe=luma_probe,
                                      hero_obstacles=hero_obstacles)
                    if box is not None:
                        fallbacks.append({"sentence": si, "word_ids": [w["id"] for w in clause],
                                          "reason": "planned size has no safe zone",
                                          "action": "smaller size", "size": smaller})
                        size = smaller
                        break
            if box is None and len(clause) > 1 and onset_lead == .2:
                middle = len(clause) // 2
                clauses[gi:gi + 1] = [clause[:middle], clause[middle:]]
                fallbacks.append({"sentence": si, "word_ids": [w["id"] for w in clause],
                                  "reason": f"whole group has no safe zone at {minimum} px",
                                  "action": "split"})
                continue
            flat = box is None
            unplaced = False
            tracking = 0.0
            if flat:
                if onset_lead != .2:
                    from yap_studio.render.camera3d_fallback import flat_size

                    size = flat_size(text)
                    _end_replaced_groups(_replaced_groups(groups, on), on)
                else:
                    size = 44
                depth = DEPTHS[-1]
                box = _flat_box(text, section, groups, on, off, hero_obstacles,
                                 look=onset_lead != .2)
                if box is None:
                    replaced = _replaced_groups(groups, on)
                    if replaced:
                        remaining = [group for group in groups
                                     if all(group is not item for item in replaced)]
                        box = _flat_box(text, section, remaining, on, off, hero_obstacles,
                                         look=onset_lead != .2)
                        if box is not None:
                            _end_replaced_groups(replaced, on)
                if box is None and onset_lead == .2:
                    fallbacks.append({"sentence": si, "word_ids": [w["id"] for w in clause],
                                      "reason": "no conflict-free torso box", "action": "unplaced"})
                    gi += 1
                    continue
                if box is None:
                    from yap_studio.render.camera3d_fallback import lowest_box

                    box = lowest_box(text)
                    unplaced = True
                tracking = min(0.0, (SAFE[2] - SAFE[0]
                                     - _advance(text, "caption") * size)
                               / max(1, len(text) - 1))
                fallbacks.append({"sentence": si, "word_ids": [w["id"] for w in clause],
                                  "reason": "no safe zone after shrinking and splitting",
                                  "action": "flat torso"})
            if finale:
                stair_x += _advance(text, "caption") * size + 64
            landing = camera[min(frames, max(on, replacement - 1))]
            x, y = _unproject(landing, box[0], box[1] + size * _metrics()["base"]["caption"], depth)
            group = {
                "sentence": si,
                "section": section_index,
                "text": text,
                "words": [dict(word) for word in clause],
                "word_ids": [source for word in clause
                             for source in word.get("source_ids", (word["id"],))],
                "on_frame": on,
                "fade_start_frame": replacement,
                "off_frame": off,
                "fade_steps": 3,
                "stair_index": gi if finale else None,
                "depth": depth,
                "size": size,
                "box": box,
                "ref_x": round(x, 5),
                "ref_y": round(y, 5),
                "ref_size": round(size / (depth / (depth - landing["cz"])), 5),
                "flat": flat,
                "tracking": round(tracking, 5),
                **({"unplaced": True} if unplaced else {}),
            }
            if flat and onset_lead != .2:
                group.update(depth=0, ref_x=box[0],
                             ref_y=box[1] + size * _metrics()["base"]["caption"], ref_size=size)
            groups.append(group)
            sentence_groups.append(len(groups) - 1)
            gi += 1
        planned_sentences.append(
            {
                "groups": sentence_groups,
                "hero": hero,
                "start": sentence[0]["start"],
                "end": sentence[-1]["end"],
            }
        )
    check_hero_intervals(planned_sentences, onset_lead)
    check_hero_repetition(planned_sentences)
    _stabilize_groups(groups, sections, plate, camera, frames, fallbacks, hero_boxes,
                      obstacles, heads)
    _resolve_caption_conflicts(groups, fallbacks, frames)
    if onset_lead != .2:
        from yap_studio.render.camera3d_clearance import reject_unsafe_groups
        from yap_studio.render.camera3d_fallback import recover_groups

        reject_unsafe_groups(groups, camera, heads, fallbacks, obstacles, hero_boxes,
                              top=_SAFE_TOP.get())
        recover_groups(groups, sections, plate, camera, fallbacks, hero_boxes, obstacles, heads)
    kept = [(old, group) for old, group in enumerate(groups) if not group.get("unplaced")]
    group_indices = {old: new for new, (old, _) in enumerate(kept)}
    groups = [group for _, group in kept]
    for sentence in planned_sentences:
        sentence["groups"] = [group_indices[index] for index in sentence["groups"]
                              if index in group_indices]
    if luminance is not None:
        _label_luminance(groups, luminance, plate, camera, frames)
    caption_color = (_ACTIVE_STYLE.get() or resolve_style("preset:clean_minimal").data)[
        "captions"]["text_color"]
    rgb = [int(caption_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    caption_luma = sum(weight * _linear_luma(value) for weight, value in
                       zip((.2126, .7152, .0722), rgb, strict=True))
    halo_color = "#101010" if caption_luma >= .35 else "#FFFFFF"
    halo_luma = _linear_luma(16 / 255) if halo_color == "#101010" else 1.0
    for group in groups:
        group["halo"] = {"color": halo_color, "contrast_ratio": round(
            (max(caption_luma, halo_luma) + .05) /
            (min(caption_luma, halo_luma) + .05), 2)}
    plan = {
        "schema": "yap-studio/camera3d-pilot@1",
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "render_duration": hyperframes.layer_duration_s(frames),
        "silhouette": sections[0],
        "silhouette_summary": [
            {key: item[key] for key in ("start", "end", "head_box", "shoulder_y", "body_box")}
            for item in sections
        ],
        "silhouettes": sections,
        "fallbacks": fallbacks,
        "plate": plate,
        "camera": {"samples": camera, "whips": whips, "padding": padding},
        "groups": groups,
        "sentences": planned_sentences,
        **({"safe_text_top": _SAFE_TOP.get()} if onset_lead != .2 else {}),
    }
    check_hero_caption_clearance(plan)
    check_caption_clearance(plan)
    if onset_lead != .2:
        from yap_studio.render.camera3d_look import check_display_words

        check_display_words(plan, [word for index, word in enumerate(source_words)
                                    if index not in past_cut_words])
    return plan


def plan_camera3d(
    cut_words: dict, duration: float, silhouette: dict | list[dict], *,
    luminance: np.ndarray | None = None, matte: Path | None = None,
    style: ResolvedStyle | None = None,
    word_fixes: Mapping[str, str] | None = None,
    width: int = WIDTH, height: int = HEIGHT, fps: int = FPS,
    onset_lead: float = .2, max_words: int = 4, max_characters: int | None = None,
    obstacles: list[dict] = (), hero_free: list[tuple[int, int]] = (),
    skip_heroes: set[int] = frozenset(),
    display_words: list[dict] | None = None,
    caption_gaps: list[tuple[int, int]] = (),
    faces: FacesData | None = None,
    replacements: list[tuple[int, int]] = (),
) -> dict:
    """Resolve the creator look before measuring and placing camera text."""
    selected = style or resolve_style("preset:clean_minimal")
    token = _ACTIVE_STYLE.set(selected.data)
    top_token = _SAFE_TOP.set(selected.data["internal"]["key_text_top_px"]
                              if onset_lead != .2 else SAFE[1])
    try:
        if onset_lead != .2:
            onset_lead = min(.09, onset_lead)
        source_words = display_words if display_words is not None else cut_words["words"]
        words = [SimpleNamespace(id=word.get("id", str(index)), text=word["text"],
                                 start=word["start"], end=word["end"])
                 for index, word in enumerate(source_words)]
        vocabulary = bleep.listed(selected.data["music"])
        display = ([PhraseWord(word.id, word.text, word.start, word.end,
                                tuple(source.get("source_ids", (word.id,))))
                    for word, source in zip(words, source_words, strict=True)]
                   if display_words is not None else bleep.display_words(words, word_fixes))
        occurrences = bleep.find_display(display, vocabulary)
        drawn = (bleep.mask_display_words(display, vocabulary)
                 if bleep.enabled(selected.data) else display)
        original = {(word.id, word.start): source
                    for word, source in zip(words, source_words, strict=True)}
        screen_words = [{**original[(word.id, word.start)], "text": word.text, "start": word.start,
                         "end": word.end, "source_ids": word.source_ids,
                         "_bleep": any(index in item.indices for item in occurrences)}
                        for index, word in enumerate(drawn)]
        plan = _plan_camera3d({**cut_words, "words": screen_words}, duration, silhouette,
                              luminance=luminance,
                              matte=matte, width=width, height=height, fps=fps,
                              onset_lead=onset_lead, max_words=max_words,
                              max_characters=max_characters, obstacles=obstacles,
                              hero_free=hero_free, skip_heroes=skip_heroes,
                              display_words=screen_words if display_words is not None else None,
                              caption_gaps=caption_gaps, faces=faces,
                              replacements=replacements)
        plan["style"] = {"source": selected.source, "name": selected.name,
                         "sha256": selected.sha256(), "data": selected.data}
        plan["font_metrics"] = _plan_font_metrics(plan)
        plan["metric_fallbacks"] = [
            font for font in (selected.data["captions"]["font"],
                              selected.data["hook"]["font"])
            if font not in bundled_metrics()
        ]
        if plan["metric_fallbacks"]:
            plan["metric_notes"] = {
                font: "conservative fallback from bundled font advances"
                for font in plan["metric_fallbacks"]
            }
        plan["bleeps"] = []
        if bleep.enabled(selected.data):
            visible = [group["text"] for group in plan["groups"]]
            visible.extend(sentence["hero"]["text"] for sentence in plan["sentences"]
                           if sentence["hero"])
            if any(bleep.mask_text(text, vocabulary) != text for text in visible):
                raise ValueError("Bleep check failed: mask the camera caption and hero text.")
            plan["bleeps"] = [
                {"word": " ".join(display[index].text for index in item.indices),
                 "start": max(0, item.start_s - bleep.PAD_S),
                 "end": item.end_s + bleep.PAD_S}
                for item in occurrences
            ]
        return plan
    finally:
        _SAFE_TOP.reset(top_token)
        _ACTIVE_STYLE.reset(token)


def _plan_font_metrics(plan: dict) -> dict:
    base = _metrics()["base"]
    result = {"base": base, "adv": {}}
    for kind, family, weight, texts in (
        ("caption", plan["style"]["data"]["captions"]["font"],
         _caption_weight(plan["style"]["data"]),
         [group["text"] for group in plan["groups"]]),
        ("hero", plan["style"]["data"]["hook"]["font"], 400,
         [sentence["hero"]["text"] for sentence in plan["sentences"]
          if sentence["hero"]]),
    ):
        chars = set("".join(texts))
        result["adv"][kind] = {char: text_width_em(char, family, weight)
                               for char in sorted(chars)}
        result["adv"][kind]["_max"] = 1.25
    return result


_PAINT_SCRIPT = r"""
(function () {
  "use strict";
  var plan = window.CAMERA3D_PLAN, C3 = window.CAM3D, T = window.CAM3D_TABLES;
  var $ = function (id) { return document.getElementById(id); };
  C3.init(window.CAM3D_METRICS, { W: plan.width, H: plan.height });
  var samples = plan.camera.samples;
  var camera = new C3.Camera(function (t) {
    var f = Math.max(0, Math.min(samples.length - 1, t * plan.fps));
    var i = Math.min(samples.length - 2, Math.floor(f)), u = f - i;
    var a = samples[i], b = samples[i + 1], c = {};
    ['cz', 'px', 'py', 'tx', 'ty'].forEach(function (key) {
      c[key] = a[key] + (b[key] - a[key]) * u;
    });
    return c;
  }, plan.plate.depth);
  var P = plan.plate;
  var plate = new C3.Plate({ el: $('plate'), clipEl: $('plateClip'),
    personEl: $('person'), personClipEl: $('personClip'), padLeft: P.pad_left,
    padTop: P.pad_top, refU: P.ref_u, refV: P.ref_v, refX: P.ref_x,
    refY: P.ref_y, S0: P.scale, D0: P.depth });
  function fade(item) {
    return function (t) {
      var n = Math.floor((t - item.fade_start_frame / plan.fps) * 15 + 1e-6);
      return n < 0 ? 1 : Math.max(0, 1 - (n + 1) / 3);
    };
  }
  function word(item, parent, cls, font) {
    var result = new C3.Word(parent, { text: item.text, cls: cls, font: font,
      size: item.ref_size, on: item.on_frame / plan.fps,
      off: item.off_frame / plan.fps, D0: item.depth,
      opacity: fade(item), sharpRest: cls === 'cap',
      shadow: cls === 'cap' && !item.flat ? [plan.style.data.captions.shadow_y_px,
        plan.style.data.captions.shadow_blur_px,
        Math.max(.25, plan.style.data.captions.shadow_opacity), 1500] : null,
      state: function (t) {
        var rise = cls === 'hero' ? 140 : 30;
        var step = (t - item.on_frame / plan.fps) * 15;
        var x = item.ref_x, y = item.ref_y, scale = 1;
        if (item.poses && item.poses.length) {
          var f = Math.max(0, Math.min(item.poses.length - 1,
            t * plan.fps - item.on_frame));
          var i = Math.min(item.poses.length - 2, Math.floor(f));
          if (i < 0) i = 0;
          var a = item.poses[i], b = item.poses[i + 1] || a, u = f - i;
          x = a.ref_x + (b.ref_x - a.ref_x) * u;
          y = a.ref_y + (b.ref_y - a.ref_y) * u;
          scale = a.scale + (b.scale - a.scale) * u;
        }
        return { x: x, y: y + (item.flat ? 0 : rise * C3.entry(T.ENTRY, step)),
          s: scale, screen: !!item.flat,
          ax: 0, ay: C3.baseOf(font) * item.ref_size };
      }
    });
    if (item.flat && item.tracking) result.pcs.forEach(function (row) {
      row.forEach(function (piece) { piece.style.letterSpacing = item.tracking + 'px'; });
    });
    if (item.halo) result.pcs.forEach(function (row) {
      row.forEach(function (piece) { piece.style.textShadow =
        '-2px 0 ' + item.halo.color + ',2px 0 ' + item.halo.color +
        ',0 -2px ' + item.halo.color + ',0 2px ' + item.halo.color +
        ',-1px -1px ' + item.halo.color + ',1px -1px ' + item.halo.color +
        ',-1px 1px ' + item.halo.color + ',1px 1px ' + item.halo.color; });
    });
    return result;
  }
  var specs = [], live = new Map();
  plan.sentences.forEach(function (sentence) {
    if (sentence.hero) specs.push({ item: sentence.hero,
      parent: $(sentence.hero.layer === 'front' ? 'front' : 'behind'),
      cls: 'hero', font: 'hero' });
  });
  plan.groups.forEach(function (group) {
    specs.push({ item: group,
      parent: $(group.layer === 'behind' ? 'behind' : 'front'),
      cls: 'cap', font: 'caption' });
  });
  function active(item, t) {
    var tp = Math.floor((t + 1e-6) * 15 + 1e-4) / 15;
    return tp >= item.on_frame / plan.fps - 1e-6 &&
      tp < item.off_frame / plan.fps - 1e-6;
  }
  function makeWord(index) {
    var spec = specs[index];
    var result = word(spec.item, spec.parent, spec.cls, spec.font);
    for (var next = index + 1; next < specs.length; next++) {
      var later = live.get(next);
      if (later && later.g.parentNode === spec.parent) {
        spec.parent.insertBefore(result.g, later.g);
        break;
      }
    }
    live.set(index, result);
    return result;
  }
  if (window.FINISH && document.getElementById('finish')) window.FINISH.init();
  function paint(t) {
    if (window.FINISH && document.getElementById('finish')) window.FINISH.seed(t);
    plate.paint(camera, t);
    specs.forEach(function (spec, index) {
      var item = live.get(index);
      if (!active(spec.item, t)) {
        if (item) { item.g.remove(); live.delete(index); }
        return;
      }
      if (!item) item = makeWord(index);
      item.paint(t, camera, function (word, c) {
        if (spec.item.flat || spec.item.bottom_band) return 0;
        var distance = word.o.D0 - c.cz;
        var blur = Math.min(1.2, 6.7 * Math.abs(distance - (P.depth - c.cz)) / distance);
        return spec.cls === 'cap' ? Math.min(.6, blur) : blur;
      });
      if (spec.cls === 'cap' && spec.item.flat) {
        var shadow = plan.style.data.captions;
        item.g.style.filter = 'drop-shadow(' + shadow.shadow_x_px + 'px ' +
          shadow.shadow_y_px + 'px ' + shadow.shadow_blur_px + 'px rgba(0,0,0,' +
          shadow.shadow_opacity + '))';
      }
    });
  }
  var clock = { t: 0 }, duration = plan.render_duration;
  var timeline = gsap.timeline({ paused: true });
  timeline.to(clock, { t: duration, duration: duration, ease: 'none',
    onUpdate: function () { paint(clock.t); } }, 0);
  paint(0);
  window.__mainPaint = paint;
  window.__timelines = window.__timelines || {};
  window.__timelines['main'] = timeline;
})();
"""


def _finish_defs() -> str:
    """Return the grade filters used by the optional per-frame finish pass."""
    return """<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>
<filter id="finish" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">
<feComponentTransfer in="SourceGraphic" result="crv">
<feFuncR class="crv" type="table"/><feFuncG class="crv" type="table"/>
<feFuncB class="crv" type="table"/></feComponentTransfer>
<feColorMatrix in="crv" type="saturate" values="1.12" result="sat"/>
<feTurbulence class="nz" type="fractalNoise" baseFrequency="0.9"
 numOctaves="2" seed="1" result="nz"/>
<feComposite in="nz" in2="sat" operator="arithmetic" k1="0" k2="0.09" k3="1" k4="-0.045"/>
</filter>
<filter id="finishT" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">
<feComponentTransfer in="SourceGraphic" result="crv">
<feFuncR class="crv" type="table"/><feFuncG class="crv" type="table"/>
<feFuncB class="crv" type="table"/></feComponentTransfer>
<feColorMatrix in="crv" type="saturate" values="1.12" result="sat"/>
<feTurbulence class="nz" type="fractalNoise" baseFrequency="0.9"
 numOctaves="2" seed="1" result="nz"/>
<feColorMatrix in="nz" type="matrix"
 values="1.8 0 0 0 -0.4  0 1.8 0 0 -0.4  0 0 1.8 0 -0.4  0 0 0 0 0.05" result="nz1"/>
<feComposite in="nz1" in2="sat" operator="atop"/>
</filter></defs></svg>"""


def build_composition(
    plan: dict,
    plate_video: Path,
    person_video: Path,
    out_dir: Path,
    *,
    grain: bool = True,
    behind_only: bool = False,
    mode: str = "full",
    gsap_source: Path | None = None,
) -> Path:
    """Write a self-contained local composition for the pinned renderer."""
    if behind_only:
        mode = "behind"
    if mode not in {"full", "picture", "front", "behind"}:
        raise ValueError("unknown camera composition mode")
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = out_dir / "assets"
    assets.mkdir(exist_ok=True)
    shutil.copyfile(plate_video, assets / "plate.mp4")
    shutil.copyfile(person_video, assets / "person.webm")
    for name in ("cam3d.js", "tables.js", "finish.js"):
        shutil.copyfile(TEMPLATE / name, assets / name)
    look = plan.get("style", {}).get("data") or resolve_style("preset:clean_minimal").data
    caption_font, hero_font = look["captions"]["font"], look["hook"]["font"]
    copy_fonts((caption_font, hero_font), assets / "fonts")
    source = gsap_source
    if source is None:
        source = hyperframes.gsap_file(hyperframes.ensure_installed())
    shutil.copyfile(source, out_dir / "gsap.min.js")
    metrics = plan.get("font_metrics", _metrics())
    (assets / "metrics.js").write_text(
        "window.CAM3D_METRICS = " + json.dumps(metrics, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    (assets / "plan.js").write_text(
        "window.CAMERA3D_PLAN = "
        + json.dumps(plan, ensure_ascii=False, allow_nan=False,
                     separators=(",", ":")).replace("<", "\\u003c")
        + ";\n",
        encoding="utf-8",
    )
    (assets / "paint.js").write_text(_PAINT_SCRIPT, encoding="utf-8")
    p = plan["plate"]
    defs = _finish_defs() if grain else ""
    finish_script = '<script src="assets/finish.js"></script>' if grain else ""
    scene_filter = "filter:url(#finish);" if grain else ""
    front_filter = "filter:url(#finishT);" if grain else ""
    layer_css = ""
    if mode in {"front", "behind"}:
        hidden = "behind" if mode == "front" else "front"
        layer_css = ("html,body,#root{background:transparent!important;}"
                     f"#plateClip,#personClip,#{hidden}{{display:none!important;}}")
    elif mode == "picture":
        layer_css = "#front{display:none!important;}"
    face_css = font_face_css((caption_font, hero_font), url_prefix="assets/fonts/")
    shadow_opacity = max(.25, float(look["captions"].get("shadow_opacity", .25)))
    shadow_blur = max(3, float(look["captions"].get("shadow_blur_px", 3)))
    shadow_x = look["captions"].get("shadow_x_px", 0)
    shadow_y = look["captions"].get("shadow_y_px", 0)
    caption_weight = _caption_weight(look)
    hero_color = look.get("brand", {}).get("accent_color") or look["hook"].get(
        "accent_color", look["captions"]["highlight_color"])
    html = f"""<!doctype html>
<html lang="en" data-composition-variables='[]'><head><meta charset="utf-8">
<title>Camera caption pilot</title>
<style>
{face_css}
html,body {{ margin:0; background:#000; overflow:hidden; }}
#root {{ position:relative; width:1080px; height:1920px; overflow:hidden; background:#000; }}
{layer_css}
.scene,.clipbox,.tl,.wg,.gh {{ position:absolute; left:0; top:0; width:1080px; height:1920px; }}
.scene {{ {scene_filter} }}
.clipbox {{ overflow:hidden; }}
.plate,.person {{ position:absolute; left:0; top:0; transform-origin:0 0; }}
.plate {{ width:{p["width"]}px; height:{p["height"]}px; }}
.person {{ width:1080px; height:1920px; }}
video {{ display:block; width:100%; height:100%; }}
#front {{ {front_filter} }}
.wg {{ isolation:isolate; }}
.gh {{ mix-blend-mode:plus-lighter; opacity:0.142857; }}
.pc {{ position:absolute; left:0; top:0; line-height:1; white-space:nowrap; transform-origin:0 0;
  font-kerning:none; font-feature-settings:'kern' 0; font-optical-sizing:none; }}
.cap .pc {{ color:{look["captions"]["text_color"]};
  font-family:{font_stack(caption_font)}; font-weight:{caption_weight};
  text-shadow:{shadow_x}px {shadow_y}px {shadow_blur}px rgba(0,0,0,{shadow_opacity}),
    0 0 3px #111; }}
.hero .pc {{ color:{hero_color}; font-family:{font_stack(hero_font)}; font-weight:400;
  text-shadow:0 2px 12px rgba(0,0,0,.8); }}
</style><script src="gsap.min.js"></script><script src="assets/cam3d.js"></script>
<script src="assets/tables.js"></script>{finish_script}<script src="assets/metrics.js"></script>
<script src="assets/plan.js"></script></head><body>
<div id="root" data-composition-id="main" data-start="0" data-width="1080"
 data-height="1920" data-duration="{plan["render_duration"]}" data-fps="30">
{defs}
<div class="scene"><div id="plateClip" class="clipbox">
<div id="plate" class="plate" data-layout-allow-overflow="">
<video src="assets/plate.mp4" data-start="0" data-duration="{plan["render_duration"]}"
 data-media-start="0" data-track-index="0" muted playsinline></video></div></div>
<div id="behind" class="tl" data-layout-allow-occlusion="" data-layout-allow-overflow=""></div>
<div id="personClip" class="clipbox"><div id="person" class="person" data-layout-allow-overflow="">
<video src="assets/person.webm" data-start="0" data-duration="{plan["render_duration"]}"
 data-media-start="0" data-track-index="1" muted playsinline></video></div></div></div>
<div id="front" class="tl" data-layout-allow-overflow=""></div></div>
<script src="assets/paint.js"></script></body></html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    hyperframes.preflight(out_dir, plan["frames"])
    return out_dir


def _check_media(path: Path, *, width: int, height: int, frames: int) -> None:
    info = ffmpeg.probe(path, count_frames=True)
    if (info.width, info.height, info.video_frames, info.fps) != (width, height, frames, FPS):
        raise ValueError(f"media clock or size mismatch: {path}")


def prepare_plate(
    cut_video: Path, out_dir: Path, *, progress: Callable[[str], None] | None = None,
    matte: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Reflect-pad a cut and combine its picture with the project's cleaned matte."""
    cut = ffmpeg.probe(cut_video, count_frames=True)
    if ((cut.width, cut.height, cut.fps) != (WIDTH, HEIGHT, FPS)
            or not cut.video_frames or cut.variable_frame_rate):
        raise ValueError("pilot requires a constant 1080x1920 30 fps cut")
    out_dir.mkdir(parents=True, exist_ok=True)
    plate = out_dir / "plate.mp4"
    person = out_dir / "person.webm"
    matte_dir = out_dir / "matte"
    gray = matte if matte is not None else matte_dir / "matte.mkv"
    record_file = out_dir / "prep-record.json"
    input_record = {
        "cut_sha256": sha256_file(cut_video),
        "matte_settings": {"backend": "auto", "quality": "accurate",
                           "cleanup": Cleanup().to_dict()},
        "prep_version": 2,
    }
    if matte is not None:
        input_record = {"cut_sha256": input_record["cut_sha256"],
                        "matte_sha256": sha256_file(matte), "prep_version": 3}
    previous = None
    if record_file.is_file():
        try:
            previous = json.loads(record_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = None
    elif matte is None and (matte_dir / "matte.json").is_file():
        # Adopt a completed prep made before the input record was introduced.
        try:
            legacy = json.loads((matte_dir / "matte.json").read_text(encoding="utf-8"))
            if (legacy["status"] == "measured"
                    and legacy["video"]["sha256"] == input_record["cut_sha256"]
                    and legacy["cleanup"] == input_record["matte_settings"]["cleanup"]
                    and legacy["backend"].get("quality", "accurate") == "accurate"):
                previous = input_record
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if previous == input_record and all(path.is_file() for path in (gray, plate, person)):
        try:
            _check_media(gray, width=WIDTH, height=HEIGHT, frames=cut.video_frames)
            _check_media(plate, width=WIDTH + PAD_LEFT + PAD_RIGHT,
                         height=HEIGHT + PAD_TOP + PAD_BOTTOM, frames=cut.video_frames)
            _check_media(person, width=WIDTH, height=HEIGHT, frames=cut.video_frames)
        except (YapError, OSError, ValueError):
            pass
        else:
            if not record_file.is_file():
                record_file.write_text(json.dumps(input_record, indent=2) + "\n",
                                       encoding="utf-8")
            if progress is not None:
                progress("prep reused")
            return plate, person, gray
    if matte is None:
        if progress is not None:
            progress("matte")
        result = make_matte(cut_video, matte_dir)
        if result.status != "measured":
            raise ValueError(result.reason or "person matte unavailable")
    _check_media(gray, width=WIDTH, height=HEIGHT, frames=cut.video_frames)
    graph = (
        f"[0:v]split=3[m][t][b];[t]crop=iw:{PAD_TOP}:0:0,vflip[top];"
        f"[b]crop=iw:{PAD_BOTTOM}:0:ih-{PAD_BOTTOM},vflip[bot];"
        "[top][m][bot]vstack=inputs=3,split=3[c][l][r];"
        f"[l]crop={PAD_LEFT}:ih:0:0,hflip[ll];"
        f"[r]crop={PAD_RIGHT}:ih:iw-{PAD_RIGHT}:0,hflip[rr];"
        "[ll][c][rr]hstack=inputs=3[v]"
    )
    if progress is not None:
        progress("plate")
    ffmpeg.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(cut_video),
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            str(plate),
        ],
        timeout_s=120 + cut.video_frames,
    )
    if progress is not None:
        progress("person video")
    ffmpeg.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(cut_video),
            "-i",
            str(gray),
            "-filter_complex",
            "[0:v]format=rgba[fg];[1:v]format=gray[a];"
            "[fg][a]alphamerge=ts_sync_mode=nearest,format=yuva420p[v]",
            "-map",
            "[v]",
            "-an",
            "-frames:v",
            str(cut.video_frames),
            "-c:v",
            "libvpx-vp9",
            "-crf",
            "18",
            "-b:v",
            "0",
            "-row-mt",
            "1",
            "-deadline",
            "good",
            "-cpu-used",
            "4",
            "-auto-alt-ref",
            "0",
            "-pix_fmt",
            "yuva420p",
            str(person),
        ],
        timeout_s=120 + cut.video_frames * 2,
    )
    _check_media(cut_video, width=WIDTH, height=HEIGHT, frames=cut.video_frames)
    _check_media(
        plate,
        width=WIDTH + PAD_LEFT + PAD_RIGHT,
        height=HEIGHT + PAD_TOP + PAD_BOTTOM,
        frames=cut.video_frames,
    )
    _check_media(person, width=WIDTH, height=HEIGHT, frames=cut.video_frames)
    record_file.write_text(json.dumps(input_record, indent=2) + "\n", encoding="utf-8")
    return plate, person, gray


def summarize_matte(gray_video: Path, duration: float) -> list[dict]:
    """Sample one cleaned mask per second without holding the video in memory."""
    from contextlib import closing

    sections = []
    stride = WIDTH * HEIGHT
    pending = bytearray()
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(gray_video),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        "fps=1,format=gray",
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    with closing(ffmpeg.run_stream(command, timeout_s=120 + duration * 2)) as stream:
        for chunk in stream:
            pending.extend(chunk)
            while len(pending) >= stride:
                index = len(sections)
                alpha = np.frombuffer(bytes(pending[:stride]), dtype=np.uint8).reshape(
                    HEIGHT, WIDTH
                )
                del pending[:stride]
                summary = measure_silhouette(alpha)
                sections.append({"start": float(index), "end": float(index + 1), **summary})
    if pending or not sections:
        raise ValueError("could not sample complete person matte frames")
    sections[-1]["end"] = max(duration, sections[-1]["end"])
    return sections


def sample_cut_luminance(cut_video: Path, duration: float) -> np.ndarray:
    """Decode two small luma samples per second from the cut picture."""
    from contextlib import closing

    stride = LUMA_WIDTH * LUMA_HEIGHT
    pending = bytearray()
    samples = []
    sampling = f"fps={LUMA_FPS}" if duration >= .5 / LUMA_FPS else "select=eq(n\\,0)"
    command = [
        "ffmpeg", "-v", "error", "-i", str(cut_video), "-map", "0:v:0", "-an",
        "-vf", f"{sampling},scale={LUMA_WIDTH}:{LUMA_HEIGHT},format=gray",
        "-fps_mode", "passthrough",
        "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1",
    ]
    with closing(ffmpeg.run_stream(command, timeout_s=120 + duration * 2)) as stream:
        for chunk in stream:
            pending.extend(chunk)
            while len(pending) >= stride:
                samples.append(np.frombuffer(bytes(pending[:stride]), dtype=np.uint8)
                               .reshape(LUMA_HEIGHT, LUMA_WIDTH))
                del pending[:stride]
    if pending or not samples:
        raise ValueError("could not sample complete cut picture frames")
    return np.stack(samples)


def render_pilot(
    composition: Path, cut_video: Path, out: Path, frames: int,
    *, workers: int = hyperframes.RENDER_WORKERS,
    bleeps: list[dict] | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Render an opaque full frame, copy the cut audio, and verify the final clock."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if progress is not None:
        progress("render")
    hyperframes.preflight(composition, frames)
    renderer = hyperframes.ensure_installed()
    picture = composition.parent / "camera3d-picture.mp4"
    retry = hyperframes.render_with_retry(partial(
        hyperframes._run_renderer,
        renderer.executable,
        [
            "render",
            str(composition.absolute()),
            "--format",
            "mp4",
            "--no-best-effort",
            "--quiet",
            "--fps",
            str(FPS),
            "--workers",
            str(workers),
            "--experimental-fast-capture=false",
            "--player-ready-timeout",
            "240000",
            "-o",
            str(picture.absolute()),
        ],
        timeout_s=180 + frames * WIDTH * HEIGHT / 1e6,
    ))
    if retry is not None:
        print(f"Renderer retried once: {retry['detail']}", file=sys.stderr)
    _check_media(picture, width=WIDTH, height=HEIGHT, frames=frames)
    cut = ffmpeg.probe(cut_video, count_frames=True)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(picture),
        "-i",
        str(cut_video),
        "-map",
        "0:v:0",
    ]
    if cut.has_audio:
        if bleeps:
            bleep_file = composition.parent / "camera3d-bleep.wav"
            write_bleep_audio(cut_video, bleep_file, bleeps, cut)
            command[command.index(str(cut_video))] = str(bleep_file)
            command += ["-map", "1:a:0", "-c:a", "alac"]
        else:
            command += ["-map", "1:a:0", "-c:a", "copy"]
    command += ["-c:v", "copy", "-movflags", "+faststart", str(out)]
    if progress is not None:
        progress("mux")
    ffmpeg.run(command, timeout_s=120 + frames / FPS)
    final = ffmpeg.probe(out, count_frames=True)
    _check_media(out, width=WIDTH, height=HEIGHT, frames=frames)
    expected_codec = "alac" if bleeps else cut.audio_codec
    if cut.has_audio and (not final.has_audio or final.audio_codec != expected_codec):
        raise ValueError("final audio does not match the cut audio")
    return out


def bleep_pcm(samples: np.ndarray, sample_rate: int,
              spans: list[dict]) -> np.ndarray:
    """Replace only the named sample ranges with a speech-level 1 kHz tone."""
    edited = samples.copy()
    count = len(edited)
    for span in spans:
        first = max(0, min(count, round(float(span["start"]) * sample_rate)))
        last = max(first, min(count, round(float(span["end"]) * sample_rate)))
        if last == first:
            continue
        speech = samples[first:last].astype(np.float64)
        rms = float(np.sqrt(np.mean((speech / 32768) ** 2)))
        if rms < .005:
            edge = round(.15 * sample_rate)
            context = np.concatenate((samples[max(0, first - edge):first],
                                      samples[last:min(count, last + edge)]))
            if len(context):
                rms = float(np.sqrt(np.mean((context.astype(np.float64) / 32768) ** 2)))
        amplitude = min(.95, max(.005, rms * math.sqrt(2)))
        clock = np.arange(last - first) / sample_rate
        tone = np.sin(2 * math.pi * 1000 * clock) * amplitude * 32767
        replacement = np.broadcast_to(tone[:, None], speech.shape).copy()
        fade = min(round(.01 * sample_rate), (last - first) // 2)
        if fade:
            up = np.linspace(0, 1, fade, endpoint=False)[:, None]
            down = np.linspace(1, 0, fade, endpoint=False)[:, None]
            replacement[:fade] = speech[:fade] * (1 - up) + replacement[:fade] * up
            replacement[-fade:] = speech[-fade:] * (1 - down) + replacement[-fade:] * down
        edited[first:last] = np.rint(np.clip(replacement, -32768, 32767)).astype(np.int16)
    return edited


def write_bleep_audio(source: Path, out: Path, spans: list[dict],
                      info: ffmpeg.MediaInfo) -> Path:
    """Decode once to PCM, edit named samples, and store lossless audio for mux."""
    if not info.sample_rate or not info.channels:
        raise ValueError("bleep needs a known sample rate and channel count")
    raw = ffmpeg.run([
        "ffmpeg", "-v", "error", "-i", str(source), "-map", "0:a:0",
        "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(info.sample_rate),
        "-ac", str(info.channels), "pipe:1",
    ], timeout_s=120, binary=True).stdout
    samples = np.frombuffer(raw, dtype="<i2").reshape(-1, info.channels)
    edited = bleep_pcm(samples, info.sample_rate, spans)
    with wave.open(str(out), "wb") as writer:
        writer.setnchannels(info.channels)
        writer.setsampwidth(2)
        writer.setframerate(info.sample_rate)
        writer.writeframes(edited.tobytes())
    return out


def caption_contrast(frame: np.ndarray, box: list[float], color: str) -> float:
    """Measure rendered text against its immediate halo and surroundings."""
    x0, y0, x1, y1 = (int(box[0]), int(box[1]),
                      math.ceil(box[2]), math.ceil(box[3]))
    x0, y0 = max(0, x0 - 7), max(0, y0 - 7)
    x1, y1 = min(frame.shape[1], x1 + 7), min(frame.shape[0], y1 + 7)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    rgb = frame[y0:y1, x0:x1, :3].astype(np.float32)
    target = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)])
    ink = np.max(np.abs(rgb - target), axis=2) <= 45
    if np.count_nonzero(ink) < 10:
        return 0.0
    padded = np.pad(ink, 5)
    dilated = np.zeros_like(ink)
    for dy in range(11):
        for dx in range(11):
            dilated |= padded[dy:dy + len(ink), dx:dx + ink.shape[1]]
    surround = dilated & ~ink
    if np.count_nonzero(surround) < 10:
        return 0.0
    linear = rgb / 255
    linear = np.where(linear <= .04045, linear / 12.92,
                      ((linear + .055) / 1.055) ** 2.4)
    luma = linear @ np.array([.2126, .7152, .0722])
    foreground = float(np.median(luma[ink]))
    background = float(np.median(luma[surround]))
    return round((max(foreground, background) + .05) /
                 (min(foreground, background) + .05), 3)


def check_pilot_render(plan: dict, picture: Path, matte: Path, plate: Path,
                       person: Path, folder: Path, *, grain: bool = True,
                       workers: int = hyperframes.RENDER_WORKERS) -> list[Finding]:
    """Check rendered behind ink and each caption at the middle of its rest."""
    from yap_studio.render import alpha as render_alpha

    findings: list[Finding] = []
    behind = [sentence["hero"] for sentence in plan["sentences"]
              if sentence["hero"] and sentence["hero"]["layer"] == "behind"
              and sentence["hero"].get("contains_spoken_text", True)]
    behind.extend(group for group in plan["groups"] if group.get("layer") == "behind"
                  and group.get("contains_spoken_text", True))
    if behind:
        comp = build_composition(plan, plate, person, folder / "behind-composition",
                                 grain=grain, behind_only=True)
        renderer = hyperframes.ensure_installed()
        raw, layer = folder / "behind-raw.mov", folder / "behind.mov"
        retry = hyperframes.render_with_retry(partial(
            hyperframes._run_renderer,
            renderer.executable,
            ["render", str(comp.absolute()), "--format", "mov", "--no-best-effort",
             "--quiet", "--fps", str(FPS), "--workers", str(workers),
             "--experimental-fast-capture=false", "--player-ready-timeout", "240000",
             "-o", str(raw.absolute())],
            timeout_s=180 + plan["frames"] * WIDTH * HEIGHT / 1e6,
        ))
        if retry is not None:
            print(f"Renderer retried once: {retry['detail']}", file=sys.stderr)
        render_alpha.convert_to_bt709(raw, layer, frames=plan["frames"])
        raw.unlink(missing_ok=True)
        info = ffmpeg.probe(picture, count_frames=True)
        specs = [{"id": f"{hero['text']}@{hero['on_frame']}",
                  "start_frame": hero["on_frame"],
                  "layer_start_frame": hero["on_frame"],
                  "frames": hero["off_frame"] - hero["on_frame"],
                  "x": 0, "y": 0, "width": WIDTH, "height": HEIGHT,
                  "path": layer.name} for hero in behind]
        records = measure_behind_layers(
            folder, matte, info, specs,
            hidden_glyph_share=round(1 - MIN_BEHIND_VISIBLE_SHARE, 2),
            hidden_time_share=0, require_ink=True, report_worst=True,
            covered_pixels=partial(_render_person_covered,
                                   camera=plan["camera"]["samples"]),
        )
        for hero, record in zip(behind, records, strict=True):
            worst = record["worst_visible_share"]
            print(f"Behind {hero['text']} {hero['on_frame'] / FPS:.2f}s: "
                  f"worst visible {worst:.3f}", flush=True)
            findings.append(Finding(
                "captions3d.behind_words", "fail" if record["flag"] else "info",
                f"{hero['text']}: worst visible share {worst:.3f}",
                time_s=hero["on_frame"] / FPS, evidence=record,
            ))
    probes: dict[int, list[dict]] = {}
    for group in plan["groups"]:
        rest_end = min(group["fade_start_frame"], group["off_frame"])
        frame = max(group["on_frame"], min(rest_end - 1,
                    (group["on_frame"] + rest_end) // 2))
        probes.setdefault(frame, []).append(group)
    if probes:
        info = ffmpeg.probe(picture, count_frames=True)
        color = plan["style"]["data"]["captions"]["text_color"]
        for frame, pixels in enumerate(_window_frames(picture, info, 0,
                                                       max(probes) + 1, rgba=True)):
            for group in probes.get(frame, ()):
                pose = group["poses"][frame - group["on_frame"]]
                ratio = caption_contrast(pixels, pose["box"], color)
                findings.append(Finding(
                    "captions3d.contrast", "fail" if ratio < 4.5 else "info",
                    f"{group['text']}: contrast {ratio:.2f}:1",
                    time_s=frame / FPS, evidence={"ratio": ratio, "frame": frame,
                                                   "text": group["text"]},
                ))
    return findings
