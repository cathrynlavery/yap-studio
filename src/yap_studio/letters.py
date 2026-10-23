"""Fit spoken words, render alpha plates, and open them onto picture."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from yap_studio import ffmpeg, paths, receipt
from yap_studio.errors import ValidationError
from yap_studio.project.receipts import projection_sha256
from yap_studio.render import fonts, hyperframes
from yap_studio.render.text_metrics import text_width_em
from yap_studio.sfx import bleep

if TYPE_CHECKING:
    from yap_studio.project.model import CutWords
    from yap_studio.stages.captions import LayerAssets, LayerPlan
    from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline, TimelineFinding
    from yap_studio.style.loader import ResolvedStyle

FPS = 30  # All picture windows use the delivery frame grid.
WIDTH_PX = 1080  # Plates cover the whole vertical delivery frame.
HEIGHT_PX = 1920  # Plates cover the whole vertical delivery frame.
WORD_HOLD_S = 1.2  # Give the viewer time to read before opening the letters.
MIN_WORD_HOLD_S = 1.0  # Shorter holds do not leave enough reading time.
MAX_WORD_HOLD_S = 1.5  # Longer holds delay the supporting picture.
REVEAL_FRAMES = 12  # A short opening carries the viewer into the picture.
MIN_REVEAL_FRAMES = 10  # Keep the opening visible on the frame grid.
MAX_REVEAL_FRAMES = 14  # Keep the opening brisk.
SIDE_MARGIN_PX = 60  # Match the side margins of a default card.
TEXT_WIDTH_PX = WIDTH_PX - 2 * SIDE_MARGIN_PX  # Leave 960 px for the word block.
MAX_SIZE_PX = 640  # Keep short words and stacked lines inside the reading area.
MIN_SIZE_PX = 150  # Thinner strokes cannot read as windows onto a picture.
LINE_HEIGHT_EM = 0.9  # Stacked words share one tight line rhythm.
CENTER_Y_PX = 860  # A full two-line block ends above the default caption band.
CAPTION_TOP_PX = 1503  # Reserve the default caption band's top edge below the word block.
MAX_WORDS = 2  # Keep the word large enough for the picture inside it to read.
WEIGHT = 900  # Heavy strokes give the picture room to read.
BLUR_SIGMA_PX = 30  # Separate the sharp letters from the full-frame background.
BLUR_MULTIPLIER = 0.45  # Darken the background enough for the letters to read.
WORDS_MARGIN_FRAMES = 90  # Accept nearby spoken copy within three seconds each way.
OPENING_FRAME = 30  # Only inserts starting before one second are opening inserts.
MAX_MID_INSERTS = 2  # Extra emphasis loses its impact and earns a warning.
INK_ALPHA = 128  # Count half-covered raster pixels as ink for the zoom origin.
ORIGIN_DEPTH_SHARE = 0.6  # Choose a central stroke with substantial solid ink.
COVER_DEPTH_SHARE = 0.8  # Leave a safety margin around the final solid-ink coverage.
MIN_SCALE = 8  # Even a broad glyph needs a decisive opening.
MAX_SCALE = 400  # Bound the SVG transform for unusually thin fonts.
FADE_FRAMES = 2  # Finish with a completely transparent plate.
TEMPLATE = Path(__file__).parent / "templates/letters/template.html"


def phase_frames(options: Mapping[str, Any] | None = None) -> int:
    """Return the exact hold plus reveal duration on the delivery grid."""
    options = {} if options is None else options
    if not isinstance(options, Mapping):
        raise ValidationError("letters", "Set letters to an object with text and optional timing.")
    hold = options.get("word_hold_s", WORD_HOLD_S)
    reveal = options.get("reveal_frames", REVEAL_FRAMES)
    if (isinstance(hold, bool) or not isinstance(hold, (int, float))
            or not MIN_WORD_HOLD_S <= hold <= MAX_WORD_HOLD_S):
        raise ValidationError("letters.word_hold_s", f"Set word_hold_s to a number from "
                              f"{MIN_WORD_HOLD_S} to {MAX_WORD_HOLD_S}.")
    if (isinstance(reveal, bool) or not isinstance(reveal, (int, float))
            or not MIN_REVEAL_FRAMES <= reveal <= MAX_REVEAL_FRAMES or reveal != int(reveal)):
        raise ValidationError("letters.reveal_frames", f"Set reveal_frames to an integer from "
                              f"{MIN_REVEAL_FRAMES} to {MAX_REVEAL_FRAMES}.")
    return round(hold * FPS) + int(reveal)


def _word_spans(text: str) -> list[tuple[int, int]]:
    """Keep combining marks and internal apostrophes with their displayed word."""
    spans = []
    start = None
    for index, char in enumerate(text):
        if char.isalnum():
            if start is None:
                start = index
        elif start is not None and (
            unicodedata.category(char).startswith("M")
            or char in "'’" and index + 1 < len(text) and text[index + 1].isalnum()
        ):
            continue
        elif start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(text)))
    return spans


def _tokens(text: str) -> list[str]:
    # Decompose accents only after finding spans, then drop marks and apostrophes.
    return ["".join(char for char in unicodedata.normalize("NFD", text[start:end].casefold())
                    if char.isalnum()) for start, end in _word_spans(text)]


def fit(words: str, font: str) -> tuple[list[str], int]:
    """Choose the largest one-line or stacked layout with readable strokes."""
    displayed = " ".join(words.upper().split())
    spans = _word_spans(displayed)
    if len(spans) not in {1, MAX_WORDS}:
        raise ValidationError("letters.text", "Use one or two words from the spoken line.")
    candidates = [[displayed]]
    if len(spans) == MAX_WORDS:
        # Keep contractions together and leave joining punctuation on the first line.
        split_at = spans[1][0]
        candidates.append([displayed[:split_at].rstrip(), displayed[split_at:]])
    layouts = [(lines, min(MAX_SIZE_PX, math.floor(TEXT_WIDTH_PX / max(
        text_width_em(line, font, WEIGHT) for line in lines)))) for lines in candidates]
    lines, size = max(layouts, key=lambda layout: layout[1])
    if size < MIN_SIZE_PX:
        widest = max(lines, key=lambda line: text_width_em(line, font, WEIGHT))
        count = max(1, math.floor(TEXT_WIDTH_PX * len(widest) / (
            text_width_em(widest, font, WEIGHT) * MIN_SIZE_PX)))
        raise ValidationError("letters.text", f"Use a shorter word: this text needs {size} px, "
                              f"below the {MIN_SIZE_PX} px floor; about {count} letters fit.")
    return lines, size


def permission_message(start_frame: int) -> str:
    """Use the same concrete style correction in requests and timeline findings."""
    if start_frame < OPENING_FRAME:
        return ('Turn on hook.style: video_letters for the opening, or add video_letters '
                'to broll.emphasis_on; otherwise remove letters from this insert.')
    return ('Add video_letters to broll.emphasis_on for this key line, or remove letters '
            'from this insert.')


def allowed(style: Mapping[str, Any], start_frame: int) -> bool:
    """Keep both entry points off until the creator selects them."""
    return ("video_letters" in style.get("broll", {}).get("emphasis_on", []) or
            start_frame < OPENING_FRAME and
            style.get("hook", {}).get("style") == "video_letters")


def tokens_for(style: Mapping[str, Any], start_frame: int, *, title: bool = False) -> Mapping:
    """Use opening tokens at the start and graphics tokens for later inserts."""
    return style["hook" if title or start_frame < OPENING_FRAME else "graphics"]


def check(item: ResolvedElement, style: Mapping[str, Any], cut_words: CutWords, *,
          image: bool = False, mid_count: int = 1) -> list[TimelineFinding]:
    """Return fixable violations and the one advisory emphasis count finding."""
    from yap_studio.stages.timeline import TimelineFinding

    result = []

    def finding(rule: str, message: str, severity: str = "violation") -> None:
        result.append(TimelineFinding(rule, severity, f"{item.id}: {message}", (item.id,),
                                      (item.lane,), message, message))

    title = item.kind == "title"
    options = {} if title else item.payload["letters"]
    text = " ".join(item.payload["lines"]) if title else options["text"]
    text_field = "this title's lines" if title else "letters.text"
    if not title and not allowed(style, item.start_frame):
        finding("letters_off", permission_message(item.start_frame))
    words = _tokens(text)
    if len(words) not in {1, MAX_WORDS}:
        finding("letters_text", f"Change {text_field} to one or two words from the spoken line.")
    else:
        nearby = [word.text for word in cut_words.words
                  if word.end * FPS > item.start_frame - WORDS_MARGIN_FRAMES
                  and word.start * FPS < item.end_frame + WORDS_MARGIN_FRAMES]
        spoken = _tokens(" ".join(nearby))
        if not any(spoken[index:index + len(words)] == words for index in range(len(spoken))):
            fix = " ".join(nearby[:6]) or "the spoken words in this window"
            finding("letters_words", f"Change {text_field} to consecutive words from the "
                    f"spoken line, such as {fix}.")
        try:
            fit(text, tokens_for(style, item.start_frame, title=title)["font"])
        except ValidationError as exc:
            hint = ""
            for word in nearby:
                try:
                    fit(word, tokens_for(style, item.start_frame, title=title)["font"])
                except ValidationError:
                    continue
                hint = f' Try "{word}" from the spoken line.'
                break
            finding("letters_fit", (f"Change {text_field}: " if title else "") + exc.detail + hint)
    length = item.end_frame - item.start_frame
    phase = phase_frames(options)
    if title:
        minimum = round(MIN_WORD_HOLD_S * FPS) + MIN_REVEAL_FRAMES
        maximum = round(MAX_WORD_HOLD_S * FPS) + MAX_REVEAL_FRAMES
        if not minimum <= length <= maximum:
            finding("letters_length", f"Set this title to {minimum}..{maximum} frames, "
                    f"for example hold_s: {phase / FPS}.")
    elif length < phase or image and length != phase:
        finding("letters_length", f"Set this {'photo' if image else 'clip'} insert to "
                f"{'exactly' if image else 'at least'} {phase} frames; "
                f"use hold_s: {phase / FPS} and remove end.")
    if not title and item.start_frame >= OPENING_FRAME and mid_count > MAX_MID_INSERTS:
        finding("letters_count", f"Keep at most {MAX_MID_INSERTS} mid-video letters inserts; "
                "remove letters from an extra insert.", "warning")
    return result


def check_timeline(elements: Sequence[ResolvedElement], style: Mapping[str, Any],
                   cut_words: CutWords, images: Mapping[str, bool]) -> list[TimelineFinding]:
    """Check both entry points after media kinds and output windows are resolved."""
    inserts = [item for item in elements if item.kind == "insert" and "letters" in item.payload]
    count = sum(item.start_frame >= OPENING_FRAME for item in inserts)
    titles = [item for item in elements if item.kind == "title"
              and style["hook"]["style"] == "video_letters"]
    return [finding for item in [*inserts, *titles] for finding in check(
        item, style, cut_words, image=images.get(item.id, False), mid_count=count)]


def plate_plan(text: str, font: str, color: str, *, start_frame: int = 0,
               frames: int | None = None, options: Mapping[str, Any] | None = None,
               preview: bool = False) -> LayerPlan:
    """Plan a full-frame plate; gallery copy always has a renderable fallback."""
    from yap_studio.stages.captions import LayerPlan

    try:
        lines, size = fit(text, font)
    except ValidationError:
        if not preview:
            raise
        first = (text.upper().split() or ["LOOK"])[0]
        lines = [first]
        size = max(1, min(MAX_SIZE_PX, math.floor(TEXT_WIDTH_PX / text_width_em(
            first, font, WEIGHT))))
    options = options or {}
    reveal = options.get("reveal_frames", REVEAL_FRAMES)
    hold = round(options.get("word_hold_s", WORD_HOLD_S) * FPS)
    if frames is None:
        frames = hold + reveal
    elif not options:
        reveal = max(MIN_REVEAL_FRAMES, min(MAX_REVEAL_FRAMES, frames - hold))
        hold = frames - reveal
    return LayerPlan("letters", 0, 0, WIDTH_PX, HEIGHT_PX, start_frame, frames, {
        "lines": json.dumps(lines, ensure_ascii=False), "font_name": font,
        "font_family": fonts.font_stack(font), "size_px": size, "color": color,
        "center_y": CENTER_Y_PX, "hold_frames": hold, "reveal_frames": reveal,
        "fps": FPS, "look": "video_letters",
    })


def title_plan(plain: str, key: str, style: ResolvedStyle, *,
               lines: Sequence[str] | None, frames: int | None, start_frame: int) -> LayerPlan:
    """Use explicit title copy or choose a safe short gallery preview."""
    if lines is None:
        accent = key.split()
        text = " ".join(accent if len(accent) in {1, MAX_WORDS}
                        else (plain + " " + key).split()[:MAX_WORDS])
    else:
        text = " ".join(lines)
    text = _display_text(text, style.data)
    hook = style.data["hook"]
    return plate_plan(text, hook["font"], hook["accent_color"], start_frame=start_frame,
                      frames=frames, preview=lines is None)


def _display_text(text: str, style: Mapping[str, Any]) -> str:
    return bleep.mask_text(text, bleep.listed(style["music"])) if bleep.enabled(style) else text


def _bleep_input(style: Mapping[str, Any]) -> dict[str, Any]:
    return {"bleep": bleep.projection(style["music"])} if bleep.enabled(style) else {}


def _cached(folder: Path, key: str, frames: int) -> Path | None:
    if not folder.exists():
        return None
    movie = folder / "plate.mov"
    try:
        record = json.loads((folder / "receipt.json").read_bytes())
        valid = (record["schema"] == "yap-studio/letters-cache@1" and record["key"] == key
                 and record["sha256"] == receipt.sha256_file(movie))
        if not valid:
            raise ValueError("Cache identity changed")
        _assert_layer(movie, frames)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValidationError("letters.cache", f"Remove the incomplete cache {folder} "
                              "and render again.") from exc
    return paths.validate_input(movie)


def _assert_layer(movie: Path, frames: int) -> None:
    info = ffmpeg.probe(movie)
    if ((info.width, info.height, info.video_frames, info.fps) != (
        WIDTH_PX, HEIGHT_PX, frames, FPS)
            or (info.color_primaries, info.color_transfer, info.color_space) != ("bt709",) * 3
            or info.has_audio or not (info.pix_fmt or "").startswith("yuva444p")):
        raise ValidationError("letters.plate", "Render a silent BT.709 alpha plate at "
                              f"{WIDTH_PX}x{HEIGHT_PX}, {FPS} fps, with {frames} frames.")


def render_plates(project_dir: Path, timeline: ResolvedTimeline,
                  style: ResolvedStyle, work_dir: Path) -> list[dict[str, Any]]:
    """Publish content-keyed alpha movies atomically and verify every reuse."""
    from yap_studio.stages.captions import render_layer_plan

    items = [item for item in timeline.elements if item.kind == "insert"
             and "letters" in item.payload]
    if not items:
        return []
    with paths.operation():
        root = Path(project_dir).resolve()
        work = paths.validate_output_dir(work_dir)
        cache = paths.validate_output_dir(root / ".yap/cache/letters")
        placements = []
        for item in items:
            options = item.payload["letters"]
            tokens = tokens_for(style.data, item.start_frame)
            frames = phase_frames(options)
            text = _display_text(options["text"], style.data)
            plan = plate_plan(text, tokens["font"], tokens["accent_color"],
                              frames=frames, options=options)
            assets = plate_assets(tokens["font"])
            key = projection_sha256({
                "text": text, "variables": plan.variables, "frames": frames,
                **_bleep_input(style.data),
                "fonts": assets.font_identity(),
                "template": assets.template_identity(workers=hyperframes.RENDER_WORKERS),
            })
            folder = cache / key
            movie = _cached(folder, key, frames)
            if movie is None:
                with tempfile.TemporaryDirectory(prefix="letters-", dir=work) as temporary:
                    rendered = Path(temporary) / "plate.mov"
                    render_layer_plan(plan, Path(temporary), rendered, assets=assets)
                    _assert_layer(rendered, frames)
                    with tempfile.TemporaryDirectory(prefix=".letters-", dir=cache) as staged:
                        staging = Path(staged)
                        shutil.copyfile(rendered, staging / "plate.mov")
                        paths.atomic_write_text(staging / "receipt.json", json.dumps({
                            "schema": "yap-studio/letters-cache@1", "key": key,
                            "sha256": receipt.sha256_file(rendered),
                        }) + "\n")
                        try:
                            os.rename(staging, folder)
                        except OSError:
                            if not folder.is_dir():
                                raise
                        movie = _cached(folder, key, frames)
                        assert movie is not None
            placements.append({"id": item.id, "path": str(movie),
                               "start_frame": item.start_frame, "frames": frames,
                               **({"kind": item.kind, "display_text": json.loads(
                                   plan.variables["lines"])} if bleep.enabled(style.data) else {})})
        return placements


def plate_assets(font: str) -> LayerAssets:
    """Retain precisely the font and template bytes used by a plate."""
    from yap_studio.stages.captions import LayerAssets

    return LayerAssets(
        TEMPLATE.read_text(encoding="utf-8"), TEMPLATE.with_name("template.json").read_bytes(),
        fonts.font_face_css([font]),
        tuple((name, source.read_bytes()) for name, source in fonts.font_files([font])),
    )


def assembly_inputs(item: ResolvedElement, style: Mapping[str, Any]) -> dict[str, str]:
    """Invalidate assembly when plate assets or compositing tokens change."""
    tokens = tokens_for(style, item.start_frame)
    assets = plate_assets(tokens["font"])
    options = item.payload["letters"]
    plan = plate_plan(_display_text(options["text"], style), tokens["font"], tokens["accent_color"],
                      frames=phase_frames(options), options=options)
    return {"template:letters": receipt.sha256_file(TEMPLATE),
            f"letters:{item.id}": projection_sha256({
                "variables": plan.variables, "frames": plan.frames,
                **_bleep_input(style),
                "font": assets.font_identity(), "color": tokens["accent_color"],
                "template": assets.template_identity(workers=hyperframes.RENDER_WORKERS),
                "blur_sigma": BLUR_SIGMA_PX, "blur_multiplier": BLUR_MULTIPLIER,
            })}


def composite(graph: Any, fill_branch: str, plate_branch: str,
              item: ResolvedElement) -> str:
    """Apply the plate only during the letters phase, preserving the clip tail."""
    frames = phase_frames(item.payload["letters"])
    end = item.start_frame + frames
    if item.payload["letters"].get("background", "color") == "blur":
        sharp, soft = f"letters_sharp{len(graph.filters)}", f"letters_soft{len(graph.filters)}"
        graph.filters.append(f"[{fill_branch}]split=2[{sharp}][{soft}]")
        soft = graph.filter(soft, f"format=gbrp,gblur=sigma={BLUR_SIGMA_PX},"
                            f"lutrgb=r='val*{BLUR_MULTIPLIER}':"
                            f"g='val*{BLUR_MULTIPLIER}':b='val*{BLUR_MULTIPLIER}'")
        mask = graph.filter(plate_branch, "alphaextract")
        plate_branch = graph.filter(f"{soft}][{mask}", "alphamerge=shortest=1")
        fill_branch = sharp
    # This branch starts at the insert, so its frame count is local to the insert.
    return graph.overlay(fill_branch, plate_branch, item.start_frame, end,
                         enable=f"between(n,0,{frames - 1})")


def placements(timeline: ResolvedTimeline, layers: list[dict[str, Any]]) -> dict[str, dict]:
    """Refuse missing, duplicate, extra, or mistimed plate inputs before encoding."""
    items = {item.id: item for item in timeline.elements
             if item.kind == "insert" and "letters" in item.payload}
    if {layer["id"] for layer in layers} != set(items) or len(layers) != len(items):
        raise ValidationError("letters.layers", "Render letters plates for exactly the inserts "
                              "carrying letters, then assemble again.")
    for layer in layers:
        item = items[layer["id"]]
        if (layer["start_frame"] != item.start_frame
                or layer["frames"] != phase_frames(item.payload["letters"])
                or item.end_frame < item.start_frame + layer["frames"]):
            raise ValidationError("letters.layers", f"Render {item.id}'s exact letters window "
                                  "again and extend the insert to cover it.")
    return {layer["id"]: layer for layer in layers}
