"""Opt-in caption glyph geometry and frame-window checks against rendered alpha."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_captions_render import _phrases, _style

from yap_studio import ffmpeg, paths
from yap_studio.captions import CaptionWord
from yap_studio.render.text_metrics import text_width_em
from yap_studio.stages.captions import (
    LayerPlan,
    caption_layer_plan,
    caption_layout,
    caption_shrink,
    render_layer_plan,
)


def _full_alpha_frames(path: Path, plan: LayerPlan) -> np.ndarray:
    """Decode alpha in layer pixels without scaling or changing the frame cadence."""
    data = ffmpeg.run([
        "ffmpeg", "-nostdin", "-i", str(path), "-map", "0:v:0", "-an",
        "-vf", "alphaextract", "-pix_fmt", "gray", "-fps_mode", "passthrough",
        "-f", "rawvideo", "pipe:1",
    ], timeout_s=120 + plan.frames, binary=True).stdout
    assert len(data) == plan.frames * plan.height * plan.width
    return np.frombuffer(data, dtype=np.uint8).reshape(plan.frames, plan.height, plan.width)


def _pixel_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, columns = np.nonzero(mask)
    if not rows.size:
        return None
    return int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())


@pytest.mark.render
@pytest.mark.parametrize("highlight", ["none", "clip_wipe", "pill_karaoke"])
def test_wrapped_caption_has_no_orphan_row(
    tmp_path: Path, highlight: str,
) -> None:
    style = _style(highlight=highlight, font="Instrument Serif", size_px=83,
                   text_color="#FFFFFF", background_mode="none", italic_stress=True,
                   position="bottom", top_px=1503, max_characters_per_phrase=48)
    source = [CaptionWord(str(i), word, 0.2 + i * 0.2, 0.3 + i * 0.2)
              for i, word in enumerate("These thoughtful little moments matter".split())]
    phrase = replace(_phrases(style)[0], words=tuple(source), start=0.2, end=1.6,
                     stress_id="2")
    plan = caption_layer_plan([phrase], style, frames=60)
    slot = json.loads(plan.variables["phrases"])[0]
    assert len(slot["words"]) == 5
    with paths.operation():
        rendered = render_layer_plan(plan, tmp_path, tmp_path / f"{highlight}.mov")
    document = caption_layout(rendered.checks, plan)
    assert document["measured"] is True, document
    boxes = document["captions"][0]["boxes"]
    assert len(boxes) == 5
    rows: list[list[dict]] = []
    for box in boxes:
        row = next((row for row in rows if abs(row[0]["y"] - box["y"]) < 25), None)
        if row is None:
            row = []
            rows.append(row)
        row.append(box)
    assert len(rows) <= 2
    assert len(rows[-1]) >= 2
    for row in rows:
        assert min(box["x"] for box in row) >= 68
        assert max(box["x"] + box["w"] for box in row) <= 1012


@pytest.mark.render
def test_two_word_clip_wipe_wraps_without_shrinking(tmp_path: Path) -> None:
    style = _style(highlight="clip_wipe", font="Inter", size_px=83,
                   background_mode="none", italic_stress=False, letter_spacing_px=0,
                   width_scale=1)
    texts = ("Thoughtful", "Wonderful")
    widths = [text_width_em(text.upper(), "Inter", 800) * 83 for text in texts]
    assert max(widths) < 940 < sum(widths)
    phrases = _phrases(style)[:2]
    phrases[0] = replace(phrases[0], words=tuple(
        CaptionWord(str(i), text, 0.2 + i * 0.2, 0.4 + i * 0.2)
        for i, text in enumerate(texts)), stress_id=None)
    plan = caption_layer_plan(phrases, style, frames=60)
    with paths.operation():
        rendered = render_layer_plan(plan, tmp_path, tmp_path / "two-words.mov")
    document = caption_layout(rendered.checks, plan)
    assert document["measured"] is True, document
    first, second = document["captions"]
    assert len(first["boxes"]) == 2
    assert abs(first["boxes"][0]["y"] - first["boxes"][1]["y"]) > 50
    assert abs(second["boxes"][0]["y"] - second["boxes"][1]["y"]) < 25
    shrink = caption_shrink(rendered.checks, plan)
    assert shrink["measured"] is True, shrink
    assert shrink["style_size_px"] == 83
    assert shrink["phrases"] == []


@pytest.mark.render
@pytest.mark.parametrize("highlight", ["none", "slam", "clip_wipe"])
def test_real_caption_layout_matches_visible_frames_and_glyph_pixels(
    tmp_path: Path, highlight: str,
) -> None:
    fps = 30
    style = _style(highlight=highlight, size_px=48)
    phrases = []
    for i, (base, texts) in enumerate(((6, ("Go", "up")), (36, ("We", "do")))):
        start, end = (base + 0.5) / fps, (base + 19.5) / fps
        next_start = start + 0.3
        words = (
            CaptionWord(f"w{i * 2}", texts[0], start, start + 0.2),
            CaptionWord(f"w{i * 2 + 1}", texts[1], next_start, next_start + 0.2),
        )
        phrases.append(replace(_phrases(style)[i], words=words, start=start, end=end,
                               stress_id=None))
    plan = caption_layer_plan(phrases, style, frames=60, fps=fps)
    with paths.operation():
        result = render_layer_plan(plan, tmp_path, tmp_path / "captions.mov")
    document = caption_layout(result.checks, plan)
    assert document["measured"] is True, document
    assert len(document["captions"]) == 2, document
    alpha = _full_alpha_frames(result.path, plan)
    rows, columns = np.indices((plan.height, plan.width))
    grow = max(3, 0.06 * plan.variables["size_px"])
    for slot, phrase in enumerate(document["captions"]):
        boxes = [
            {**box, "x": box["x"] - plan.x, "y": box["y"] - plan.y}
            for box in phrase["boxes"]
        ]
        start = phrase["start_frame"] - plan.start_frame
        end = phrase["end_frame"] - plan.start_frame
        assert (start, end) == ((7, 26) if slot == 0 else (37, 56)), phrase
        for frame in range(start - 1, end + 1):
            visible = start <= frame < end
            bounds = _pixel_bounds(alpha[frame] > 0)
            assert (bounds is not None) == visible, (
                f"{highlight} {phrase['id']} frame={frame} visible={visible} "
                f"pixel_bounds={bounds} boxes={boxes}"
            )
        pixels = alpha[start:end].max(axis=0) >= 128
        assert pixels.any(), f"{highlight} {phrase['id']} has no opaque pixels; boxes={boxes}"
        union = np.zeros(pixels.shape, dtype=bool)
        for box in boxes:
            inside = ((columns >= box["x"]) & (columns <= box["x"] + box["w"])
                      & (rows >= box["y"]) & (rows <= box["y"] + box["h"]))
            assert (pixels & inside).any(), (
                f"{highlight} {phrase['id']} empty_box={box} "
                f"pixel_bounds={_pixel_bounds(pixels)} boxes={boxes}"
            )
            union |= ((columns >= box["x"] - grow) & (columns <= box["x"] + box["w"] + grow)
                      & (rows >= box["y"] - grow) & (rows <= box["y"] + box["h"] + grow))
        outside = pixels & ~union
        assert not outside.any(), (
            f"{highlight} {phrase['id']} offending_pixel_bounds={_pixel_bounds(outside)} "
            f"pixel_bounds={_pixel_bounds(pixels)} boxes={boxes} grow={grow}"
        )
