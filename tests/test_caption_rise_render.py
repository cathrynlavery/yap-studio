"""Opt-in pixel checks for caption Rise using synthetic phrase timings."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_captions_render import _ALL_HIGHLIGHTS, _pilot_plan, _rgba_frames, _style

from yap_studio import ffmpeg, paths
from yap_studio.captions import CaptionWord, build_phrases
from yap_studio.render import hyperframes, layer_checks
from yap_studio.stages.captions import (
    LayerPlan,
    caption_layer_plan,
    caption_layout,
    render_layer_plan,
)

pytestmark = pytest.mark.render


@pytest.fixture(scope="module")
def rise_movie(tmp_path_factory: pytest.TempPathFactory) -> Callable:
    cached: dict[tuple, tuple] = {}

    def render(entrance: str = "rise", *, highlight: str = "none", repeat: bool = False,
               one_word: bool = False, backing: str | None = None, **tokens: object) -> tuple:
        key = (entrance, highlight, one_word, backing, tuple(sorted(tokens.items())))
        if key not in cached or repeat:
            style = _style(highlight=highlight, entry_ms=0, entry_rise_px=0,
                           shadow_opacity=0, italic_stress=False, **tokens)
            style.data["captions"]["entrance"] = entrance
            plan = _pilot_plan(style)
            phrases = json.loads(plan.variables["phrases"])
            if one_word:
                style.data["captions"]["max_words_per_phrase"] = 1
                words = [CaptionWord(str(i), w["text"], 0.2 + i * 0.5, 0.4 + i * 0.5)
                         for i, w in enumerate(phrases[0]["words"][:2])]
                plan = caption_layer_plan(build_phrases(words, style.data["captions"]),
                                          style, frames=42)
            else:
                plan = replace(plan, frames=48)
            assert plan.variables["entrance"] == entrance
            if backing:
                phrases = json.loads(plan.variables["phrases"])
                for phrase in phrases:
                    phrase["backing"] = backing
                plan = replace(plan, variables={**plan.variables, "phrases": json.dumps(phrases)})
            work = tmp_path_factory.mktemp("rise-render")
            with paths.operation():
                result = render_layer_plan(plan, work, work / "captions.mov")
            movie = (plan, result.path, caption_layout(result.checks, plan))
            if repeat:
                return movie
            cached[key] = movie
        return cached[key]

    return render


def _rest_pixels(plan: LayerPlan, movie: Path, frame: int = 20) -> np.ndarray:
    """Compare lossless renderer pixels before alpha video encoding."""
    composition = movie.parent / "captions"
    output = movie.parent / f"snapshot-{frame}"
    if not output.exists():
        layer_checks._set_defaults(composition / "index.html", plan.variables)
        install = hyperframes.ensure_installed()
        hyperframes._run_renderer(install.executable, [
            "snapshot", str(composition), "--at", str(frame / plan.variables["fps"]),
            "--no-end", "--describe", "false", "--timeout", "180000", "-o", str(output),
            *(["--no-browser-gpu"] if hyperframes.CAPTION_BROWSER_GPU is False else []),
        ], timeout_s=240)
    images = list(output.glob("frame-*.png"))
    assert len(images) == 1
    data = ffmpeg.run([
        "ffmpeg", "-nostdin", "-i", str(images[0]), "-pix_fmt", "rgba",
        "-f", "rawvideo", "pipe:1",
    ], timeout_s=120, binary=True).stdout
    return np.frombuffer(data, dtype=np.uint8).reshape(plan.height, plan.width, 4)


@pytest.mark.parametrize("highlight", _ALL_HIGHLIGHTS)
def test_real_rise_phrase_staggers_and_matches_slide_at_rest(
    rise_movie: Callable, highlight: str,
) -> None:
    plan, out, layout = rise_movie(highlight=highlight)
    slide, baseline, slide_layout = rise_movie("slide", highlight=highlight)
    frames = _rgba_frames(out, plan)
    still = _rgba_frames(baseline, slide)
    assert not frames[6, ..., 3].any()
    assert frames[7, ..., 3].any()
    assert not np.array_equal(frames[7], still[7])
    # The fourth word finishes after 6 frames of stagger and 8 frames of motion.
    rest = 20 if highlight != "slam" else 41
    np.testing.assert_array_equal(_rest_pixels(plan, out, rest),
                                  _rest_pixels(slide, baseline, rest))
    assert layout["measured"] and slide_layout["measured"]
    rise_boxes = layout["captions"][0]["boxes"]
    slide_boxes = slide_layout["captions"][0]["boxes"]
    assert len(rise_boxes) == len(slide_boxes) == 4
    for actual, expected in zip(rise_boxes, slide_boxes, strict=True):
        assert actual["w"] <= expected["w"] + 0.02
        assert actual["h"] <= expected["h"] + 0.02
        assert actual["y"] + actual["h"] <= expected["y"] + expected["h"] + 0.02


def test_real_rise_never_draws_below_rest_line_box(rise_movie: Callable) -> None:
    plan, out, layout = rise_movie()
    frames = _rgba_frames(out, plan)
    boxes = layout["captions"][0]["boxes"]
    rest_bottom = max(b["y"] + b["h"] for b in boxes) - plan.y
    assert not frames[7:20, int(np.ceil(rest_bottom)):, :, 3].any()
    first = boxes[0]
    left, right = int(np.floor(first["x"])), int(np.ceil(first["x"] + first["w"]))
    partial_y = np.where(frames[7, :, left:right, 3] > 0)[0]
    rested_y = np.where(frames[20, :, left:right, 3] > 0)[0]
    assert partial_y.size and rested_y.size
    assert partial_y.min() > rested_y.min()
    assert partial_y.max() - partial_y.min() < rested_y.max() - rested_y.min()
    last = boxes[-1]
    assert not frames[7, :, int(last["x"]):int(last["x"] + last["w"]), 3].any()


def test_real_rise_one_word_phrases_enter_independently(rise_movie: Callable) -> None:
    plan, out, _ = rise_movie(one_word=True)
    frames = _rgba_frames(out, plan)
    phrases = json.loads(plan.variables["phrases"])
    assert len(phrases) == 2 and all(len(p["words"]) == 1 for p in phrases)
    for phrase in phrases:
        start = round(phrase["start"] * 30)
        assert not frames[start, ..., 3].any()
        assert frames[start + 1, ..., 3].any()
        assert frames[start + 8, ..., 3].sum() > frames[start + 1, ..., 3].sum()


def test_real_rise_shadow_halo_is_present_and_uncut_at_rest(rise_movie: Callable) -> None:
    plan, out, _ = rise_movie(backing="shadow")
    slide, baseline, _ = rise_movie("slide", backing="shadow")
    plain, plain_out, _ = rise_movie()
    frame = _rest_pixels(plan, out)
    np.testing.assert_array_equal(frame, _rest_pixels(slide, baseline))
    plain_frame = _rest_pixels(plain, plain_out)
    assert np.count_nonzero(frame[..., 3]) > np.count_nonzero(plain_frame[..., 3])
    assert not frame[[0, -1], :, 3].any()


@pytest.mark.parametrize("mode", ["box", "band"])
def test_real_rise_ground_fades_while_words_move(rise_movie: Callable, mode: str) -> None:
    plan, out, layout = rise_movie(background_mode=mode)
    slide, baseline, _ = rise_movie("slide", background_mode=mode)
    frames = _rgba_frames(out, plan)
    assert not frames[6, ..., 3].any()
    assert frames[7, ..., 3].any()
    assert frames[9, ..., 3].sum() > frames[7, ..., 3].sum()
    top = min(b["y"] for b in layout["captions"][0]["boxes"]) - plan.y
    ground_y, ground_x = int(top) - 6, plan.width // 2
    assert 0 < frames[7, ground_y, ground_x, 3] < frames[9, ground_y, ground_x, 3]
    assert frames[9, ground_y, ground_x, 3] == frames[20, ground_y, ground_x, 3]
    np.testing.assert_array_equal(_rest_pixels(plan, out), _rest_pixels(slide, baseline))


def test_real_rise_two_renders_have_identical_bytes(rise_movie: Callable) -> None:
    plan, first, _ = rise_movie()
    second_plan, second, _ = rise_movie(repeat=True)
    assert plan == second_plan
    assert first.read_bytes() == second.read_bytes()


def test_real_rise_wrapped_ground_keeps_weight_shift_pixels(rise_movie: Callable) -> None:
    tokens = {"highlight": "weight_shift", "size_px": 180,
              "width_scale": 0.8, "background_mode": "box"}
    plan, out, _ = rise_movie(**tokens)
    slide, baseline, _ = rise_movie("slide", **tokens)
    np.testing.assert_array_equal(_rest_pixels(plan, out), _rest_pixels(slide, baseline))
