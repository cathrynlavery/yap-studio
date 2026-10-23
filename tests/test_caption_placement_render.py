"""Opt-in pixel checks for phrase placement inside a taller caption layer."""

from pathlib import Path

import numpy as np
import pytest
from test_caption_layout_render import _full_alpha_frames
from test_captions_render import _phrases, _style

from yap_studio import paths
from yap_studio.stages.captions import caption_layer_plan, caption_layout, render_layer_plan


@pytest.mark.render
# Slam's contrasting outer guard (kept for legibility with shadows off) anti-aliases a few
# edge pixels differently in some runs when another phrase moves the layer's capture
# origin. Fixing that needs a fixed capture origin for every caption render, which is
# deferred to the owner.
@pytest.mark.parametrize("highlight", [
    "none",
    pytest.param("slam", marks=pytest.mark.xfail(
        reason="slam guard edge anti-aliasing can shift with the capture origin (about 8 px); "
        "fixed-origin capture is an owner decision",
        strict=False)),
    "weight_shift",
])
def test_real_phrase_placement_preserves_unmoved_ink_and_moves_second_up(
    tmp_path: Path, highlight: str,
) -> None:
    style = _style(highlight=highlight, shadow_opacity=0, italic_stress=False)
    phrases = _phrases(style)[:2]
    baseline = caption_layer_plan(phrases, style, frames=54)
    placed = caption_layer_plan(phrases, style, frames=54, offsets=[0, -300])
    with paths.operation():
        old = render_layer_plan(baseline, tmp_path / "baseline", tmp_path / "baseline.mov")
        new = render_layer_plan(placed, tmp_path / "placed", tmp_path / "placed.mov")
    old_layout = caption_layout(old.checks, baseline)
    new_layout = caption_layout(new.checks, placed)
    assert old_layout["measured"] and new_layout["measured"]
    old_alpha = _full_alpha_frames(old.path, baseline)
    new_alpha = _full_alpha_frames(new.path, placed)
    rows, columns = np.indices((placed.height, placed.width))
    grow = max(3, 0.06 * placed.variables["size_px"])
    for slot, (before, after) in enumerate(zip(old_layout["captions"], new_layout["captions"],
                                              strict=True)):
        assert (after["start_frame"], after["end_frame"]) == (
            before["start_frame"], before["end_frame"],
        )
        assert len(after["boxes"]) == len(before["boxes"])
        for a, b in zip(before["boxes"], after["boxes"], strict=True):
            assert b == {**a, "y": a["y"] - (300 if slot else 0)}
        mask = np.zeros((placed.height, placed.width), dtype=bool)
        for box in after["boxes"]:
            mask |= ((columns + placed.x >= box["x"] - grow) &
                     (columns + placed.x < box["x"] + box["w"] + grow) &
                     (rows + placed.y >= box["y"] - grow) &
                     (rows + placed.y < box["y"] + box["h"] + grow))
        # Slam shows one word at a time, so single frames inside the window may be empty.
        assert (new_alpha[after["start_frame"]:after["end_frame"]] > 0).any()
        for frame in range(after["start_frame"], after["end_frame"]):
            ink = new_alpha[frame] >= 128
            assert not (ink & ~mask).any()
            before_output = np.zeros((1920, 1080), dtype=np.uint8)
            after_output = np.zeros_like(before_output)
            old_y = baseline.y - (300 if slot else 0)
            before_output[old_y:old_y + baseline.height] = old_alpha[frame]
            after_output[placed.y:placed.y + placed.height] = new_alpha[frame]
            np.testing.assert_array_equal(after_output, before_output)
