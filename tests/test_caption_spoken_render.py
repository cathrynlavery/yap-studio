"""Pixel checks for a phrase building at synthetic spoken word times."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_captions_render import _pilot_plan, _rgba_frames, _style

from yap_studio import ffmpeg, paths
from yap_studio.captions import CaptionWord, build_phrases
from yap_studio.render import hyperframes
from yap_studio.stages.captions import (
    caption_layer_plan,
    caption_layout,
    prepare_layer_plan,
    render_layer_plan,
)

pytestmark = pytest.mark.render


@pytest.mark.parametrize("highlight", ["none", "slam", "moving_block"])
def test_real_caption_spoken_builds_without_moving_words(tmp_path, highlight) -> None:
    style = _style(highlight=highlight, shadow_opacity=0, italic_stress=False)
    style.data["captions"]["entrance"] = "spoken"
    words = [CaptionWord("w0", "Small", 0.2, 0.6),
             CaptionWord("w1", "steps", 0.6, 1.0),
             CaptionWord("w2", "matter", 1.0, 1.4)]
    plan = caption_layer_plan(build_phrases(words, style.data["captions"]), style, frames=45)
    assert len(json.loads(plan.variables["phrases"])) == 1
    with paths.operation():
        result = render_layer_plan(plan, tmp_path, tmp_path / "captions.mov")
    frames = _rgba_frames(result.path, plan)
    boxes = caption_layout(result.checks, plan)["captions"][0]["boxes"]
    assert len(boxes) == 3
    crops = []
    for box in boxes:
        left, right = int(np.floor(box["x"])), int(np.ceil(box["x"] + box["w"]))
        crops.append(frames[:, :, left:right, 3])
    assert crops[0][10].any() and crops[0][22].any()
    assert not crops[1][10].any() and crops[1][22].any()
    assert not crops[2][10].any() and not crops[2][22].any()
    for index, word in enumerate(words):
        first = round(word.start * 30)
        assert not crops[index][first - 1].any()
        assert crops[index][first].any()
    # The held first word stays on the same pixels as the second word appears.
    if highlight != "moving_block":
        np.testing.assert_array_equal(crops[0][10] > 100, crops[0][22] > 100)


@pytest.mark.parametrize("mode", ["box", "band"])
@pytest.mark.parametrize("font", ["Inter", "Instrument Serif"])
def test_real_caption_spoken_reserves_wrapped_layout_and_ground(tmp_path, mode, font) -> None:
    style = _style(highlight="none", size_px=180, font=font, background_mode=mode)
    style.data["captions"]["entrance"] = "spoken"
    plan = _pilot_plan(style)
    composition = prepare_layer_plan(plan, tmp_path)
    variables = tmp_path / "variables.json"
    variables.write_text(json.dumps(plan.variables))
    install = hyperframes.ensure_installed()
    node = shutil.which("node")
    assert node is not None and install.browser_path is not None
    result = subprocess.run([node, str(Path(__file__).parent / "fixtures/caption_spoken.mjs"),
                             str(install.prefix / "node_modules"), install.browser_path,
                             str(composition), str(variables), str(tmp_path)],
                            check=True, timeout=60, capture_output=True, text=True)
    box = json.loads(result.stdout)["first_word"]
    crops = []
    for name in ("first.png", "second.png"):
        raw = ffmpeg.run(["ffmpeg", "-nostdin", "-i", str(tmp_path / name),
                          "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1"],
                         timeout_s=120, binary=True).stdout
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(500, 1080, 4)
        left, top = int(np.floor(box["x"])), int(np.floor(box["y"]))
        right = int(np.ceil(box["x"] + box["width"]))
        bottom = int(np.ceil(box["y"] + box["height"]))
        crops.append((pixels[top:bottom, left:right, :3] > 200).all(axis=2))
    assert crops[0].any()
    np.testing.assert_array_equal(*crops)
