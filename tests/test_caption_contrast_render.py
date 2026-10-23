"""Opt-in contrast against white and dark pictures for every caption look."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_captions_render import _phrases, _style

from yap_studio import ffmpeg, paths
from yap_studio.checks.caption_contrast import measure_caption_contrast
from yap_studio.stages.captions import caption_layer_plan, caption_layout, render_layer_plan
from yap_studio.style import SCHEMA_PATH

_STYLES = json.loads(SCHEMA_PATH.read_bytes())["properties"]["captions"]["properties"][
    "highlight_style"
]["enum"]


@pytest.mark.render
@pytest.mark.parametrize("highlight,ground", [(style, "none") for style in _STYLES]
                         + [("none", "box")])
def test_every_caption_style_measures_on_the_final(tmp_path, highlight, ground) -> None:
    style = _style(highlight=highlight, text_color="#FFFFFF", highlight_color="#FFFFFF",
                   shadow_opacity=0, background_mode=ground, background_opacity=0.85,
                   background_color="#000000")
    phrases = [replace(_phrases(style)[0], end=0.9)]
    plan = caption_layer_plan(phrases, style, frames=30)
    with paths.operation():
        rendered = render_layer_plan(plan, tmp_path, tmp_path / "layer.mov")
    layout = caption_layout(rendered.checks, plan)
    placement = {"path": rendered.path.name, "x": plan.x, "y": plan.y,
                 "width": plan.width, "height": plan.height,
                 "start_frame": plan.start_frame, "frames": plan.frames}
    ratios = {}
    for name, color in (("white", "white"), ("black", "0x080808")):
        final = Path(tmp_path) / f"{name}.mp4"
        ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    f"color={color}:s=1080x1920:r=30:d=1", "-i", str(rendered.path),
                    "-filter_complex", f"[1:v]setpts=PTS+{plan.start_frame}/30/TB[layer];"
                    f"[0:v][layer]overlay=x={plan.x}:y={plan.y}:format=auto,format=yuv420p",
                    "-frames:v", "30", "-c:v", "libx264", "-preset", "ultrafast",
                    "-crf", "0", str(final)], timeout_s=120)
        result = measure_caption_contrast(final, rendered.path, placement, layout)
        assert result["measured"], (highlight, name, result)
        ratios[name] = [caption["ratio"] for caption in result["captions"]]
        assert ratios[name] and all(ratio is not None for ratio in ratios[name]), (
            highlight, ratios,
        )
    assert all(ratio >= 3 for ratio in ratios["black"]), (highlight, ratios)
    if ground == "box":
        assert all(ratio >= 3 for ratio in ratios["white"]), (highlight, ratios)
    elif highlight in {"none", "color"}:
        assert all(ratio < 1.5 for ratio in ratios["white"]), (highlight, ratios)


@pytest.mark.render
@pytest.mark.parametrize("highlight", _STYLES)
def test_every_style_backing_contrast_on_white(tmp_path, highlight) -> None:
    style = _style(highlight=highlight, text_color="#FFFFFF", highlight_color="#FFFFFF",
                   shadow_opacity=0, background_mode="none", background_color="#FFFFFF",
                   background_opacity=0.1)
    first = _phrases(style)[0]
    phrases = [replace(first, index=i, start=first.start + i, end=0.9 + i,
                       words=tuple(replace(word, start=word.start + i, end=word.end + i)
                                   for word in first.words)) for i in range(3)]
    plan = caption_layer_plan(phrases, style, frames=90, backing=[None, "shadow", "box"])
    with paths.operation():
        rendered = render_layer_plan(plan, tmp_path, tmp_path / "layer.mov")
    layout = caption_layout(rendered.checks, plan)
    placement = plan.placement(rendered.path.name)
    final = tmp_path / "white.mp4"
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "color=white:s=1080x1920:r=30:d=3", "-i", str(rendered.path),
                "-filter_complex", f"[0:v][1:v]overlay=x={plan.x}:y={plan.y}:"
                "format=auto,format=yuv420p", "-frames:v", "90", "-c:v", "libx264",
                "-preset", "ultrafast", "-crf", "0", str(final)], timeout_s=120)
    result = measure_caption_contrast(final, rendered.path, placement, layout)
    assert result["measured"], (highlight, result)
    ratios = {look: caption["ratio"] for look, caption in zip(
        ("plain", "shadow", "box"), result["captions"], strict=True)}
    assert all(ratio is not None for ratio in ratios.values()), (highlight, ratios)
    assert ratios["box"] >= 3, (highlight, ratios)
    if highlight in {"none", "color"}:
        assert ratios["shadow"] > ratios["plain"], (highlight, ratios)
