"""Synthetic picture assembly, native zoom sampling and composited end holds."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Generator
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import synth
from test_cards_render import renderer_install as renderer_install
from test_cut import make_project, time_segment, write_edit, write_json
from test_matte_project import synthetic_matte

from yap_studio import cards, ffmpeg, hdr, receipt
from yap_studio.errors import ProcessingError
from yap_studio.faces.model import Box
from yap_studio.matte import project as matte_cache
from yap_studio.media import catalog
from yap_studio.project.receipts import StageRun
from yap_studio.project.status import ProjectContext, cut_inputs
from yap_studio.stages import assemble_picture as stage
from yap_studio.stages.timeline import (
    ResolvedElement,
    ResolvedTimeline,
    TimelineRefusal,
    add_timeline,
    resolve_timeline,
)
from yap_studio.style import resolve


@pytest.fixture(autouse=True)
def refuse_browser_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every rendering test provide synthetic layers before invoking assembly."""
    def unexpected(_: Path) -> dict[str, Any]:
        pytest.fail("Test must supply synthetic alpha layers")

    monkeypatch.setattr(stage, "graphics", unexpected)
    monkeypatch.setattr(stage, "captions", unexpected)
    monkeypatch.setattr(matte_cache, "make_matte",
                        lambda *args, joins=(): pytest.fail("Unexpected person matte work"))


@pytest.fixture(scope="module")
def pictures(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Share lossless lavfi pictures; no browser is involved in layer fixtures."""
    folder = tmp_path_factory.mktemp("pictures")
    return {
        "cut": synth.picture_source(folder / "cut.mov",
                                    "color=c=blue:s=1080x1920:r=30", 90),
        "take": synth.picture_source(folder / "take.mov",
                                     "color=c=blue:s=160x90:r=30", 120),
        "insert": synth.picture_source(folder / "insert.mov",
                                       "color=c=red:s=320x180:r=30", 90),
        "title": synth.picture_source(folder / "title.mov",
                                      "color=c=yellow@0.5:s=200x100:r=30,format=rgba", 15,
                                      alpha=True),
        "captions": synth.picture_source(folder / "captions.mov",
                                         "color=c=lime:s=100x80:r=30,format=rgba", 90,
                                         alpha=True),
    }


def project(root: Path, pictures: dict[str, Path], *, frames: int = 90,
            hold_frames: int = 3, cut_pieces: list[dict] | None = None,
            word_rows: list[tuple[str, float, float]] | None = None,
            spans: list[dict[str, str]] | None = None) -> Path:
    """Publish real ingest/cut receipts around a synthetic output and exact spans."""
    if word_rows is None:
        word_rows = [("alpha", 0.1, 0.2)]
    make_project(root, [pictures["take"]], word_rows=[word_rows])
    data = json.loads((root / "project.json").read_bytes())
    fixture_style = resolve(data["style"]).data
    fixture_style["broll"]["color_match"] = "off"
    write_json(root / "picture-style.json", fixture_style)
    data["style"] = "picture-style.json"
    data["rules"] = {"zoom_min_s": 0.1, "zoom_max_s": 4, "zoom_step_min": 0,
                     "elements_at_once_max": 3}
    write_json(root / "project.json", data)
    write_edit(root, [time_segment("s1", "t1", 0, frames / 30)], end_hold_s=hold_frames / 30)
    inputs = cut_inputs(ProjectContext(root))
    word_map = write_document(root / "words.json", {
        "schema": "yap-studio/cut-words@1", "time_base": "output",
        "words_sha256": {"t1": receipt.sha256_file(root / "ingest/t1/words.json")},
        "words": [{"id": f"t1:{index}", "text": text, "segment": "s1", "start": start,
                   "end": end} for index, (text, start, end) in enumerate(word_rows)],
    })
    spans = write_document(root / "spans.json", {
        "sources": {"t1": "../takes/t1.mov"}, "fps": "30/1",
        "segments": ([{"source": p["take"], "start": str(p["src_f0"] / 30),
                       "end": str(p["src_f1"] / 30)} for p in cut_pieces] if cut_pieces else
                     spans if spans is not None else
                     [{"source": "t1", "start": "0", "end": str(frames / 30)}]),
    })
    staged = root / "picture.mp4"
    ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(pictures["cut"]),
                "-map", "0:v:0", "-an", "-frames:v", str(frames), "-c:v", "libx264",
                "-preset", "ultrafast", "-crf", "0", "-pix_fmt", "yuv420p",
                "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                str(staged)], timeout_s=120)
    with StageRun(root, "cut") as run:
        for name, digest in inputs.items():
            run.record_inputs(name, digest)
        for name, source in (("cut.mp4", staged), ("cut-words.json", word_map),
                             ("spans.json", spans)):
            run.replace_output(root / "cut" / name, source)
        run.finish(extra={"plan": {"total_frames": frames,
                                   **({"pieces": cut_pieces} if cut_pieces else {})}})
    return root


def write_document(path: Path, data: dict[str, Any]) -> Path:
    """Write a synthetic contract and return its staging path."""
    write_json(path, data)
    return path


@pytest.mark.parametrize("speed", [0.5, 1, 2])
def test_card_window_exact_frames_mask_and_fades(tmp_path: Path, speed: float) -> None:
    source = synth.picture_source(tmp_path / "clip.mov",
                                  "nullsrc=s=128x64:r=30,geq=lum='mod(N*8,256)':cb=128:cr=128",
                                  90)
    cut = stage._source(synth.picture_source(tmp_path / "cut.mov",
                                            "color=c=blue:s=160x120:r=30", 40), [], [], owner="cut")
    window = {"slot": "clip", "index": 0, "x": 20, "y": 20, "width": 64, "height": 64,
              "radius": 16, "start_frame": 10, "frames": 20, "fade_in_frames": 4,
              "fade_out_frames": 4,
              "source": {"kind": "media", "path": source.name,
                         "sha256": receipt.sha256_file(source), "in_s": 0.2,
                         "speed": speed, "focus": [0.5, 0.5], "audio": "mute"}}
    placement = {"id": "agent-video", "windows": [window]}
    sources = stage._window_sources(tmp_path, [placement], cut, [], [])
    graph = stage._Graph()
    base = graph.filter(graph.input(cut), "settb=1/30,setpts=N,format=rgba")
    base, seconds = stage._card_windows(graph, base, placement, sources)
    assert seconds >= 20 / 30
    result = ffmpeg.run(["ffmpeg", "-v", "error", "-filter_complex_threads", "1",
                         *graph.inputs, "-filter_complex", ";".join(graph.filters),
                         "-map", f"[{base}]", "-frames:v", "40", "-pix_fmt", "rgb24",
                         "-f", "rawvideo", "pipe:1"], binary=True, timeout_s=30)
    pixels = np.frombuffer(result.stdout, dtype=np.uint8).reshape(40, 120, 160, 3)
    np.testing.assert_array_equal(pixels[9, 50, 50], pixels[30, 50, 50])
    np.testing.assert_array_equal(pixels[20, 20, 20], pixels[9, 20, 20])
    expected = ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(source), "-pix_fmt", "rgb24",
                           "-f", "rawvideo", "pipe:1"], binary=True, timeout_s=30)
    frames = np.frombuffer(expected.stdout, dtype=np.uint8).reshape(90, 64, 128, 3)
    for offset in (4, 8, 12):
        np.testing.assert_allclose(pixels[10 + offset, 50, 50],
                                   frames[6 + round(offset * speed), 32, 64], atol=2)
    blend = pixels[12, 50, 50].astype(float)
    want = (pixels[9, 50, 50].astype(float) + frames[6 + round(2 * speed), 32, 64]) / 2
    np.testing.assert_allclose(blend, want, atol=2)
    assert sum("geq=" in value for value in graph.filters) == 1
    assert any("trim=end_frame=1" in value and "loop=loop=-1:size=1" in value
               for value in graph.filters)


@pytest.mark.render
@pytest.mark.parametrize("graded", [False, True])
def test_file_drop_window_plays_poster_frame_in_sync(
    tmp_path: Path, renderer_install, graded: bool,
) -> None:
    from yap_studio.grade import GradeSpec, apply_rgb, write_cube
    from yap_studio.style import resolve

    clip = synth.picture_source(tmp_path / "clip.mov",
                               "nullsrc=s=320x180:r=30,geq=lum='mod(N*8,256)':cb=128:cr=128", 90)
    plan = cards.card_layer_plan("file_drop", {"clip": clip, "playback": "play", "poster_at": 20},
                                 resolve("preset:clean_minimal"), frames=75)
    local, = plan.windows
    window = {**local, "start_frame": local["start"], "fade_in_frames": local["fade_in"],
              "fade_out_frames": local["fade_out"],
              "source": {**local["source"], "path": clip.name,
                         "sha256": receipt.sha256_file(clip)}}
    chrome = cards.render_card(plan, tmp_path / "card-work", tmp_path / "chrome.mov",
                               install=renderer_install).path
    plate = synth.picture_source(tmp_path / "plate.mov",
                                f"color=blue:s={plan.width}x{plan.height}:r=30", plan.frames)
    cut = stage._source(plate, [], [], owner="cut")
    placement = {"id": "file-card", "windows": [window]}
    sources = stage._window_sources(tmp_path, [placement], cut, [], [])
    graph = stage._Graph()
    base = graph.filter(graph.input(cut), "format=rgba,settb=1/30,setpts=N")
    spec = GradeSpec("film", 1)
    cube = write_cube(spec, tmp_path / "cube") if graded else None
    base, _ = stage._card_windows(graph, base, placement, sources, cube=cube)
    layer = graph.filter(graph.input(stage._source(chrome, [], [], owner="chrome")),
                         "format=rgba,settb=1/30,setpts=N")
    base = graph.overlay(base, layer, 0, plan.frames)
    result = ffmpeg.run(["ffmpeg", "-v", "error", "-filter_complex_threads", "1",
                         *graph.inputs, "-filter_complex", ";".join(graph.filters),
                         "-map", f"[{base}]", "-frames:v", str(plan.frames), "-pix_fmt", "rgb24",
                         "-f", "rawvideo", "pipe:1"], binary=True, timeout_s=60)
    pixels = np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        plan.frames, plan.height, plan.width, 3)
    x, y = window["x"] + window["width"] // 2, window["y"] + window["height"] // 2
    first = round(window["source"]["in_s"] * plan.fps)
    poster = synth.picture_pixels(clip, first)[90, 160]
    for frame in (window["start_frame"] - 1, window["start_frame"], window["start_frame"] + 1):
        np.testing.assert_allclose(pixels[frame, y, x], poster, atol=4)
    for offset in (2, 8, 20):
        expected = synth.picture_pixels(clip, first + offset)[90, 160]
        if graded:
            expected = apply_rgb(expected / 255, spec) * 255
        np.testing.assert_allclose(pixels[window["start_frame"] + offset, y, x], expected, atol=3)
    assert ";".join(graph.filters).count("lut3d=") == int(graded)


@pytest.mark.parametrize("position,rect", [
    ((20, 20), (18, 18, 68, 68)),
    ((0, 0), (0, 0, 66, 66)),
    ((1, 1), (0, 0, 66, 66)),
    ((1016, 1856), (1014, 1854, 66, 66)),
    ((1015, 1855), (1013, 1853, 66, 66)),
])
def test_card_window_bleeds_under_frame_within_canvas(
    pictures: dict[str, Path], position: tuple[int, int], rect: tuple[int, int, int, int],
) -> None:
    cut = stage._source(pictures["cut"], [], [], owner="cut")
    window = {"x": position[0], "y": position[1], "width": 64, "height": 64, "radius": 16,
              "start_frame": 0, "frames": 20, "fade_in_frames": 1, "fade_out_frames": 1,
              "source": {"kind": "self"}}
    graph = stage._Graph()
    stage._card_windows(graph, "base", {"id": "card", "windows": [window]},
                        {("card", 0): cut})
    x, y, width, height = rect
    filters = ";".join(graph.filters)
    assert f"overlay=x={x}:y={y}:" in filters
    assert f"crop={width}:{height}:" in filters
    assert f"color=c=white:s={width}x{height}:" in filters
    assert "255*clip(18.0+0.5-sqrt(" in filters


def test_card_window_rounded_chrome_covers_corner_without_base(tmp_path: Path) -> None:
    source = synth.picture_source(tmp_path / "clip.mov", "color=lime:s=128x64:r=30", 20)
    clip = stage._source(source, [], [], owner="clip")
    cut = stage._source(synth.picture_source(tmp_path / "cut.mov",
                                            "color=blue:s=160x120:r=30", 20), [], [], owner="cut")
    window = {"x": 20, "y": 20, "width": 64, "height": 64, "radius": 16,
              "start_frame": 0, "frames": 20, "fade_in_frames": 1, "fade_out_frames": 1,
              "source": {"kind": "media"}}
    graph = stage._Graph()
    base = graph.filter(graph.input(cut), "format=rgba,settb=1/30,setpts=N")
    base, _ = stage._card_windows(graph, base, {"id": "card", "windows": [window]},
                                 {("card", 0): clip})
    dx = "max(max(36-X,X-68),0)"
    dy = "max(max(36-Y,Y-68),0)"
    hole = f"clip(16-sqrt({dx}*{dx}+{dy}*{dy}),0,1)"
    graph.filters.append("color=yellow:s=160x120:r=30,format=rgba,"
                         "geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':"
                         f"a='255*(1-{hole})'[chrome]")
    base = graph.overlay(base, "chrome", 0, 20)
    result = ffmpeg.run(["ffmpeg", "-v", "error", "-filter_complex_threads", "1",
                         *graph.inputs, "-filter_complex", ";".join(graph.filters),
                         "-map", f"[{base}]", "-frames:v", "20", "-pix_fmt", "rgb24",
                         "-f", "rawvideo", "pipe:1"], binary=True, timeout_s=30)
    pixels = np.frombuffer(result.stdout, dtype=np.uint8).reshape(20, 120, 160, 3)
    np.testing.assert_allclose(pixels[10, 21, 21], [255, 255, 0], atol=3)
    np.testing.assert_allclose(pixels[10, 40, 40], [0, 255, 0], atol=3)
    assert pixels[10, 20:40, 20:40, 2].max() < 4


@pytest.mark.parametrize("focus,color", [([0, 0], [255, 0, 0]), ([1, 1], [0, 0, 255])])
def test_card_window_focus_cover(tmp_path: Path, focus: list, color: list) -> None:
    source = synth.picture_source(tmp_path / "clip.mov",
                                  "color=c=red:s=128x64:r=30,drawbox=x=64:y=0:w=64:h=64:"
                                  "color=blue:t=fill", 30)
    cut = stage._source(source, [], [], owner="clip")
    window = {"x": 0, "y": 0, "width": 64, "height": 64, "radius": 8,
              "start_frame": 0, "frames": 20, "fade_in_frames": 1, "fade_out_frames": 1,
              "source": {"kind": "media", "in_s": 0, "speed": 1, "focus": focus}}
    graph = stage._Graph()
    clip = stage._window_clip(graph, cut, window, owner="clip")
    result = ffmpeg.run(["ffmpeg", "-v", "error", "-filter_complex_threads", "1",
                         *graph.inputs, "-filter_complex", ";".join(graph.filters),
                         "-map", f"[{clip}]", "-frames:v", "20", "-pix_fmt", "rgba",
                         "-f", "rawvideo", "pipe:1"], binary=True, timeout_s=30)
    pixels = np.frombuffer(result.stdout, dtype=np.uint8).reshape(20, 66, 66, 4)
    np.testing.assert_allclose(pixels[10, 32, 32, :3], color, atol=3)
    assert pixels[10, 0, 0, 3] == 0


@pytest.mark.parametrize("rotation,crop,focus,color", [
    (0, [1, 1, 63, 61], [1, 1], [255, 0, 0]),
    (0, [65, 1, 63, 61], [0, 0], [0, 0, 255]),
    (90, [1, 1, 61, 63], [1, 1], [0, 0, 255]),
    (90, [1, 65, 61, 63], [0, 0], [255, 0, 0]),
])
def test_card_window_pixel_crop_precedes_cover_and_focus(
    tmp_path: Path, rotation: int, crop: list[int], focus: list[int], color: list[int],
) -> None:
    source = synth.picture_source(tmp_path / "clip.mov",
                                  "color=c=red:s=128x64:r=30,drawbox=x=64:y=0:w=64:h=64:"
                                  "color=blue:t=fill", 30)
    if rotation:
        upright = tmp_path / "rotated.mov"
        ffmpeg.run(["ffmpeg", "-v", "error", "-display_rotation", str(rotation), "-i", str(source),
                    "-c", "copy", str(upright)], timeout_s=30)
        source = upright
    decoded = stage._source(source, [], [], owner="clip")
    window = {"x": 0, "y": 0, "width": 64, "height": 64, "radius": 8,
              "start_frame": 0, "frames": 20, "fade_in_frames": 1, "fade_out_frames": 1,
              "source": {"kind": "media", "crop": crop, "focus": focus}}
    graph = stage._Graph()
    clip = stage._window_clip(graph, decoded, window, owner="clip")
    filters = ";".join(graph.filters)
    exact_crop = f"crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]}:exact=1"
    assert exact_crop in filters and filters.index(exact_crop) < filters.index("scale=w=")
    result = ffmpeg.run(["ffmpeg", "-v", "error", "-filter_complex_threads", "1",
                         *graph.inputs, "-filter_complex", filters,
                         "-map", f"[{clip}]", "-frames:v", "20", "-pix_fmt", "rgba",
                         "-f", "rawvideo", "pipe:1"], binary=True, timeout_s=30)
    pixels = np.frombuffer(result.stdout, dtype=np.uint8).reshape(20, 66, 66, 4)
    np.testing.assert_allclose(pixels[10, 30:36, 30:36, :3],
                               np.broadcast_to(color, (6, 6, 3)), atol=3)
    assert pixels[10, 0, 0, 3] == 0


def test_card_window_self_refuses_pixel_crop(tmp_path: Path) -> None:
    source = synth.picture_source(tmp_path / "cut.mov", "color=s=64x64:r=30", 30)
    cut = stage._source(source, [], [], owner="cut")
    window = {"x": 0, "y": 0, "width": 64, "height": 64, "radius": 8,
              "start_frame": 0, "frames": 20, "fade_in_frames": 1, "fade_out_frames": 1,
              "source": {"kind": "self", "crop": [0, 0, 32, 32]}}
    with pytest.raises(stage.PictureError, match="self takes no crop"):
        stage._window_sources(tmp_path, [{"id": "card", "windows": [window]}], cut, [], [])


def test_card_window_self_uses_independent_cut_input_at_output_frames(tmp_path: Path) -> None:
    source = synth.picture_source(tmp_path / "cut.mov",
                                  "nullsrc=s=64x64:r=30,geq=lum='mod(N*8,256)':cb=128:cr=128",
                                  40)
    cut = stage._source(source, [], [], owner="cut")
    graph = stage._Graph()
    window = {"x": 0, "y": 0, "width": 64, "height": 64, "radius": 8,
              "start_frame": 10, "frames": 20, "fade_in_frames": 1, "fade_out_frames": 1,
              "source": {"kind": "self"}}
    clip = stage._window_clip(graph, cut, window, owner="self")
    result = ffmpeg.run(["ffmpeg", "-v", "error", "-filter_complex_threads", "1",
                         *graph.inputs, "-filter_complex", ";".join(graph.filters),
                         "-map", f"[{clip}]", "-frames:v", "20", "-fps_mode", "passthrough",
                         "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1"],
                        binary=True, timeout_s=30)
    expected = ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(source), "-pix_fmt", "rgba",
                           "-f", "rawvideo", "pipe:1"], binary=True, timeout_s=30)
    pixels = np.frombuffer(result.stdout, dtype=np.uint8).reshape(20, 66, 66, 4)
    frames = np.frombuffer(expected.stdout, dtype=np.uint8).reshape(40, 64, 64, 4)
    for offset in (1, 8, 18):
        np.testing.assert_allclose(pixels[offset, 32, 32], frames[10 + offset, 32, 32], atol=2)
    assert graph.inputs.count(str(source)) == 1
    assert not any("split=" in value for value in graph.filters)
    assert any("trim=start_frame=10:end_frame=30" in value for value in graph.filters)


def test_playing_grid_clips_follow_staggered_window_frames(tmp_path: Path) -> None:
    from yap_studio.style import resolve

    source = synth.picture_source(tmp_path / "clip.mov",
                                  "nullsrc=s=128x64:r=30,geq=lum='mod(N*8,256)':cb=128:cr=128",
                                  90)
    cut = stage._source(synth.picture_source(tmp_path / "cut.mov",
                                            "color=blue:s=1080x1920:r=30", 60), [], [], owner="cut")
    clip = {"path": str(source), "sha256": receipt.sha256_file(source), "in_s": 0.2}
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_kinds_on"] = ["playing_grid"]
    plan = cards.card_layer_plan("playing_grid", {"clips": [clip, clip]}, style, frames=45)
    windows = [{**window, "start_frame": 6 + window["start"],
                "fade_in_frames": window["fade_in"], "fade_out_frames": window["fade_out"],
                "source": {**window["source"], "path": source.name}} for window in plan.windows]
    placement = {"id": "grid", "windows": windows}
    sources = stage._window_sources(tmp_path, [placement], cut, [], [])
    graph = stage._Graph()
    base = graph.filter(graph.input(cut), "format=rgba,settb=1/30,setpts=N")
    base, _ = stage._card_windows(graph, base, placement, sources)
    y = windows[0]["y"] + windows[0]["height"] // 2
    base = graph.filter(base, f"crop=1080:2:0:{y}")
    result = ffmpeg.run(["ffmpeg", "-v", "error", "-filter_complex_threads", "1",
                         *graph.inputs, "-filter_complex", ";".join(graph.filters),
                         "-map", f"[{base}]", "-frames:v", "60", "-pix_fmt", "rgb24",
                         "-f", "rawvideo", "pipe:1"], binary=True, timeout_s=60)
    pixels = np.frombuffer(result.stdout, dtype=np.uint8).reshape(60, 2, 1080, 3)
    expected = ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(source), "-pix_fmt", "rgb24",
                           "-f", "rawvideo", "pipe:1"], binary=True, timeout_s=30)
    frames = np.frombuffer(expected.stdout, dtype=np.uint8).reshape(90, 64, 128, 3)
    assert [window["start_frame"] for window in windows] == [6, 10]
    for window in windows:
        x, start = window["x"] + window["width"] // 2, window["start_frame"]
        np.testing.assert_allclose(pixels[start - 1, 0, x], [0, 0, 255], atol=3)
        np.testing.assert_allclose(pixels[start + 8, 0, x], frames[14, 32, 64], atol=3)
        np.testing.assert_allclose(pixels[51, 0, x], [0, 0, 255], atol=3)


def test_reel_feed_self_screen_keeps_the_cut_frame_clock(tmp_path: Path) -> None:
    from yap_studio.style import resolve

    source = synth.picture_source(tmp_path / "cut.mov",
                                  "nullsrc=s=64x64:r=30,geq=lum='mod(N*8,256)':cb=128:cr=128", 90)
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_kinds_on"] = ["reel_feed"]
    plan = cards.card_layer_plan("reel_feed", {"ending": "fade"}, style, frames=60)
    window = {**plan.windows[0], "start_frame": 6, "fade_in_frames": 9, "fade_out_frames": 9}
    graph = stage._Graph()
    decoded = stage._source(source, [], [], owner="cut")
    layer = stage._window_clip(graph, decoded, window, owner="feed")
    layer = graph.filter(layer, "crop=2:2:iw/2:ih/2")
    result = ffmpeg.run(["ffmpeg", "-v", "error", "-filter_complex_threads", "1",
                         *graph.inputs, "-filter_complex", ";".join(graph.filters),
                         "-map", f"[{layer}]", "-frames:v", "60", "-fps_mode", "passthrough",
                         "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1"],
                        binary=True, timeout_s=60)
    pixels = np.frombuffer(result.stdout, dtype=np.uint8).reshape(60, 2, 2, 4)
    expected = ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(source), "-pix_fmt", "rgba",
                           "-f", "rawvideo", "pipe:1"], binary=True, timeout_s=30)
    frames = np.frombuffer(expected.stdout, dtype=np.uint8).reshape(90, 64, 64, 4)
    for offset in (10, 20, 40):
        np.testing.assert_allclose(pixels[offset, 0, 0], frames[6 + offset, 32, 32], atol=3)


def test_reel_feed_handoff_last_frame_matches_cut_over_the_whole_canvas(tmp_path: Path) -> None:
    from yap_studio.cards_video import window_rect
    from yap_studio.stages.graphics import GraphicSnapshot
    from yap_studio.style import resolve

    source = synth.picture_source(tmp_path / "cut.mov", "nullsrc=s=1080x1920:r=30,"
                                  "geq=lum='24+mod(N*5+X/32+Y/16,200)':cb=128:cr=128", 82)
    decoded = stage._source(source, [], [], owner="cut")
    plan = cards.card_layer_plan("reel_feed", {}, resolve("preset:clean_minimal"), frames=75)
    placement = GraphicSnapshot("feed", "front", 6, plan, None, (), "a" * 64, {}).placement()
    window, = placement["windows"]
    assert window["motion"]["start_frame"] == 63
    sources = stage._window_sources(tmp_path, [placement], decoded, [], [])
    graph = stage._Graph()
    base = graph.filter(graph.input(decoded), "format=rgba,settb=1/30,setpts=N")
    base, _ = stage._card_windows(graph, base, placement, sources)
    x, y, width, height = (window[key] for key in ("x", "y", "width", "height"))
    graph.filters.append(
        "color=c=white:s=1080x1920:r=30,trim=end_frame=1,format=rgba,"
        f"geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='if(between(X,{x},{x + width - 1})*"
        f"between(Y,{y},{y + height - 1}),0,255)',loop=loop=-1:size=1,trim=end_frame=75,"
        "fade=t=out:start_frame=57:nb_frames=6:alpha=1,settb=1/30,setpts=N+6[swipe_chrome]",
    )
    base = graph.overlay(base, "swipe_chrome", 6, 81)
    selected = (63, 72, 80, 81)
    select = "+".join(f"eq(n,{frame})" for frame in selected)
    base = graph.filter(base, f"select='{select}'")
    result = ffmpeg.run(["ffmpeg", "-v", "error", "-filter_complex_threads", "1",
                         *graph.inputs, "-filter_complex", ";".join(graph.filters),
                         "-map", f"[{base}]", "-fps_mode", "passthrough", "-pix_fmt", "rgb24",
                         "-f", "rawvideo", "pipe:1"], binary=True, timeout_s=120)
    actual = np.frombuffer(result.stdout, dtype=np.uint8).reshape(4, 1920, 1080, 3)
    expected = ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(source), "-vf",
                           f"format=rgba,select='{select}'", "-fps_mode", "passthrough",
                           "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
                          binary=True, timeout_s=60)
    frames = np.frombuffer(expected.stdout, dtype=np.uint8).reshape(4, 1920, 1080, 3)
    np.testing.assert_array_equal(actual[0, 0, 0], [255, 255, 255])
    np.testing.assert_array_equal(actual[2:], frames[2:])
    x, y, width, height = window_rect(window, 72)
    assert 0 < x < window["x"] and window["width"] < width < 1080
    np.testing.assert_array_equal(actual[1, :y - 4], frames[1, :y - 4])
    assert not np.array_equal(actual[1, y + height // 2, x + width // 2],
                              frames[1, y + height // 2, x + width // 2])
    assert any("eval=frame" in value and "scale=" in value for value in graph.filters)
    assert any("eval=frame" in value and "overlay=" in value for value in graph.filters)


def test_film_moving_reel_feed_window_is_graded_once_and_hands_off(tmp_path: Path) -> None:
    from yap_studio.cards_video import window_rect
    from yap_studio.grade import GradeSpec, apply_rgb, graph_filter, write_cube
    from yap_studio.stages.graphics import GraphicSnapshot
    from yap_studio.style import resolve

    source = synth.picture_source(tmp_path / "cut.mov", "nullsrc=s=1080x1920:r=30,"
                                  "geq=lum='40+mod(N*3,100)':cb=150:cr=110", 82)
    decoded = stage._source(source, [], [], owner="cut")
    plan = cards.card_layer_plan("reel_feed", {}, resolve("preset:clean_minimal"), frames=75)
    placement = GraphicSnapshot("feed", "front", 6, plan, None, (), "a" * 64, {}).placement()
    window, = placement["windows"]
    sources = stage._window_sources(tmp_path, [placement], decoded, [], [])
    spec = GradeSpec("film", 1)
    cube = write_cube(spec, tmp_path / "cube")
    selected = (72, 80, 81)
    select = "+".join(f"eq(n,{frame})" for frame in selected)

    def pixels(*, graded: bool, feed: bool = True) -> tuple[np.ndarray, str]:
        graph = stage._Graph()
        base = graph.filter(graph.input(decoded), "settb=1/30,setpts=N,setsar=1")
        if graded:
            base = graph.filter(base, graph_filter(cube))
        if feed:
            base, _ = stage._card_windows(graph, base, placement, sources,
                                          cube=cube if graded else None)
        base = graph.filter(base, f"select='{select}'")
        filters = ";".join(graph.filters)
        result = ffmpeg.run(["ffmpeg", "-v", "error", "-filter_complex_threads", "1",
                             *graph.inputs, "-filter_complex", filters,
                             "-map", f"[{base}]", "-fps_mode", "passthrough",
                             "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
                            binary=True, timeout_s=120)
        return np.frombuffer(result.stdout, dtype=np.uint8).reshape(3, 1920, 1080, 3), filters

    plain, plain_graph = pixels(graded=False)
    graded, graded_graph = pixels(graded=True)
    base, _ = pixels(graded=True, feed=False)
    x, y, width, height = window_rect(window, selected[0])
    assert window["width"] < width < 1080
    raw = plain[0, y + height // 2, x + width // 2] / 255
    actual = graded[0, y + height // 2, x + width // 2]
    expected = apply_rgb(raw, spec) * 255
    np.testing.assert_allclose(actual, expected, atol=2)
    assert np.abs(actual.astype(float) - apply_rgb(expected / 255, spec) * 255).mean() > 3
    # The base grades at 16 bits; the RGBA window grades at 8 bits.
    np.testing.assert_allclose(graded[1], base[1], atol=3)
    assert np.abs(graded[1].astype(float) - base[1]).mean() < 1.5
    np.testing.assert_array_equal(graded[2], base[2])
    assert "lut3d=" not in plain_graph
    assert graded_graph.count("lut3d=") == 2
    assert "eval=frame" in graded_graph


def lane(root: Path, elements: list[dict[str, Any]], *, name: str = "agent") -> None:
    """Author portable time anchors for the real read-only timeline checker."""
    write_json(root / f"timeline/{name}.json", {"schema": "yap-studio/timeline@1",
                                              "words_sha256": {},
                                              "elements": [{**item, "lane": name}
                                                           for item in elements]})


def insert(root: Path, source: Path, first: int, last: int,
           *, identifier: str = "broll-one", **media: Any) -> dict[str, Any]:
    """Copy synthetic media into the project and bind its content identity."""
    target = root / f"{identifier}{source.suffix}"
    shutil.copyfile(source, target)
    return {"id": identifier, "kind": "insert", "start": {"time": first / 30},
            "end": {"time": last / 30},
            "media": {"path": target.name, "sha256": receipt.sha256_file(target), **media}}


def layers(root: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
           *, titles: list[dict[str, Any]] | None = None) -> None:
    """Supply real synthetic alpha movies through the existing stage return contracts."""
    for name in ("title", "captions"):
        shutil.copyfile(pictures[name], root / f"{name}.mov")
    if titles is None:
        titles = [{"id": "agent-title", "path": "title.mov", "x": 80, "y": 100,
                   "width": 200, "height": 100,
                   "start_frame": 0, "frames": 15, "layer": "behind", "kind": "title"}]
    monkeypatch.setattr(stage, "graphics", lambda _: {"elements": titles})
    monkeypatch.setattr(stage, "captions", lambda _: {"layer": {
        "path": "captions.mov", "x": 60, "y": 1800, "width": 100, "height": 80,
        "start_frame": 0, "frames": 90,
    }})


def render(root: Path) -> tuple[Path, stage.PictureResult]:
    """Render into a new caller-owned staging area."""
    out = root / "work/picture.mp4"
    return out, stage.render_picture(root, out, work_dir=root / "work")


@pytest.mark.parametrize("explicit", [False, True])
def test_camera_picture_is_base_and_caption_is_last(
    tmp_path, pictures, monkeypatch, explicit,
) -> None:
    root = project(tmp_path, pictures)
    layers(root, pictures, monkeypatch, titles=[])
    camera = synth.picture_source(root / "camera.mp4", "color=red:s=1080x1920:r=30", 90)
    original = stage.captions(root)
    monkeypatch.setattr(stage, "captions", lambda _: {**original, "picture": camera.name})
    out = root / "work/picture.mp4"
    timeline = replace(resolve_timeline(root), camera_3d=True)
    result = stage.render_picture(root, out, work_dir=root / "work", timeline=timeline,
                                   **({"base_picture": camera} if explicit else {}))
    assert ffmpeg.probe(out).video_frames == 93
    args = next(command for command in result.commands if "-filter_complex" in command)
    assert args[args.index("-i") + 1] == str(camera)
    assert str(root / "captions.mov") in args
    raw = ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(out), "-frames:v", "1",
                       "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"], binary=True, timeout_s=30)
    pixels = np.frombuffer(raw.stdout, np.uint8).reshape(1920, 1080, 3)
    assert_colour(pixels[800:850, 500:550], (255, 0, 0))
    assert_colour(pixels[1820:1840, 80:100], (0, 255, 0))


def test_camera_look_without_visible_captions_uses_the_cut(tmp_path, pictures, monkeypatch) -> None:
    root = project(tmp_path, pictures)
    layers(root, pictures, monkeypatch, titles=[])
    out = root / "work/picture.mp4"
    timeline = replace(resolve_timeline(root), camera_3d=True)
    result = stage.render_picture(root, out, work_dir=root / "work", timeline=timeline)
    args = next(command for command in result.commands if "-filter_complex" in command)
    assert args[args.index("-i") + 1] == str(root / "cut/cut.mp4")
    assert str(root / "captions.mov") in args
    assert ffmpeg.probe(out).video_frames == 93


def assert_colour(pixels: np.ndarray, rgb: tuple[int, int, int], *, tolerance: int = 8) -> None:
    """Compare average swatches with room for the delivery encoder's quantization."""
    np.testing.assert_allclose(pixels.mean(axis=(0, 1)), rgb, atol=tolerance)


@pytest.mark.parametrize("kind", ["media", "self"])
def test_card_window_grade_preserves_alpha_and_no_grade_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    from yap_studio.grade import GradeSpec, apply_rgb, write_cube

    source = stage._source(synth.picture_source(
        tmp_path / "clip.mov", "color=0x20506e:s=64x64:r=30", 30), [], [], owner="clip")
    window = {"x": 20, "y": 20, "width": 64, "height": 64, "radius": 16,
              "start_frame": 4, "frames": 20, "fade_in_frames": 4, "fade_out_frames": 4,
              "source": {"kind": kind}}

    def pixels(cube: Path | None = None, *, direct: bool = False) -> tuple[np.ndarray, str]:
        graph = stage._Graph()
        branches = []

        def overlay(base, layer, *args, **kwargs):
            branches.append(layer)
            return base

        monkeypatch.setattr(graph, "overlay", overlay)
        if direct:
            branch = stage._window_clip(graph, source, window, owner="clip")
        else:
            stage._card_windows(graph, "base", {"id": "card", "windows": [window]},
                                {("card", 0): source}, **({"cube": cube} if cube else {}))
            branch, = branches
        filters = ";".join(graph.filters)
        result = ffmpeg.run(["ffmpeg", "-v", "error", "-filter_complex_threads", "1",
                             *graph.inputs, "-filter_complex", filters,
                             "-map", f"[{branch}]", "-frames:v", "20", "-fps_mode", "passthrough",
                             "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1"],
                            binary=True, timeout_s=30)
        return np.frombuffer(result.stdout, dtype=np.uint8).reshape(20, 68, 68, 4), filters

    baseline, _ = pixels(direct=True)
    plain, plain_graph = pixels()
    np.testing.assert_array_equal(plain, baseline)
    assert "lut3d" not in plain_graph
    spec = GradeSpec("film", 1)
    graded, graded_graph = pixels(write_cube(spec, tmp_path / "cube"))
    np.testing.assert_array_equal(graded[..., 3], plain[..., 3])
    assert plain[0, 34, 34, 3] == plain[10, 0, 0, 3] == 0
    assert plain[10, 34, 34, 3] == 255 and 0 < plain[-1, 34, 34, 3] < 255
    expected = apply_rgb(plain[10, 34, 34, :3] / 255, spec) * 255
    np.testing.assert_allclose(graded[10, 34, 34, :3], expected, atol=2)
    assert graded_graph.count("lut3d=") == 1


@pytest.mark.render
@pytest.mark.parametrize("kind", ["media", "self"])
@pytest.mark.parametrize("behind", [False, True])
def test_film_card_window_is_graded_once_and_chrome_unchanged(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
    kind: str, behind: bool,
) -> None:
    from yap_studio.grade import GradeSpec, apply_rgb

    plate = synth.picture_source(tmp_path / "plate.mov", "color=0x20506e:s=1080x1920:r=30", 90)
    root = project(tmp_path / "project", {**pictures, "cut": plate}, hold_frames=0)
    layer = "behind" if behind else "front"
    lane(root, [{"id": "agent-card", "kind": "graphic", "layer": layer,
                 "start": {"time": 0}, "hold_s": 1,
                 "template": "text_tag", "slots": {"text": "Alpha"}}])
    chrome = synth.picture_source(
        tmp_path / "chrome.mov", "color=0xffcc66:s=160x120:r=30,format=rgba,"
        "geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':"
        "a='if(between(X,16,143)*between(Y,16,103),0,255)'", 30, alpha=True)
    source = {"kind": kind}
    if kind == "media":
        clip = synth.picture_source(root / "clip.mov", "color=0x20506e:s=128x88:r=30", 30)
        source.update(path=clip.name, sha256=receipt.sha256_file(clip))
    window = {"slot": "clip", "index": 0, "x": 96, "y": 116, "width": 128, "height": 88,
              "radius": 12, "start_frame": 0, "frames": 30, "fade_in_frames": 3,
              "fade_out_frames": 3, "source": source}
    placement = {"id": "agent-card", "path": "title.mov", "x": 80, "y": 100,
                 "width": 160, "height": 120, "start_frame": 0, "frames": 30,
                 "layer": layer, "kind": "graphic", "windows": [window]}
    layers(root, {**pictures, "title": chrome}, monkeypatch, titles=[placement])
    if behind:
        monkeypatch.setattr(matte_cache, "make_matte", lambda cut, folder: synthetic_matte(
            cut, folder, box=(800, 800, 100, 100)))
    cubes, window_cubes = [], []
    original_write_cube, original_windows = stage.write_cube, stage._card_windows

    def write_cube(*args, **kwargs):
        cube = original_write_cube(*args, **kwargs)
        cubes.append(cube)
        return cube

    def card_windows(graph, base, placement, sources, **kwargs):
        if placement.get("windows"):
            window_cubes.append(kwargs.get("cube"))
        return original_windows(graph, base, placement, sources, **kwargs)

    monkeypatch.setattr(stage, "write_cube", write_cube)
    monkeypatch.setattr(stage, "_card_windows", card_windows)
    plain, plain_result = render(root)
    assert not cubes
    assert "lut3d" not in picture_command(plain_result)[
        picture_command(plain_result).index("-filter_complex") + 1]
    spec = GradeSpec("film", 1)
    out = root / "film/picture.mp4"
    result = stage.render_picture(root, out, work_dir=out.parent, grade=spec)
    for frame in (3, 15, 26):
        raw, graded = [synth.picture_pixels(path, frame, box=(150, 140, 24, 24))
                       for path in (plain, out)]
        expected = apply_rgb(raw / 255, spec) * 255
        assert np.abs(graded.astype(float) - expected).mean() < 2
        assert np.abs(graded.astype(float) - raw).mean() > 3
        assert np.abs(graded.astype(float) - apply_rgb(expected / 255, spec) * 255).mean() > 3
        np.testing.assert_allclose(
            synth.picture_pixels(out, frame, box=(82, 102, 8, 8)).astype(float),
            synth.picture_pixels(plain, frame, box=(82, 102, 8, 8)), atol=2)
    assert len(cubes) == 1 and window_cubes == [None, cubes[0]]
    graph = picture_command(result)[picture_command(result).index("-filter_complex") + 1]
    assert graph.count("lut3d=") == 2  # One base pass and one independent window pass.


@pytest.mark.render
def test_film_picture_layers_and_repeatability(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.grade import GradeSpec

    pictures = {**pictures, "title": synth.picture_source(
        tmp_path / "opaque-title.mov", "color=c=0xffcc66:s=200x100:r=30,format=rgba",
        90, alpha=True)}
    root = project(tmp_path / "project", pictures, hold_frames=0)
    layers(root, pictures, monkeypatch, titles=[
        {"id": "agent-title", "path": "title.mov", "x": 80, "y": 100,
         "width": 200, "height": 100, "start_frame": 0, "frames": 90,
         "layer": "front", "kind": "title"},
        {"id": "agent-card", "path": "title.mov", "x": 400, "y": 1400,
         "width": 200, "height": 100, "start_frame": 0, "frames": 90,
         "layer": "front", "kind": "graphic"},
    ])
    plain, plain_result = render(root)
    outputs = []
    for name in ("film-one", "film-two"):
        out = root / name / "picture.mp4"
        result = stage.render_picture(root, out, work_dir=out.parent,
                                      grade=GradeSpec("film", 1, "style"))
        outputs.append(out)
    assert receipt.sha256_file(outputs[0]) == receipt.sha256_file(outputs[1])
    info = ffmpeg.probe(outputs[0])
    assert (info.color_space, info.color_transfer, info.color_primaries, info.pix_fmt) == (
        "bt709", "bt709", "bt709", "yuv420p")
    before, after = (synth.picture_pixels(path, 5) for path in (plain, outputs[0]))
    np.testing.assert_array_equal(before[1820:1860, 80:140], after[1820:1860, 80:140])
    for y, x in ((120, 100), (1420, 420)):
        np.testing.assert_allclose(before[y:y+40, x:x+40].astype(float),
                                   after[y:y+40, x:x+40], atol=2)
    assert np.abs(before[500:600, 500:600].astype(float)
                  - after[500:600, 500:600]).mean() > 3
    assert "grade" not in plain_result.to_dict()
    assert result.to_dict()["grade"]["version"] == 2


@pytest.mark.render
@pytest.mark.parametrize("native", [False, True])
def test_zoom_is_graded_once(tmp_path, pictures, monkeypatch, native):
    from yap_studio.grade import GradeSpec, apply_rgb

    size = "1440x2560" if native else "160x90"
    take = synth.picture_source(tmp_path / "take.mov", f"color=0x20506e:s={size}:r=30", 90)
    cut = synth.picture_source(tmp_path / "cut.mov", "color=0x20506e:s=1080x1920:r=30", 90)
    root = project(tmp_path / "project", {**pictures, "take": take, "cut": cut}, hold_frames=0)
    layers(root, pictures, monkeypatch, titles=[])
    lane(root, [{"id": "agent-zoom", "kind": "zoom", "start": {"time": 0.5},
                 "end": {"time": 1.5}, "peak_scale": 1.08, "center": [0.5, 0.5]}])
    outputs = []
    spec = GradeSpec("film", 1)
    for name, grade in (("none", None), ("film", spec)):
        out = root / name / "picture.mp4"
        result = stage.render_picture(root, out, work_dir=out.parent, grade=grade,
                                      frames=(15, 44))
        fallback = any("rendering zoom from the cut" in warning for warning in result.warnings)
        assert fallback != native
        outputs.append(out)
    for frame in (0, 15, 29):
        raw, graded = [synth.picture_pixels(out, frame, box=(450, 650, 80, 80))
                       for out in outputs]
        expected = apply_rgb(raw / 255, spec)
        error = np.abs(graded.astype(float) - expected * 255).mean()
        twice = np.abs(graded.astype(float) - apply_rgb(expected, spec) * 255).mean()
        missing = np.abs(graded.astype(float) - raw).mean()
        print("zoom", native, frame, "error", error, "missing", missing, "twice", twice)
        assert error < 3  # RGB/YUV rounding and delivery encoding, as for the HLG swatch.
        assert missing > 3 and twice > 3


@pytest.mark.render
def test_grade_compare_motion_layers_and_repeatability(tmp_path, pictures, monkeypatch):
    from yap_studio import grade
    from yap_studio.stages import assemble_grade

    root = project(tmp_path / "project", pictures, hold_frames=0)
    data = json.loads((root / "project.json").read_text())
    data["grade"] = {"look": "film"}
    write_json(root / "project.json", data)
    lane(root, [{"id": "agent-compare", "kind": "grade_compare", "start": {"time": 0.5},
                 "end": {"time": 2.5}}])
    layers(root, pictures, monkeypatch, titles=[])
    monkeypatch.setattr(stage, "CRF", 0)  # Isolate compositor pixels from temporal encode noise.
    original_handle = grade.handle_rgba

    def handle(style):
        width, height, raw = original_handle(style)
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4).copy()
        # Expose a scanline to measure the wipe independently of the opaque divider.
        pixels[390:420, :, 3] = 0
        return width, height, pixels.tobytes()

    monkeypatch.setattr(assemble_grade, "handle_rgba", handle)
    timeline = resolve_timeline(root).require_valid()
    baseline = root / "baseline/picture.mp4"
    spec = grade.GradeSpec("film", 1)
    stage.render_picture(root, baseline, work_dir=baseline.parent, grade=spec,
                         timeline=replace(timeline, elements=()))
    outputs = []
    for name in ("compare-one", "compare-two"):
        out = root / name / "picture.mp4"
        result = stage.render_picture(root, out, work_dir=out.parent, grade=spec)
        outputs.append(out)
    assert receipt.sha256_file(outputs[0]) == receipt.sha256_file(outputs[1])
    assert result.to_dict()["grade_compare"] == [
        {"id": "agent-compare", "start_frame": 15, "end_frame": 75}]
    for frame in (0, 15, 74, 89):
        np.testing.assert_array_equal(synth.picture_pixels(outputs[0], frame),
                                      synth.picture_pixels(baseline, frame))
    plain = synth.picture_pixels(root / "cut/cut.mp4", 0, box=(0, 390, 1080, 30)).mean(axis=0)
    wanted = synth.picture_pixels(baseline, 0, box=(0, 390, 1080, 30)).mean(axis=0)
    for local in (6, 10, 15, 18, 24, 31, 40, 48, 53):
        frame = synth.picture_pixels(outputs[0], 15 + local)
        strip = frame[390:420].mean(axis=0)
        is_raw = np.abs(strip-plain).mean(axis=-1) < np.abs(strip-wanted).mean(axis=-1)
        boundary = int(np.count_nonzero(is_raw))
        expected = round(1080 * grade.compare_position(local, 60))
        assert abs(boundary-expected) <= 1, (local, boundary, expected)
        knob_y = round(1920 * 0.66)
        knob = np.all(frame[knob_y-40:knob_y+40] > 220, axis=-1)
        knob[:, expected-4:expected+4] = False
        ys, xs = np.nonzero(knob)
        assert len(xs) > 500
        assert abs(xs.mean()-expected) <= 2, (local, xs.mean(), expected)
        measured_y = ys.mean() + knob_y - 40
        assert abs(measured_y - 1920 * 0.66) <= 2, (local, measured_y)
        print("compare", local, "boundary", boundary, "knob", xs.mean(), measured_y,
              "expected", expected, 1920 * 0.66)
        np.testing.assert_array_equal(frame[1820:1860, 80:140],
                                      synth.picture_pixels(baseline, 15+local)[1820:1860, 80:140])


@pytest.mark.render
@pytest.mark.parametrize("kind", ["overlay", "background",
                                pytest.param("hlg", marks=pytest.mark.helpers)])
def test_film_media_branch_colour(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    from yap_studio.grade import GradeSpec, apply_rgb

    if kind == "hlg" and not hdr.videotoolbox_available():
        pytest.skip("Shared hardware HDR conversion is unavailable")
    root = project(tmp_path / "project", pictures, hold_frames=0)
    layers(root, pictures, monkeypatch, titles=[])
    if kind == "hlg":
        media = synth.picture_source(tmp_path / "hlg.mov", "color=0x20506e:s=160x90:r=30",
                                      90, transfer="arib-std-b67")
        item = insert(root, media, 0, 90)
        box = (450, 650, 40, 40)
    else:
        media = overlay_image(tmp_path / "colour.png", "color=0x20506e:s=320x160")
        item = overlay(root, media, motion="none")
        box = (450, 650, 40, 40)
        if kind == "background":
            item["kind"] = "background"
            for key in ("box", "placement", "motion"):
                item.pop(key, None)
            matte_folder = root / "synthetic-matte"
            synthetic_matte(root / "cut/cut.mp4", matte_folder, box=(100, 100, 100, 100))
            monkeypatch.setattr(stage, "ensure_cut_matte",
                                lambda *args, **kwargs: matte_folder / "matte.mkv")
    lane(root, [item], name="broll")
    plain, _ = render(root)
    out = root / "film/picture.mp4"
    result = stage.render_picture(root, out, work_dir=out.parent, grade=GradeSpec("film", 1))
    before = synth.picture_pixels(plain, 45, box=box)
    after = synth.picture_pixels(out, 45, box=box)
    expected = apply_rgb(before / 255, GradeSpec("film", 1)) * 255
    error = float(np.abs(after.astype(float) - expected).mean())
    # The HLG swatch is uniform even at its centre. Its 8-bit reference and
    # the 16-bit LUT path include RGB/YUV rounding, then delivery encoding.
    tolerance = 3 if kind == "hlg" else 2
    print(f"{kind} grade RGB error: {error:.4f}, margin: {tolerance - error:.4f}")
    assert error < tolerance
    assert np.abs(before.astype(float) - after).mean() > 2
    assert ffmpeg.probe(out).color_transfer == "bt709"
    if kind == "hlg":
        assert result.conversions[0]["kind"] == "hlg"


def test_look_and_legacy_backgrounds_share_the_active_grade(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.grade import GradeSpec

    root = project(tmp_path / "project", pictures, hold_frames=0)
    layers(root, pictures, monkeypatch, titles=[])
    image = overlay_image(tmp_path / "colour.png", "color=0x20506e:s=320x180")
    lane(root, [
        {**insert(root, image, 3, 21, identifier="broll-look"), "kind": "background",
         "fit": "cover", "layout": "bottom", "blend": "wrap"},
        {**insert(root, image, 27, 45, identifier="broll-legacy"), "kind": "background",
         "fit": "cover"},
    ], name="broll")
    matte = root / "synthetic-matte"
    synthetic_matte(root / "cut/cut.mp4", matte, box=(300, 300, 300, 300))
    monkeypatch.setattr(stage, "ensure_cut_matte", lambda *args, **kwargs: matte / "matte.mkv")
    out = root / "film/picture.mp4"
    stage.render_picture(root, out, work_dir=out.parent, grade=GradeSpec("film", 1),
                         frames=(0, 47))
    # This region is clear of the original and scaled person mattes.
    look, legacy = [synth.picture_pixels(out, frame, box=(800, 200, 40, 40)).mean(axis=(0, 1))
                    for frame in (12, 36)]
    ungraded = synth.picture_pixels(image, 0, box=(80, 40, 40, 40)).mean(axis=(0, 1))
    np.testing.assert_allclose(look, legacy, atol=2, rtol=0)
    assert np.abs(look - ungraded).mean() > 2
    assert np.abs(legacy - ungraded).mean() > 2


def overlay_image(out: Path, lavfi: str = "color=red:s=320x160") -> Path:
    """Make a lavfi still without a renderer."""
    ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", lavfi,
                "-frames:v", "1", "-threads", "1", str(out)], timeout_s=60)
    return out


def heic_quadrants(out: Path, *, width: int = 1280, height: int = 720,
                   colour: str = "red") -> Path:
    """Encode four large color regions as a real tiled HEIC still."""
    if shutil.which("sips") is None:
        pytest.skip("sips is needed to generate a tiled HEIC still")
    png = overlay_image(
        out.with_suffix(".png"),
        f"color={colour}:s={width}x{height},"
        f"drawbox=x={width // 2}:y=0:w={width // 2}:h={height // 2}:c=green:t=fill,"
        f"drawbox=x=0:y={height // 2}:w={width // 2}:h={height // 2}:c=blue:t=fill,"
        f"drawbox=x={width // 2}:y={height // 2}:"
        f"w={width // 2}:h={height // 2}:c=yellow:t=fill",
    )
    subprocess.run(["sips", "-s", "format", "heic", str(png), "--out", str(out)],
                   check=True, capture_output=True)
    return out


def rotated_heic_quadrants(out: Path) -> Path:
    """Write a tiled HEIC with the orientation and colour tags of a phone photo."""
    if shutil.which("swift") is None:
        pytest.skip("swift is needed to generate an oriented HEIC still")
    png = overlay_image(out.with_suffix(".png"), "color=red:s=1280x720,"
                        "drawbox=x=640:y=0:w=640:h=360:c=green:t=fill,"
                        "drawbox=x=0:y=360:w=640:h=360:c=blue:t=fill,"
                        "drawbox=x=640:y=360:w=640:h=360:c=yellow:t=fill")
    subprocess.run(["swift", str(Path(__file__).parent / "fixtures/heic_fixture.swift"),
                    str(png), str(out), "6"], check=True, capture_output=True)
    return out


def preparation_commands(result: stage.PictureResult) -> list[tuple[str, ...]]:
    return [command for command in result.commands
            if "-c:v" in command and any(Path(arg).name == "frame.mkv"
                                         and command[index - 1] != "-i"
                                         for index, arg in enumerate(command) if index)]


def picture_command(result: stage.PictureResult) -> tuple[str, ...]:
    return next(command for command in result.commands
                if "-filter_complex" in command and Path(command[-1]).suffix == ".mp4")


def still_hash(source: stage._Source) -> str:
    graph = stage._Graph()
    filters = [getattr(source, "params", None),
               source.hdr.filter_prefix if source.hdr else None]
    branch = graph.filter(graph.input(source), ",".join(filter(None, filters)) or "null")
    result = ffmpeg.run([
        "ffmpeg", "-v", "error", "-filter_complex_threads", "1", *graph.inputs,
        "-filter_complex", ";".join(graph.filters), "-map", f"[{branch}]", "-an",
        "-frames:v", "1", "-f", "framehash", "-",
    ], timeout_s=60)
    return result.stdout.strip().splitlines()[-1].rsplit(",", 1)[-1].strip()


def still_properties(source: stage._Source) -> dict[str, str]:
    graph = stage._Graph()
    filters = [getattr(source, "params", None),
               source.hdr.filter_prefix if source.hdr else None, "showinfo"]
    branch = graph.filter(graph.input(source), ",".join(filter(None, filters)))
    result = ffmpeg.run([
        "ffmpeg", "-v", "info", "-filter_complex_threads", "1", *graph.inputs,
        "-filter_complex", ";".join(graph.filters), "-map", f"[{branch}]", "-an",
        "-frames:v", "1", "-f", "null", "-",
    ], timeout_s=60)
    import re

    fields = ("fmt", "cl", "color_range", "color_space", "color_primaries", "color_trc")
    return {field: re.findall(rf"\b{field}:(\S+)", result.stderr)[-1] for field in fields}


def test_many_heic_inserts_open_one_simple_input_each(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = ffmpeg.tool_versions
    version_calls = []

    def tool_versions() -> dict[str, str | None]:
        version_calls.append(True)
        return versions()

    monkeypatch.setattr(ffmpeg, "tool_versions", tool_versions)
    root = project(tmp_path / "project", pictures, hold_frames=0)
    sources = []
    for index in range(3):
        image = heic_quadrants(tmp_path / f"photo-{index}.heic", width=4032, height=3024,
                               colour=("red", "lime", "blue")[index])
        sources.append(image)
    items = []
    for index in range(30):
        item = insert(root, sources[index % 3], index * 3, index * 3 + 3,
                      identifier=f"broll-{index}", fit="crop", crop=[0, 0, 2016, 1512])
        item["motion"] = "still"
        items.append(item)
    lane(root, items, name="broll")
    layers(root, pictures, monkeypatch, titles=[])
    out, result = render(root)
    encode = picture_command(result)
    assert ":g:0" not in " ".join(encode)
    assert not any(arg.lower().endswith((".heic", ".heif")) for arg in encode)
    assert encode.count("-i") == 32
    prep = preparation_commands(result)
    assert len(prep) == 3
    for command in prep:
        assert command.count("-filter_complex") == 1
        assert command[command.index("-pix_fmt") + 1] == "+"
        assert not {"-color_range", "-colorspace", "-color_primaries", "-color_trc",
                    "-chroma_sample_location"}.intersection(command)
    assert len(version_calls) == 1
    for index in range(30):
        assert_colour(synth.picture_pixels(out, index * 3 + 1, box=(400, 400, 20, 20)),
                      ((255, 0, 0), (0, 255, 0), (0, 0, 255))[index % 3], tolerance=25)


@pytest.mark.parametrize("case", [
    "crop-push", "single-contain-blur", "overlay", "background",
    pytest.param("rotated-grid", marks=pytest.mark.helpers), "stream-tags",
])
def test_prepared_heic_frames_are_exact(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    root = project(tmp_path / "project", pictures, hold_frames=0)
    single = case == "single-contain-blur"
    width, height = (320, 160) if single else (1280, 720)
    source = rotated_heic_quadrants(tmp_path / "still.heic") if case == "rotated-grid" else \
        heic_quadrants(tmp_path / "still.heic", width=width, height=height)
    if case == "rotated-grid":
        width, height = 720, 1280
        info = stage._source(source, [], [], owner="photo")
        assert info.grid and info.info.rotation == 270
        assert (info.info.width, info.info.height) == (width, height)
    if case == "stream-tags":
        probe = stage._probe

        def stream_tags(path: Path, commands: list[tuple[str, ...]]) -> ffmpeg.MediaInfo:
            info = probe(path, commands)
            return replace(info, color_primaries="smpte432", color_transfer="bt709") \
                if path.suffix.lower() == ".heic" else info

        monkeypatch.setattr(stage, "_probe", stream_tags)
    item = insert(root, source, 0, 12, fit="contain" if single else "crop",
                  **({"blur": [[401, 503, 121, 97]]} if single else
                     {"crop": [40, 20, width - 80, height - 40]}))
    if case in {"overlay", "background"}:
        item["media"] = {key: value for key, value in item["media"].items()
                         if key not in {"fit", "crop"}}
        item.update(kind=case, fit="contain")
        if case == "overlay":
            item.update(placement="box", box=[100, 400, 800, 700], motion="drift")
        else:
            item.update(start={"word": "t1:0"}, end={"word": "t1:0", "offset_s": 0.2,
                                                     "edge": "end"})
            monkeypatch.setattr(matte_cache, "make_matte",
                                lambda cut, folder, *, joins=(): synthetic_matte(
                cut, folder, box=(80, 100, 100, 100), joins=joins))
    if case == "background":
        fragment = write_document(root / "fragment.json", {
            "schema": "yap-studio/timeline@1", "elements": [item],
        })
        add_timeline(root, fragment, lane="broll").require_valid()
    else:
        lane(root, [item], name="broll")
    layers(root, pictures, monkeypatch, titles=[])
    prepared, result = render(root)
    assert len(preparation_commands(result)) == 1
    entry = json.loads((root / ".yap/cache/stills" / result.stills[0]["cache"] /
                        "receipt.json").read_bytes())
    assert entry["properties"]["pix_fmt"] == "yuvj420p"
    assert len(entry["pixel_sha256"]) == 64
    if case in {"stream-tags", "rotated-grid"}:
        assert entry["properties"]["color_primaries"] == "unknown"
        assert entry["properties"]["color_trc"] == "unknown"
        assert entry["properties"]["chroma_location"] == "unknown"
    monkeypatch.setattr(stage, "_prepare_still", lambda source, *args, **kwargs: source)
    direct = root / "direct/picture.mp4"
    stage.render_picture(root, direct, work_dir=direct.parent)
    assert synth.picture_hashes(prepared) == synth.picture_hashes(direct)


def test_heic_cache_reuse_source_change_and_project_root(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path / "snapshot", pictures, hold_frames=0)
    real = tmp_path / "real project"
    real.mkdir()
    source = heic_quadrants(tmp_path / "photo.heic")
    item = insert(root, source, 0, 12)
    item["motion"] = "still"
    lane(root, [item], name="broll")
    layers(root, pictures, monkeypatch, titles=[])

    def cached_render(index: int) -> stage.PictureResult:
        work = root / f"work-{index}"
        return stage.render_picture(root, work / "picture.mp4", work_dir=work,
                                    matte_project_dir=real)

    first, second = cached_render(1), cached_render(2)
    record = first.to_dict()["stills"][0]
    assert record["input"] == "broll-one" and record["reused"] is False
    assert second.to_dict()["stills"] == [{**record, "reused": True}]
    assert not preparation_commands(second)
    assert not (root / ".yap/cache/stills").exists()
    folder = real / ".yap/cache/stills" / record["cache"]
    entry = json.loads((folder / "receipt.json").read_bytes())
    assert entry["schema"] == "yap-studio/still-cache@1"
    assert entry["source_sha256"] == item["media"]["sha256"]
    photo = root / item["media"]["path"]
    photo.write_bytes(photo.read_bytes() + b"changed")
    item["media"]["sha256"] = receipt.sha256_file(photo)
    lane(root, [item], name="broll")
    third = cached_render(3)
    assert third.to_dict()["stills"][0]["cache"] != record["cache"]
    assert len(preparation_commands(third)) == 1
    cached = next(folder.glob("*.mkv"))
    cached.write_bytes(b"tampered")
    photo.write_bytes(source.read_bytes())
    item["media"]["sha256"] = receipt.sha256_file(photo)
    lane(root, [item], name="broll")
    with pytest.raises(stage.PictureError) as caught:
        cached_render(4)
    assert str(folder) in str(caught.value)
    assert "delete" in str(caught.value).lower() and "run again" in str(caught.value)


@pytest.mark.parametrize("format_filter", [
    "", ",format=yuv420p", ",scale=out_range=tv,format=yuv420p", ",format=nv12,format=yuv420p",
])
def test_heic_hdr_conversion_runs_once_in_preparation(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
    format_filter: str,
) -> None:
    root = project(tmp_path / "project", pictures, hold_frames=0)
    photo = heic_quadrants(tmp_path / "photo.heic")
    lane(root, [insert(root, photo, 0, 12)], name="broll")
    layers(root, pictures, monkeypatch, titles=[])
    prefix = f"hflip{format_filter},setparams=color_primaries=bt709:"
    prefix += "color_trc=bt709:colorspace=bt709"
    monkeypatch.setattr(stage, "conversion", lambda info, **kwargs:
                        hdr.HdrConversion("fake", ["-threads", "1"], prefix)
                        if Path(info.path).suffix == ".heic" else None)
    out, result = render(root)
    prep = preparation_commands(result)
    assert len(prep) == 1 and "-threads" in prep[0]
    assert prep[0].count("-filter_complex") == 1
    for expression in prefix.split(","):
        assert " ".join(prep[0]).count(expression) == 1
    assert prefix not in " ".join(picture_command(result))
    direct = stage._source(root / "broll-one.heic", [], [], owner="direct")
    prepared = stage._prepare_still(direct, root, [], [], owner=resolve_timeline(root).elements[0],
                                     source_sha256=receipt.sha256_file(direct.path),
                                     ffmpeg_version=ffmpeg.tool_versions()["ffmpeg"])
    assert still_hash(prepared) == still_hash(direct)
    assert still_properties(prepared) == still_properties(direct)
    monkeypatch.setattr(stage, "_prepare_still", lambda source, *args, **kwargs: source)
    direct_out = root / "direct/picture.mp4"
    stage.render_picture(root, direct_out, work_dir=direct_out.parent)
    assert synth.picture_hashes(out) == synth.picture_hashes(direct_out)


@pytest.mark.parametrize("damage", [
    "receipt", "key", "source_sha256", "sha256", "frame", "properties", "pixel_sha256",
])
def test_still_cache_refuses_incomplete_entries(tmp_path: Path, damage: str) -> None:
    photo = heic_quadrants(tmp_path / "photo.heic", width=320, height=160)
    source = stage._source(photo, [], [], owner="broll-photo")
    item = ResolvedElement("broll-photo", "insert", "broll", 0, 3, "front", {})
    prepared = stage._prepare_still(source, tmp_path, [], [], owner=item,
                                     source_sha256=receipt.sha256_file(photo),
                                     ffmpeg_version=ffmpeg.tool_versions()["ffmpeg"])
    folder = prepared.path.parent
    if damage == "receipt":
        (folder / "receipt.json").unlink()
    elif damage == "frame":
        prepared.path.unlink()
    else:
        record = json.loads((folder / "receipt.json").read_bytes())
        record[damage] = "changed"
        (folder / "receipt.json").write_text(json.dumps(record))
    with pytest.raises(stage.PictureError) as caught:
        stage._prepare_still(source, tmp_path, [], [], owner=item,
                              source_sha256=receipt.sha256_file(photo),
                              ffmpeg_version=ffmpeg.tool_versions()["ffmpeg"])
    assert str(folder) in str(caught.value) and "Delete it and run again" in str(caught.value)


def test_still_cache_accepts_concurrent_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    photo = heic_quadrants(tmp_path / "photo.heic", width=320, height=160)
    source = stage._source(photo, [], [], owner="broll-photo")
    item = ResolvedElement("broll-photo", "insert", "broll", 0, 3, "front", {})

    def winner(staging: Path, folder: Path) -> None:
        assert staging.parent == folder.parent == tmp_path / ".yap/cache/stills"
        shutil.copytree(staging, folder)
        raise FileExistsError()

    monkeypatch.setattr(stage.os, "rename", winner)
    stills: list[dict[str, Any]] = []
    prepared = stage._prepare_still(source, tmp_path, [], stills, owner=item,
                                     source_sha256=receipt.sha256_file(photo),
                                     ffmpeg_version=ffmpeg.tool_versions()["ffmpeg"])
    assert prepared.path.is_file() and stills[0]["reused"] is True
    assert not list(prepared.path.parent.parent.glob(".still-*"))


def test_still_preparation_refuses_changed_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    photo = heic_quadrants(tmp_path / "photo.heic", width=320, height=160)
    source = stage._source(photo, [], [], owner="broll-photo")
    item = ResolvedElement("broll-photo", "insert", "broll", 0, 3, "front", {})
    run = ffmpeg.run

    def alter(command: list[str], **kwargs: Any):
        if "-filter_complex" in command and any(
            Path(arg).name == "frame.mkv" and command[index - 1] != "-i"
            for index, arg in enumerate(command) if index
        ):
            command = list(command)
            mapping = command.index("-map") + 1
            label = command[mapping]
            script = max(index for index, arg in enumerate(command) if arg == "-filter_complex")
            command[script + 1] += f";{label}negate[altered]"
            command[mapping] = "[altered]"
        return run(command, **kwargs)

    monkeypatch.setattr(ffmpeg, "run", alter)
    with pytest.raises(stage.PictureError, match="photo.heic.*could not be prepared exactly"):
        stage._prepare_still(source, tmp_path, [], [], owner=item,
                              source_sha256=receipt.sha256_file(photo),
                              ffmpeg_version=ffmpeg.tool_versions()["ffmpeg"])
    assert not list((tmp_path / ".yap/cache/stills").iterdir())


@pytest.mark.parametrize("line", [
    "[showinfo@still @ 0x1] n: 0 fmt:yuvj420p cl:unspecified",
    "[showinfo@still @ 0x1] color_range:pc color_space:smpte170m "
    "color_primaries:unknown color_trc:unknown",
    "[showinfo@still @ 0x1] n: 0 fmt:yuvj420p cl:",
])
def test_still_preparation_refuses_unreadable_frame_properties(line: str) -> None:
    with pytest.raises(stage.PictureError, match="photo.heic"):
        stage._decoded_properties(line, Path("photo.heic"))


def test_still_preparation_allows_large_grid_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    photo = heic_quadrants(tmp_path / "photo.heic", width=320, height=160)
    source = stage._source(photo, [], [], owner="broll-photo")
    item = ResolvedElement("broll-photo", "insert", "broll", 0, 3, "front", {})
    run = ffmpeg.run
    timeouts = []

    def timed(command: list[str], **kwargs: Any):
        if "-filter_complex" in command and any(
            Path(arg).name == "frame.mkv" and command[index - 1] != "-i"
            for index, arg in enumerate(command) if index
        ):
            timeouts.append(kwargs["timeout_s"])
        return run(command, **kwargs)

    monkeypatch.setattr(ffmpeg, "run", timed)
    stage._prepare_still(source, tmp_path, [], [], owner=item,
                          source_sha256=receipt.sha256_file(photo),
                          ffmpeg_version=ffmpeg.tool_versions()["ffmpeg"])
    assert timeouts and all(timeout >= 120 for timeout in timeouts)


@pytest.mark.parametrize("pixel_format,rotation", [("yuv420p10le", 0), ("yuv444p", 90)])
def test_prepared_ffv1_preserves_format_tags_and_upright_pixels(
    tmp_path: Path, pixel_format: str, rotation: int,
) -> None:
    movie = tmp_path / "source.mov"
    ffmpeg.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x160:r=30",
        "-frames:v", "1", "-c:v", "libx264", "-qp", "0", "-threads", "1",
        "-pix_fmt", pixel_format, "-color_range", "tv", "-colorspace", "bt709",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-chroma_sample_location", "left",
        str(movie),
    ], timeout_s=60)
    if rotation:
        rotated = tmp_path / "rotated.mov"
        ffmpeg.run(["ffmpeg", "-v", "error", "-display_rotation", str(rotation),
                    "-i", str(movie), "-map", "0:v:0", "-c", "copy", str(rotated)], timeout_s=60)
        movie = rotated
    source = replace(stage._source(movie, [], [], owner="photo"), still=True, image=True)
    assert source.info.rotation == rotation
    item = ResolvedElement("broll-photo", "insert", "broll", 0, 3, "front", {})
    commands: list[tuple[str, ...]] = []
    prepared = stage._prepare_still(source, tmp_path, commands, [], owner=item,
                                     source_sha256=receipt.sha256_file(movie),
                                     ffmpeg_version=ffmpeg.tool_versions()["ffmpeg"])
    assert any("ffv1" in command for command in commands)
    assert prepared.info.rotation == 0
    assert (prepared.info.width, prepared.info.height) == (source.info.width, source.info.height)
    assert still_properties(prepared) == still_properties(source)
    assert still_hash(prepared) == still_hash(source)


@pytest.mark.parametrize("stderr,known,resource", [
    ("[vist#2:27/hevc @ 0x1] [dec:hevc @ 0x2] Error while opening decoder: "
     "Resource temporarily unavailable", True, True),
    ("[in#2/hevc] Cannot allocate memory", True, True),
    ("Error opening input #2: Too many open files", True, True),
    ("input #2 is broken", True, False),
    ("input #99 is broken", False, False),
    ("unknown encoder failure", False, False),
])
@pytest.mark.parametrize("frames", [None, (3, 11)])
def test_picture_failure_names_input_and_fix(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
    stderr: str, known: bool, resource: bool,
    frames: tuple[int, int] | None,
) -> None:
    import shlex

    root = project(tmp_path / "project with spaces", pictures, hold_frames=0)
    lane(root, [insert(root, pictures["insert"], 0, 12, identifier="broll-first"),
                insert(root, pictures["insert"], 12, 24, identifier="broll-second")],
         name="broll")
    layers(root, pictures, monkeypatch, titles=[])
    original = ffmpeg.run

    def fail(command: list[str], **kwargs: Any):
        if "-filter_complex" in command and "libx264" in command:
            raise ProcessingError("ffmpeg", 1, stderr)
        return original(command, **kwargs)

    monkeypatch.setattr(ffmpeg, "run", fail)
    with pytest.raises(stage.PictureError) as caught:
        stage.render_picture(root, root / "work/picture.mp4", work_dir=root / "work",
                             frames=frames)
    error = caught.value
    assert error.error_type == "processing_error" and error.code == "ffmpeg_exit_1"
    assert stderr in str(error)
    rerun = f"yap-studio run {shlex.quote(str(root))}"
    if frames:
        rerun += f" --frames {frames[0]}-{frames[1]}"
    assert f"{rerun} again" in error.suggested_action
    if known:
        assert str(error).startswith("broll-second (broll-second.mov)")
        assert "timeline remove --lane broll --id broll-second" in error.suggested_action
        if resource:
            assert "Close other apps" in error.suggested_action
            assert "same element" in error.suggested_action
        else:
            assert "broll-second.mov" in error.suggested_action
    else:
        assert "yap-studio doctor" in error.suggested_action


@pytest.mark.parametrize(("width", "height", "grid"), [
    (1280, 720, True), (320, 160, False),
])
def test_heic_graph_uses_display_size_and_filter_loop(
    tmp_path: Path, width: int, height: int, grid: bool,
) -> None:
    commands: list[tuple[str, ...]] = []
    source = stage._source(heic_quadrants(tmp_path / "still.heic", width=width,
                                         height=height), commands, [], owner="test")
    original = source
    assert (source.info.width, source.info.height, source.grid) == (width, height, grid)
    assert any("-show_stream_groups" in command for command in commands)
    graph = stage._Graph()
    item = ResolvedElement("broll-one", "insert", "broll", 0, 30, "front", {
        "media": {"fit": "crop", "crop": [0, 0, width, height]},
    })
    source = stage._prepare_still(source, tmp_path, commands, [], owner=item,
                                   source_sha256=receipt.sha256_file(source.path),
                                   ffmpeg_version=ffmpeg.tool_versions()["ffmpeg"])
    assert not source.grid and source.still and source.hdr is None
    assert source.info.rotation == 0
    assert (source.info.width, source.info.height) == (width, height)
    assert still_properties(source) == still_properties(original)
    assert still_hash(source) == still_hash(original)
    stage._insert(graph, source, item)
    assert "-loop" not in graph.inputs and "-framerate" not in graph.inputs
    assert graph.filters[0].startswith(f"[0:0]{source.params},loop=loop=-1:size=1,")
    overlay = ResolvedElement("broll-two", "overlay", "broll", 0, 30, "front", {
        "media": {}, "box_px": [0, 0, 400, 400], "fit": "contain", "motion": "none",
    })
    stage._overlay(graph, source, overlay)
    assert ("-loop" not in graph.inputs and
            graph.filters[-1].startswith(f"[1:0]{source.params},loop=loop=-1:size=1,"))


def test_heic_source_refuses_failed_grid_probe(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = Path(shutil.copyfile(pictures["insert"], tmp_path / "photo.heic"))

    def failed_probe(path: Path, options: list[str]) -> dict:
        raise ProcessingError("ffprobe", -1, "timed out", code="ffmpeg_timeout")

    monkeypatch.setattr(catalog, "_probe_json", failed_probe)
    with pytest.raises(stage.PictureError, match="broll-photo.*tile grid could not be read"):
        stage._source(image, [], [], owner="broll-photo")


def test_heic_source_accepts_successful_no_grid_probe(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = Path(shutil.copyfile(pictures["insert"], tmp_path / "single.heic"))
    monkeypatch.setattr(catalog, "_probe_json", lambda path, options: {"stream_groups": []})
    source = stage._source(image, [], [], owner="broll-single")
    assert source.grid is False
    assert (source.info.width, source.info.height) == (320, 180)


@pytest.mark.render
@pytest.mark.parametrize(("width", "height", "grid"), [
    (1280, 720, True), (320, 160, False),
])
def test_heic_insert_crop_renders_all_quadrants(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
    width: int, height: int, grid: bool,
) -> None:
    root = project(tmp_path / "project", pictures, hold_frames=0)
    source = heic_quadrants(tmp_path / "still.heic", width=width, height=height)
    lane(root, [insert(root, source, 0, 90, fit="crop", crop=[0, 0, width, height])],
         name="broll")
    layers(root, pictures, monkeypatch, titles=[])
    out, result = render(root)
    for box, colour in [
        ((100, 200, 20, 20), (255, 0, 0)),
        ((960, 200, 20, 20), (0, 128, 0)),
        ((100, 1700, 20, 20), (0, 0, 255)),
        ((960, 1700, 20, 20), (255, 255, 0)),
    ]:
        # Allow small swatch shifts from delivery color conversion.
        assert_colour(synth.picture_pixels(out, 40, box=box), colour, tolerance=25)
    encode = picture_command(result)
    assert "[1:0]setparams=" in encode[encode.index("-filter_complex") + 1]
    assert ",loop=loop=-1:size=1" in encode[encode.index("-filter_complex") + 1]


def textured_image(out: Path, *, seed: int = 7) -> Path:
    """Write a detailed still with a red marker right of center."""
    pixels = np.random.default_rng(seed).integers(40, 170, (480, 270, 3), dtype=np.uint8)
    pixels[150:330, 200:205] = (255, 0, 0)
    out.write_bytes(b"P6\n270 480\n255\n" + pixels.tobytes())
    return out


def marker_x(path: Path, frame: int) -> int:
    pixels = synth.picture_pixels(path, frame)
    red = (pixels[:, :, 0] > 220) & (pixels[:, :, 1] < 60) & (pixels[:, :, 2] < 60)
    return int(np.argmax(red.sum(axis=0)))


@pytest.mark.render
@pytest.mark.parametrize("direction,sign", [("in", 1), ("out", -1)])
def test_still_image_default_push_moves_without_freeze(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
    direction: str, sign: int,
) -> None:
    root = project(tmp_path / "project", pictures, frames=90, hold_frames=0)
    image = textured_image(tmp_path / "photo.ppm")
    lane(root, [insert(root, image, 0, 30)], name="broll")
    style = json.loads((root / "project.json").read_bytes())
    style["style"] = "style.json"
    write_json(root / "project.json", style)
    write_json(root / "style.json", {"schema_version": 1, "name": "Still push",
                                      "broll": {"still_push": direction, "color_match": "off"}})
    layers(root, pictures, monkeypatch, titles=[])
    out, _ = render(root)
    scan = ffmpeg.run(["ffmpeg", "-v", "info", "-i", str(out),
                       "-vf", "trim=end_frame=30,freezedetect=n=-60dB:d=0.5",
                       "-an", "-f", "null", "-"], timeout_s=120)
    assert "freeze_start" not in scan.stderr
    assert sign * (marker_x(out, 29) - marker_x(out, 0)) >= 8


@pytest.mark.render
def test_insert_still_keeps_consecutive_frames_identical(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path / "project", pictures, frames=90, hold_frames=0)
    image = textured_image(tmp_path / "diagram.ppm")
    lane(root, [{**insert(root, image, 0, 30), "motion": "still"}], name="broll")
    layers(root, pictures, monkeypatch, titles=[])
    out, _ = render(root)
    np.testing.assert_array_equal(synth.picture_pixels(out, 10), synth.picture_pixels(out, 11))


@pytest.mark.render
@pytest.mark.parametrize("motion", [None, "still"])
def test_jpeg_insert_converts_full_range_to_tv(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
    motion: str | None,
) -> None:
    pixels = np.random.default_rng(7).integers(50, 190, (480, 270, 3), dtype=np.uint8)
    pixels[80:160, 80:160] = 128
    pixels[300:380, 80:160] = 255
    source = tmp_path / "photo.ppm"
    source.write_bytes(b"P6\n270 480\n255\n" + pixels.tobytes())
    jpeg = tmp_path / "photo.jpg"
    ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(source), "-frames:v", "1",
                "-c:v", "mjpeg", "-q:v", "2", "-pix_fmt", "yuvj420p",
                "-threads", "1", str(jpeg)], timeout_s=60)
    assert (ffmpeg.probe(jpeg).pix_fmt, ffmpeg.probe(jpeg).color_range) == (
        "yuvj420p", "pc",
    )

    root = project(tmp_path / "project", pictures, frames=90, hold_frames=0)
    item = insert(root, jpeg, 0, 30)
    lane(root, [{**item, **({"motion": motion} if motion else {})}], name="broll")
    layers(root, pictures, monkeypatch, titles=[])
    out, _ = render(root)
    info = ffmpeg.probe(out)
    assert (info.pix_fmt, info.color_range) == ("yuv420p", "tv")
    data = ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(out), "-frames:v", "1",
                       "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1"],
                      timeout_s=60, binary=True).stdout
    luma = np.frombuffer(data, dtype=np.uint8, count=1080 * 1920).reshape(1920, 1080)
    assert abs(float(luma[430:450, 430:450].mean()) - 126) <= 5
    assert abs(float(luma[1350:1370, 430:450].mean()) - 235) <= 5


@pytest.mark.render
def test_insert_push_scale_changes_marker_travel(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = textured_image(tmp_path / "photo.ppm")
    travel = []
    for scale in (1.02, 1.08):
        root = project(tmp_path / f"project-{scale}", pictures, frames=90, hold_frames=0)
        lane(root, [{**insert(root, image, 0, 30), "motion": "push",
                     "push_scale": scale}], name="broll")
        layers(root, pictures, monkeypatch, titles=[])
        out, _ = render(root)
        travel.append(marker_x(out, 29) - marker_x(out, 0))
    assert travel[1] > travel[0] + 8


def overlay(root: Path, source: Path, first: int = 30, last: int = 60,
            **changes: Any) -> dict[str, Any]:
    item = insert(root, source, first, last, identifier="broll-overlay")
    return {**item, "kind": "overlay", "placement": "box", "box": [300, 500, 400, 400],
            "motion": "none", **changes}


@pytest.mark.parametrize("alpha,window", [
    (0, None), (63, None), (64, (36, 55)), (128, (33, 58)), (255, (32, 59)),
])
def test_overlay_alpha_runs_follow_source_alpha_and_fades(
    tmp_path: Path, alpha: int, window: tuple[int, int] | None,
) -> None:
    media = overlay_image(tmp_path / "alpha.png", "color=red:s=80x40,format=rgba,"
                          f"geq=r=255:g=0:b=0:a='{alpha}*lt(X,40)'")
    item = ResolvedElement("broll-photo", "overlay", "broll", 30, 60, "front", {
        "media": {}, "box_px": [101, 503, 80, 40], "motion": "none",
    })
    expected = [] if window is None else [(*window, Box(101, 503, 40, 40))]
    assert stage.overlay_alpha_runs(media, item) == expected


def test_overlay_alpha_runs_keep_empty_clip_frames_separate(tmp_path: Path) -> None:
    media = synth.picture_source(tmp_path / "alpha.mov", "color=red:s=80x40:r=30,"
                                 "format=rgba,geq=r=255:g=0:b=0:"
                                 "a='255*lt(X,40)*not(between(N,10,14))'", 30, alpha=True)
    item = ResolvedElement("broll-photo", "overlay", "broll", 30, 60, "front", {
        "media": {}, "box_px": [101, 503, 80, 40], "motion": "none",
    })
    assert stage.overlay_alpha_runs(media, item) == [
        (32, 40, Box(101, 503, 40, 40)), (45, 59, Box(101, 503, 40, 40)),
    ]


@pytest.mark.parametrize("fit", ["cover", "contain"])
def test_overlay_alpha_runs_match_buffered_drift_and_fade(tmp_path: Path, fit: str) -> None:
    media = overlay_image(tmp_path / "drift.png", "color=red:s=400x200,format=rgba,"
                          "geq=r=255:g=0:b=0:a='128*between(X,50,349)*between(Y,25,174)'")
    item = ResolvedElement("broll-photo", "overlay", "broll", 30, 60, "front", {
        "media": {}, "box_px": [101, 503, 400, 400], "motion": "drift", "fit": fit,
    })
    source = stage._source(media, [], [], owner=item.id)
    graph = stage._Graph()
    alpha = graph.filter(stage._overlay(graph, source, item), "alphaextract")
    data = ffmpeg.run([
        "ffmpeg", "-nostdin", "-v", "error", "-filter_complex_threads", "1",
        *graph.inputs, "-filter_complex", ";".join(graph.filters), "-map", f"[{alpha}]",
        "-an", "-frames:v", "30", "-pix_fmt", "gray", "-fps_mode", "passthrough",
        "-f", "rawvideo", "pipe:1",
    ], timeout_s=60, binary=True).stdout
    pixels = np.frombuffer(data, dtype=np.uint8).reshape(30, 400, 400)
    expected: list[tuple[int, int, Box]] = []
    for local_frame, frame_pixels in enumerate(pixels):
        rows, columns = np.nonzero(frame_pixels >= 64)
        if not rows.size:
            continue
        left, top = int(columns.min()), int(rows.min())
        box = Box(101 + left, 503 + top,
                  int(columns.max()) - left + 1, int(rows.max()) - top + 1)
        frame = item.start_frame + local_frame
        if expected and expected[-1][1] == frame and expected[-1][2] == box:
            expected[-1] = (expected[-1][0], frame + 1, box)
        else:
            expected.append((frame, frame + 1, box))
    assert len({box for _, _, box in expected}) > 1
    assert expected[0][0] > item.start_frame
    assert expected[-1][1] < item.end_frame
    assert stage.overlay_alpha_runs(media, item) == expected


@pytest.fixture
def small_alpha_overlay(tmp_path: Path) -> tuple[Path, ResolvedElement]:
    media = overlay_image(tmp_path / "small.png", "color=red:s=8x6")
    item = ResolvedElement("broll-photo", "overlay", "broll", 30, 36, "front", {
        "media": {}, "box_px": [101, 503, 8, 6], "motion": "none",
    })
    return media, item


def test_overlay_alpha_runs_stream_odd_chunks_with_bounded_buffer(
    small_alpha_overlay: tuple[Path, ResolvedElement], monkeypatch: pytest.MonkeyPatch,
) -> None:
    media, item = small_alpha_overlay
    frame_size = 8 * 6
    pixels = np.zeros((6, 6, 8), dtype=np.uint8)
    pixels[1:3, 1:4, 2:6] = 64
    pixels[3, 1:4, 2:6] = 63
    pixels[4, 2:6, 1:4] = 255
    pixels[5, 1:4, 2:6] = 128
    data = pixels.tobytes()
    chunk_sizes = (1, 17, 101, 29)
    buffered = peak_buffered = measured = 0
    buffers: set[int] = set()
    closed = False
    frombuffer = np.frombuffer

    def measure_frame(buffer: bytearray, *, dtype: Any) -> np.ndarray:
        nonlocal buffered, measured
        assert len(buffer) == frame_size
        buffers.add(id(buffer))
        buffered -= frame_size
        assert buffered >= 0
        measured += 1
        return frombuffer(buffer, dtype=dtype)

    def chunks(cmd: list[str], *, timeout_s: float, chunk_size: int
               ) -> Generator[bytes, None, None]:
        nonlocal buffered, peak_buffered, closed
        offset = index = 0
        try:
            while offset < len(data):
                size = min(chunk_size, chunk_sizes[index % len(chunk_sizes)])
                chunk = data[offset:offset + size]
                buffered += len(chunk)
                peak_buffered = max(peak_buffered, buffered)
                assert buffered <= frame_size + max(chunk_sizes)
                yield chunk
                offset += len(chunk)
                index += 1
        finally:
            closed = True

    monkeypatch.setattr(ffmpeg, "run_stream", chunks)
    monkeypatch.setattr(stage.np, "frombuffer", measure_frame)
    assert stage.overlay_alpha_runs(media, item) == [
        (31, 33, Box(103, 504, 4, 3)),
        (34, 35, Box(102, 505, 3, 4)),
        (35, 36, Box(103, 504, 4, 3)),
    ]
    assert measured == 6
    assert buffered == 0
    assert len(buffers) == 1
    assert peak_buffered <= frame_size + max(chunk_sizes)
    assert closed


@pytest.mark.parametrize("size", [1, 47, 49, 95])
def test_overlay_alpha_runs_reject_partial_frame(
    small_alpha_overlay: tuple[Path, ResolvedElement], monkeypatch: pytest.MonkeyPatch, size: int,
) -> None:
    media, item = small_alpha_overlay
    closed = False

    def chunks(*args: Any, **kwargs: Any) -> Generator[bytes, None, None]:
        nonlocal closed
        try:
            data = bytes(size)
            yield data[:17]
            yield data[17:]
        finally:
            closed = True

    monkeypatch.setattr(ffmpeg, "run_stream", chunks)
    with pytest.raises(stage.PictureError, match="partial frame") as caught:
        stage.overlay_alpha_runs(media, item)
    assert caught.value.message == "broll-photo: overlay alpha decode ended with a partial frame."
    assert closed


def test_overlay_alpha_runs_close_stream_on_measurement_error(
    small_alpha_overlay: tuple[Path, ResolvedElement], monkeypatch: pytest.MonkeyPatch,
) -> None:
    media, item = small_alpha_overlay
    closed = False

    def chunks(*args: Any, **kwargs: Any) -> Generator[bytes, None, None]:
        nonlocal closed
        try:
            yield bytes(8 * 6)
            pytest.fail("Stream continued after a failed frame measurement")
        finally:
            closed = True

    def fail_measurement(pixels: np.ndarray) -> Any:
        raise RuntimeError("Frame measurement failed")

    monkeypatch.setattr(ffmpeg, "run_stream", chunks)
    monkeypatch.setattr(stage.np, "nonzero", fail_measurement)
    with pytest.raises(RuntimeError, match="Frame measurement failed"):
        stage.overlay_alpha_runs(media, item)
    assert closed


@pytest.mark.parametrize("image", [True, False])
@pytest.mark.parametrize("fit", ["cover", "contain"])
def test_overlay_media_fit_fade_window_and_silent_picture(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
    image: bool, fit: str,
) -> None:
    root = project(tmp_path / "project", pictures, hold_frames=0)
    if image:
        media = overlay_image(tmp_path / "still.png")
    else:
        media = tmp_path / "clip.mov"
        ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "color=red:s=320x160:r=30:d=1", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=1", "-c:v", "libx264", "-c:a", "aac",
                    "-threads", "1", str(media)], timeout_s=60)
        assert ffmpeg.probe(media).has_audio
    lane(root, [overlay(root, media, fit=fit)], name="broll")
    layers(root, pictures, monkeypatch, titles=[])
    out, result = render(root)
    for frame in (29, 30, 60):
        assert_colour(synth.picture_pixels(out, frame, box=(450, 650, 20, 20)), (0, 0, 255))
    for frame, alpha in ((31, 1 / 6), (36, 1), (53, 1), (59, 1 / 6)):
        assert_colour(synth.picture_pixels(out, frame, box=(450, 650, 20, 20)),
                      (round(255 * alpha), 0, round(255 * (1 - alpha))), tolerance=10)
        assert_colour(synth.picture_pixels(out, frame, box=(280, 650, 16, 16)), (0, 0, 255))
    assert_colour(synth.picture_pixels(out, 45, box=(450, 520, 20, 20)),
                  (0, 0, 255) if fit == "contain" else (255, 0, 0))
    assert not ffmpeg.probe(out).has_audio
    encode = next(command for command in result.commands if "-c:v" in command)
    if image:
        at = encode.index(str(root / "broll-overlay.png"))
        assert encode[at - 5:at] == ("-loop", "1", "-framerate", "30", "-i")
        assert encode.count("-loop") == 1
    else:
        assert "-loop" not in encode


@pytest.mark.parametrize("fit", ["cover", "contain"])
def test_still_inserts_and_montage_render_exact_windows(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch, fit: str,
) -> None:
    root = project(tmp_path / "project", pictures, hold_frames=0)
    photo = overlay_image(tmp_path / "photo.png", "color=red:s=320x160,"
                          "drawbox=x=0:y=40:w=320:h=80:c=white:t=fill")
    other = overlay_image(tmp_path / "other.png", "color=lime:s=320x160")
    first = insert(root, photo, 30, 60, fit=fit, in_s=100, speed="fit")
    second = insert(root, other, 60, 81, identifier="broll-two", fit=fit)
    first["group"] = second["group"] = "photos"
    lane(root, [first, second], name="broll")
    resolved = resolve_timeline(root).require_valid()
    assert resolved.elements[0].payload["media"]["speed"] == 1
    layers(root, pictures, monkeypatch, titles=[])
    out, result = render(root)
    for frame in (29, 81, 89):
        assert_colour(synth.picture_pixels(out, frame, box=(500, 710, 16, 16)), (0, 0, 255))
    for frame in (30, 31, 45, 59):
        assert_colour(synth.picture_pixels(out, frame, box=(500, 710, 16, 16)),
                      (255, 0, 0) if fit == "contain" else (255, 255, 255))
        assert_colour(synth.picture_pixels(out, frame, box=(500, 960, 16, 16)), (255, 255, 255))
        assert_colour(synth.picture_pixels(out, frame, box=(500, 200, 16, 16)), (255, 0, 0))
    for frame in (60, 80):
        assert_colour(synth.picture_pixels(out, frame, box=(500, 710, 16, 16)), (0, 255, 0))
    assert ffmpeg.probe(out).video_frames == 90
    encode = next(command for command in result.commands if "-c:v" in command)
    assert encode.count("-loop") == 2
    for item in resolved.elements:
        at = encode.index(str(root / item.payload["media"]["path"]))
        assert encode[at - 5:at] == ("-loop", "1", "-framerate", "30", "-i")


@pytest.mark.parametrize("fit", ["cover", "contain"])
def test_overlay_drift_grows_about_centre_inside_box(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch, fit: str,
) -> None:
    root = project(tmp_path / "project", pictures, hold_frames=0)
    height = 400 if fit == "cover" else 200
    media = overlay_image(tmp_path / "pattern.png", f"color=red:s=400x{height},format=rgba,"
                          "geq=r=255:g=0:b=0:a='255*between(X,50,349)*"
                          f"between(Y,50,{height - 51})'")
    lane(root, [overlay(root, media, fit=fit, motion="drift")], name="broll")
    layers(root, pictures, monkeypatch, titles=[])
    out, _ = render(root)
    bounds = []
    for frame in (31, 45, 59):
        pixels = synth.picture_pixels(out, frame, box=(300, 500, 400, 400))
        rows, columns = np.nonzero(pixels[:, :, 0] > pixels[:, :, 0].max() / 2)
        bounds.append((columns.min(), rows.min(), columns.max(), rows.max()))
        assert abs((columns.min() + columns.max()) / 2 - 199.5) < 1.5
        assert abs((rows.min() + rows.max()) / 2 - 199.5) < 1.5
        assert_colour(synth.picture_pixels(out, frame, box=(280, 650, 16, 16)), (0, 0, 255))
    widths = [right - left + 1 for left, _, right, _ in bounds]
    assert widths[0] < widths[1] < widths[2]
    assert widths[2] / widths[0] == pytest.approx(1.04, abs=0.02)
    item = resolve_timeline(root).require_valid().elements[0]
    runs = stage.overlay_alpha_runs(root / item.payload["media"]["path"], item)
    assert (runs[0][0], runs[-1][1]) == (32, 59)
    assert all(first < last for first, last, _ in runs)
    assert all(first[1] == second[0] and first[2] != second[2]
               for first, second in zip(runs, runs[1:], strict=False))
    boxes = {frame: box for first, last, box in runs for frame in range(first, last)}
    assert boxes[36].w < boxes[45].w < boxes[53].w
    assert boxes[36].h < boxes[53].h
    assert boxes[36].h <= boxes[45].h <= boxes[53].h
    for frame in (36, 45, 53):
        pixels = synth.picture_pixels(out, frame, box=(300, 500, 400, 400))
        rows, columns = np.nonzero(pixels[:, :, 0] >= 64)
        measured = Box(300 + int(columns.min()), 500 + int(rows.min()),
                       int(columns.max() - columns.min()) + 1,
                       int(rows.max() - rows.min()) + 1)
        np.testing.assert_allclose(tuple(measured.to_dict().values()),
                                   tuple(boxes[frame].to_dict().values()), atol=2)


def test_overlay_above_restored_person_below_front_title_and_captions(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    plate = synth.picture_source(tmp_path / "person.mov", "color=blue:s=1080x1920:r=30,"
                                 "drawbox=x=80:y=100:w=100:h=100:c=red:t=fill", 90)
    root = project(tmp_path / "project", {**pictures, "cut": plate}, hold_frames=0)
    image = overlay_image(tmp_path / "still.png", "color=lime:s=320x160")
    item = overlay(root, image, 6, 36, fit="contain", box=[80, 100, 100, 100])
    resolved_overlay = ResolvedElement(item["id"], "overlay", "broll", 6, 36, "front",
                                       {**item, "box_px": item["box"]})
    behind = ResolvedElement("agent-card", "graphic", "agent", 6, 36, "behind", {})
    title = synth.picture_source(tmp_path / "opaque.mov",
                                 "color=yellow:s=200x100:r=30,format=rgba", 30, alpha=True)
    layers(root, {**pictures, "title": title}, monkeypatch, titles=[
        {"id": "agent-card", "path": "title.mov", "x": 80, "y": 100,
         "width": 200, "height": 100, "start_frame": 6, "frames": 30,
         "layer": "behind", "kind": "graphic"},
        {"id": "agent-front", "path": "front.mov", "x": 90, "y": 130,
         "width": 10, "height": 10, "start_frame": 6, "frames": 30,
         "layer": "front", "kind": "graphic"},
        {"id": "agent-title", "path": "text.mov", "x": 110, "y": 130,
         "width": 10, "height": 10, "start_frame": 6, "frames": 30,
         "layer": "behind", "kind": "title"},
    ])
    for name, colour in (("front", "magenta"), ("text", "cyan"), ("caption", "yellow")):
        synth.picture_source(root / f"{name}.mov", f"color={colour}:s=10x10:r=30,format=rgba",
                             30, alpha=True)
    monkeypatch.setattr(stage, "captions", lambda _: {"layer": {
        "path": "caption.mov", "x": 130, "y": 130, "width": 10, "height": 10,
        "start_frame": 6, "frames": 30,
    }})
    monkeypatch.setattr(matte_cache, "make_matte", lambda cut, folder, *, joins=(): synthetic_matte(
        cut, folder, box=(80, 100, 100, 100),
    ))
    out = root / "work/picture.mp4"
    timeline = ResolvedTimeline((behind, resolved_overlay), (), (), 90, 0)
    result = stage.render_picture(root, out, work_dir=root / "work",
                                  timeline=timeline)
    for box, colour in (
        ((90, 110, 8, 8), (255, 0, 0)), ((150, 140, 8, 8), (0, 255, 0)),
        ((210, 110, 8, 8), (255, 255, 0)), ((90, 130, 8, 8), (255, 0, 255)),
        ((110, 130, 8, 8), (0, 255, 255)), ((130, 130, 8, 8), (255, 255, 0)),
    ):
        assert_colour(synth.picture_pixels(out, 15, box=box), colour)
    assert not any("skipped" in warning for warning in result.warnings)


def test_overlay_clip_retimes_and_holds_last_decoded_frame(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path / "project", pictures, hold_frames=0)
    clip = synth.picture_source(tmp_path / "colours.mov", "color=red:s=320x160:r=30,"
                                "drawbox=c=lime:t=fill:enable='gte(n,30)'", 60)
    item = overlay(root, clip, 30, 60)
    item["media"].update(in_s=0.5, speed=1.5)
    lane(root, [item], name="broll")
    resolved = resolve_timeline(root).require_valid()
    # Simulate a decoder ending early after timeline duration validation.
    synth.picture_source(root / "short.mov", "color=red:s=320x160:r=30,"
                         "drawbox=c=lime:t=fill:enable='gte(n,30)'", 48)
    media = {**resolved.elements[0].payload["media"], "path": "short.mov",
             "sha256": receipt.sha256_file(root / "short.mov")}
    retained_item = replace(resolved.elements[0], payload={**resolved.elements[0].payload,
                                                          "media": media})
    layers(root, pictures, monkeypatch, titles=[])
    out = root / "work/picture.mp4"
    stage.render_picture(root, out, work_dir=root / "work",
                         timeline=replace(resolved, elements=(retained_item,)))
    assert_colour(synth.picture_pixels(out, 36, box=(450, 650, 20, 20)), (255, 0, 0))
    assert_colour(synth.picture_pixels(out, 45, box=(450, 650, 20, 20)), (0, 255, 0))
    assert_colour(synth.picture_pixels(out, 53, box=(450, 650, 20, 20)), (0, 255, 0))


@pytest.mark.parametrize("frames", [1, 2, 7, 11])
def test_short_overlay_fades_stay_inside_window(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch, frames: int,
) -> None:
    root = project(tmp_path / "project", pictures, hold_frames=0)
    media = overlay_image(tmp_path / "still.png")
    lane(root, [overlay(root, media, 30, 30 + frames)], name="broll")
    layers(root, pictures, monkeypatch, titles=[])
    out, _ = render(root)
    for frame in (29, 30 + frames):
        assert_colour(synth.picture_pixels(out, frame, box=(450, 650, 20, 20)), (0, 0, 255))
    if frames > 1:
        pixels = synth.picture_pixels(out, 30 + frames // 2, box=(450, 650, 20, 20))
        assert pixels[:, :, 0].mean() > 200


def test_transition_window_with_cards_and_captions_on_top(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path / "project", pictures, hold_frames=0)
    layers(root, pictures, monkeypatch, titles=[{
        "id": "agent-card", "path": "title.mov", "x": 80, "y": 100,
        "width": 200, "height": 100, "start_frame": 36, "frames": 15,
        "layer": "front", "kind": "graphic",
    }])
    movie = synth.picture_source(root / "transition.mov", "color=c=red:s=1080x1920:r=30", 12)
    transition = ResolvedElement("agent-transition", "transition", "agent", 36, 48, "front",
                                 {"look": "whip_pan", "frames": 12, "join_frame": 42})
    card = ResolvedElement("agent-card", "graphic", "agent", 36, 51, "front", {})
    resolved = ResolvedTimeline((transition, card), (), (), 90, 0)
    placement = {"id": transition.id, "path": str(movie), "start_frame": 36, "frames": 12}
    out = root / "work/picture.mp4"
    stage.render_picture(root, out, work_dir=root / "work", timeline=resolved,
                         transition_layers=[placement])
    for frame in (35, 48, 60):
        assert_colour(synth.picture_pixels(out, frame, box=(500, 700, 24, 24)), (0, 0, 255))
    for frame in range(36, 48):
        assert_colour(synth.picture_pixels(out, frame, box=(500, 700, 24, 24)), (255, 0, 0))
    assert_colour(synth.picture_pixels(out, 42, box=(80, 1820, 24, 24)), (0, 255, 0))
    assert_colour(synth.picture_pixels(out, 42, box=(100, 120, 24, 24)), (255, 127, 0),
                  tolerance=10)


def test_no_transition_layers_is_byte_identical(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path / "project", pictures, hold_frames=0)
    layers(root, pictures, monkeypatch)
    first, _ = render(root)
    second = root / "second/picture.mp4"
    stage.render_picture(root, second, work_dir=root / "second", transition_layers=[])
    assert first.read_bytes() == second.read_bytes()


def test_transition_missing_or_wrong_placement_refuses_picture(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path / "project", pictures, hold_frames=0)
    layers(root, pictures, monkeypatch, titles=[])
    transition = ResolvedElement("agent-transition", "transition", "agent", 36, 48, "front",
                                 {"look": "whip_pan", "frames": 12, "join_frame": 42})
    resolved = ResolvedTimeline((transition,), (), (), 90, 0)
    with pytest.raises(stage.PictureError, match="match the resolved"):
        stage.render_picture(root, root / "missing/picture.mp4", work_dir=root / "missing",
                             timeline=resolved)
    with pytest.raises(stage.PictureError, match="window differs"):
        stage.render_picture(root, root / "wrong/picture.mp4", work_dir=root / "wrong",
                             timeline=resolved, transition_layers=[{
                                 "id": transition.id, "path": "absent.mov",
                                 "start_frame": 35, "frames": 12,
                             }])


@pytest.mark.parametrize("kind", ["title", "graphic"])
def test_no_behind_graphics_keep_legacy_output_without_matte_work(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    root = project(tmp_path / "project", pictures)
    payload = {"lines": ["Alpha"]} if kind == "title" else {
        "template": "text_tag", "slots": {"text": "Alpha"}, "layer": "front",
    }
    lane(root, [{"id": "agent-title", "kind": kind, "start": {"time": 0},
                 "hold_s": 0.5, **payload}])
    layers(root, pictures, monkeypatch, titles=[{
        "id": "agent-title", "path": "title.mov", "x": 80, "y": 100,
        "width": 200, "height": 100, "start_frame": 0, "frames": 15,
        "layer": "behind" if kind == "title" else "front", "kind": kind,
    }])
    out, result = render(root)
    assert result.behind_layers == () and "behind_layers" not in result.to_dict()
    assert not (root / ".yap/cache/matte").exists()
    # Freeze the pre-cutout recipe as an independent encoded-output reference.
    legacy = (
        "[0:0]trim=end_frame=90,settb=1/30,setpts=N,setsar=1[p0];"
        "[1:0]trim=end_frame=15,setsar=1,settb=1/30,setpts=N+0[p1];"
        "[p0][p1]overlay=x=80:y=100:eof_action=pass:repeatlast=0:format=auto:"
        "enable='between(n,0,14)'[p2];"
        "[2:0]trim=end_frame=90,setsar=1,settb=1/30,setpts=N+0[p3];"
        "[p2][p3]overlay=x=60:y=1800:eof_action=pass:repeatlast=0:format=auto:"
        "enable='between(n,0,89)'[p4];"
        "[p4]trim=end_frame=90,tpad=stop_mode=clone:stop=3,trim=end_frame=93,"
        "settb=1/30,setpts=N,format=yuv420p,"
        "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709[p5]"
    )
    command = list(next(command for command in result.commands if "-c:v" in command))
    assert command[command.index("-filter_complex") + 1] == legacy
    command[0] = "ffmpeg"
    command[-1] = str(root / "work/legacy.mp4")
    ffmpeg.run(command, timeout_s=120)
    assert receipt.sha256_file(out) == receipt.sha256_file(root / "work/legacy.mp4")


def test_behind_graphic_person_pixels_titles_front_layers_captions_and_matte_alignment(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    plate = synth.picture_source(tmp_path / "person.mov", "color=blue:s=1080x1920:r=30,"
                                 "drawbox=x=80:y=100:w=100:h=100:c=red:t=fill", 90)
    root = project(tmp_path / "project", {**pictures, "cut": plate}, hold_frames=0)
    lane(root, [{"id": "agent-card", "kind": "graphic", "layer": "behind",
                 "start": {"time": 0.2}, "hold_s": 1,
                 "template": "text_tag", "slots": {"text": "Alpha"}},
                {"id": "agent-front", "kind": "graphic", "layer": "front",
                 "start": {"time": 0.2}, "hold_s": 1,
                 "template": "text_tag", "slots": {"text": "Front"}},
                {"id": "agent-title", "kind": "title", "start": {"time": 0.2},
                 "hold_s": 1, "lines": ["Front"]}])
    title = synth.picture_source(tmp_path / "opaque.mov",
                                 "color=yellow:s=200x100:r=30,format=rgba,"
                                 "drawbox=x=20:y=60:w=50:h=8:c=black:t=fill,"
                                 "drawbox=x=120:y=60:w=50:h=8:c=black:t=fill", 30, alpha=True)
    layers(root, {**pictures, "title": title}, monkeypatch, titles=[
        {"id": "agent-card", "path": "title.mov", "x": 80, "y": 100,
         "width": 200, "height": 100, "start_frame": 6, "frames": 30,
         "layer": "behind", "kind": "graphic"},
        {"id": "agent-front", "path": "front.mov", "x": 130, "y": 110,
         "width": 40, "height": 30, "start_frame": 6, "frames": 30,
         "layer": "front", "kind": "graphic"},
        {"id": "agent-title", "path": "text.mov", "x": 90, "y": 135,
         "width": 40, "height": 16, "start_frame": 6, "frames": 30,
         "layer": "behind", "kind": "title"},
    ])
    synth.picture_source(root / "front.mov", "color=magenta:s=40x30:r=30,format=rgba",
                         30, alpha=True)
    synth.picture_source(root / "text.mov", "color=cyan:s=40x16:r=30,format=rgba",
                         30, alpha=True)
    monkeypatch.setattr(stage, "captions", lambda _: {"layer": {
        "path": "captions.mov", "x": 90, "y": 160, "width": 100, "height": 80,
        "start_frame": 0, "frames": 90,
    }})
    monkeypatch.setattr(matte_cache, "make_matte", lambda cut, folder, *, joins=(): synthetic_matte(
        cut, folder, box=(80, 100, 100, 100), active_windows=((6, 21),),
    ))
    out, result = render(root)
    for frame in (6, 20):
        assert_colour(synth.picture_pixels(out, frame, box=(90, 110, 20, 20)), (255, 0, 0))
        assert_colour(synth.picture_pixels(out, frame, box=(210, 110, 20, 20)), (255, 255, 0))
        assert_colour(synth.picture_pixels(out, frame, box=(140, 115, 20, 20)), (255, 0, 255))
        assert_colour(synth.picture_pixels(out, frame, box=(100, 140, 20, 8)), (0, 255, 255))
        assert_colour(synth.picture_pixels(out, frame, box=(100, 170, 20, 20)), (0, 255, 0))
    for frame in (21, 35):
        assert_colour(synth.picture_pixels(out, frame, box=(90, 110, 20, 20)), (255, 255, 0))
    assert_colour(synth.picture_pixels(out, 5, box=(210, 110, 20, 20)), (0, 0, 255))
    assert_colour(synth.picture_pixels(out, 36, box=(210, 110, 20, 20)), (0, 0, 255))
    # Exactly half the glyphs are covered, so no frame crosses the strict threshold.
    assert result.behind_layers == ({"id": "agent-card", "start_frame": 6, "frames": 30,
                                     "hidden_frames": 0, "hidden_share": 0.0, "flag": False},)


@pytest.mark.parametrize("behind", [False, True])
def test_card_window_under_chrome_and_restored_person(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch, behind: bool,
) -> None:
    plate = synth.picture_source(tmp_path / "person.mov", "color=blue:s=1080x1920:r=30,"
                                 "drawbox=x=100:y=130:w=20:h=20:c=red:t=fill", 90)
    root = project(tmp_path / "project", {**pictures, "cut": plate}, hold_frames=0)
    lane(root, [{"id": "agent-card", "kind": "graphic", "layer": "behind" if behind else "front",
                 "start": {"time": 0.2}, "hold_s": 1,
                 "template": "text_tag", "slots": {"text": "Alpha"}}])
    chrome = synth.picture_source(
        tmp_path / "chrome.mov", "color=yellow:s=160x120:r=30,format=rgba,"
        "geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':"
        "a='if(between(X,16,143)*between(Y,16,103),0,255)'", 30, alpha=True)
    clip = synth.picture_source(root / "clip.mov", "color=c=lime:s=128x88:r=30", 90)
    window = {"slot": "clip", "index": 0, "x": 96, "y": 116, "width": 128, "height": 88,
              "radius": 12, "start_frame": 6, "frames": 30, "fade_in_frames": 1,
              "fade_out_frames": 1,
              "source": {"kind": "media", "path": clip.name,
                         "sha256": receipt.sha256_file(clip), "in_s": 0, "speed": 1,
                         "focus": [0.5, 0.5], "audio": "mute"}}
    placement = {"id": "agent-card", "path": "title.mov", "x": 80, "y": 100,
                 "width": 160, "height": 120, "start_frame": 6, "frames": 30,
                 "layer": "behind" if behind else "front", "kind": "graphic", "windows": [window]}
    layers(root, {**pictures, "title": chrome}, monkeypatch, titles=[placement])
    if behind:
        monkeypatch.setattr(matte_cache, "make_matte", lambda cut, folder: synthetic_matte(
            cut, folder, box=(100, 130, 20, 20)))
    out, result = render(root)
    assert_colour(synth.picture_pixels(out, 15, box=(150, 140, 16, 16)), (0, 255, 0))
    assert_colour(synth.picture_pixels(out, 15, box=(82, 102, 8, 8)), (255, 255, 0))
    assert_colour(synth.picture_pixels(out, 15, box=(104, 134, 8, 8)),
                  (255, 0, 0) if behind else (0, 255, 0))
    for frame in (5, 36):
        assert_colour(synth.picture_pixels(out, frame, box=(150, 140, 16, 16)), (0, 0, 255))
    command = picture_command(result)
    graph = command[command.index("-filter_complex") + 1]
    assert graph.index("fade=t=in") < graph.index("trim=end_frame=30,setsar=1,settb=1/30")


@pytest.mark.parametrize("media_kind", ["image", "clip"])
def test_background_replaces_room_only_between_word_anchors_and_keeps_person(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
    media_kind: str,
) -> None:
    plate = synth.picture_source(tmp_path / "person.mov", "color=blue:s=1080x1920:r=30,"
                                 "drawbox=x=80:y=100:w=100:h=100:c=red:t=fill", 90)
    root = project(tmp_path / "project", {**pictures, "cut": plate}, hold_frames=0)
    source = (overlay_image(tmp_path / "page.png", "color=lime:s=320x180")
              if media_kind == "image" else
              synth.picture_source(tmp_path / "page.mov", "color=lime:s=320x180:r=30", 60))
    item = insert(root, source, 3, 36, identifier="broll-page")
    item.update(kind="background", fit="cover", start={"word": "t1:0"},
                end={"word": "t1:0", "edge": "end", "offset_s": 1})
    fragment = write_document(root / "fragment.json", {
        "schema": "yap-studio/timeline@1", "elements": [item],
    })
    resolved = add_timeline(root, fragment, lane="broll").require_valid()
    assert (resolved.elements[0].start_frame, resolved.elements[0].end_frame) == (3, 36)
    layers(root, pictures, monkeypatch, titles=[])
    monkeypatch.setattr(matte_cache, "make_matte", lambda cut, folder, *, joins=(): synthetic_matte(
        cut, folder, box=(80, 100, 100, 100),
    ))
    out, _ = render(root)
    for frame, outside in ((2, (0, 0, 255)), (3, (0, 255, 0)),
                           (35, (0, 255, 0)), (36, (0, 0, 255))):
        assert_colour(synth.picture_pixels(out, frame, box=(90, 110, 20, 20)), (255, 0, 0))
        assert_colour(synth.picture_pixels(out, frame, box=(210, 110, 20, 20)), outside)


@pytest.mark.parametrize("cutout", ["auto", "screen", "person"])
def test_chroma_picture_records_key_and_uses_style_request(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch, cutout: str,
) -> None:
    plate = synth.picture_source(tmp_path / "short.mov", "color=blue:s=1080x1920:r=30", 12)
    root = project(tmp_path / "project", {**pictures, "cut": plate}, frames=12, hold_frames=0)
    document = json.loads((root / "project.json").read_bytes())
    document["style"] = "style.json"
    write_json(root / "project.json", document)
    write_json(root / "style.json", {"schema_version": 1, "broll": {"cutout": cutout}})
    image = overlay_image(tmp_path / "page.png", "color=gray:s=320x180")
    lane(root, [{**insert(root, image, 3, 9, identifier="broll-page"),
                 "kind": "background", "fit": "cover"}], name="broll")
    short_captions = synth.picture_source(tmp_path / "short-captions.mov",
                                           "color=lime:s=100x80:r=30,format=rgba", 12,
                                           alpha=True)
    layers(root, {**pictures, "captions": short_captions}, monkeypatch, titles=[])
    caption_layer = {**stage.captions(root)["layer"], "frames": 12}
    monkeypatch.setattr(stage, "captions", lambda _: {"layer": caption_layer})
    backend = {"name": "chroma-key", "screen": "blue", "key": "#0000ff",
               "border_share": 0.91}
    calls = []

    def ensure(project_dir: Path, cut: Path, **kwargs) -> Path:
        calls.append(kwargs.get("cutout", "auto"))
        folder = root / "test-matte"
        data = replace(synthetic_matte(cut, folder, box=(100, 100, 100, 100)), backend=backend)
        write_json(folder / "matte.json", data.to_dict())
        return folder / "matte.mkv"

    monkeypatch.setattr(stage, "ensure_cut_matte", ensure)
    _, result = render(root)
    assert calls == [cutout]
    assert result.to_dict()["cutout"] == backend
    graph = next(command[command.index("-filter_complex") + 1]
                 for command in result.commands if "-filter_complex" in command)
    assert "despill=type=blue:mix=0.5" in graph and "maskedmerge" in graph
    assert "gblur=sigma=4" in graph


def test_chroma_background_despills_rim_and_preserves_large_mint_interior(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.matte import chroma
    from yap_studio.matte.make import make_matte
    from yap_studio.matte.model import load_matte

    plate = tmp_path / "mint.mkv"
    ffmpeg.run([
        "ffmpeg", "-v", "error", "-filter_complex_threads", "1",
        "-f", "lavfi", "-i", "color=0x30b050:s=1080x1920:r=30,format=gbrp",
        "-f", "lavfi", "-i", "color=0xb9d5cc:s=1080x1920:r=30,format=gbrp",
        "-f", "lavfi", "-i", "color=black:s=1080x1920:r=30,format=gray,"
        "drawbox=x=300:y=700:w=480:h=600:c=white:t=fill,format=gray,"
        "lut=c0='if(gt(val,128),255,0)',gblur=sigma=8,format=gray,format=gbrp",
        "-filter_complex", "[0:v][1:v][2:v]maskedmerge,scale=out_color_matrix=bt709,"
        "format=yuv444p,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709",
        "-frames:v", "12", "-c:v", "ffv1", "-pix_fmt", "yuv444p",
        "-threads", "2", str(plate),
    ], timeout_s=60)
    root = project(tmp_path / "project", {**pictures, "cut": plate}, frames=12, hold_frames=0)
    background = overlay_image(tmp_path / "page.png", "color=gray:s=108x192")
    lane(root, [{**insert(root, background, 3, 9, identifier="broll-page"),
                 "kind": "background", "fit": "cover"}], name="broll")
    caption = synth.picture_source(root / "empty-captions.mov",
                                   "color=black@0:s=16x16:r=30,format=rgba", 12, alpha=True)
    monkeypatch.setattr(stage, "graphics", lambda _: {"elements": []})
    monkeypatch.setattr(stage, "captions", lambda _: {"layer": {
        "path": caption.name, "x": 0, "y": 0, "width": 16, "height": 16,
        "start_frame": 0, "frames": 12,
    }})
    monkeypatch.setattr(matte_cache, "make_matte", make_matte)
    out, result = render(root)
    assert result.cutout is not None and result.cutout["screen"] == "green"
    document = next((root / ".yap/cache/matte").glob("*/matte.json"))
    assert load_matte(document).backend["name"] == "chroma-key"
    # Decode before cropping so both frames use the same colour conversion.
    original = synth.picture_pixels(root / "cut/cut.mp4", 4).astype(int)
    restored = synth.picture_pixels(out, 4).astype(int)
    np.testing.assert_allclose(restored[900:1060, 460:620], original[900:1060, 460:620], atol=2)

    monkeypatch.setattr(chroma, "despill_people", lambda graph, people, metadata: people)
    plain = root / "plain/picture.mp4"
    stage.render_picture(root, plain, work_dir=plain.parent)
    before = synth.picture_pixels(plain, 4).astype(int)[900:1060, 760:800]
    after = restored[900:1060, 760:800]
    excess_before = before[:, :, 1] - np.maximum(before[:, :, 0], before[:, :, 2])
    excess_after = after[:, :, 1] - np.maximum(after[:, :, 0], after[:, :, 2])
    assert np.any((after[:, :, 1] < before[:, :, 1] - 3)
                  & (excess_after < excess_before - 3))


def test_background_keeps_soft_matte_person_interior_opaque(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    plate = synth.picture_source(tmp_path / "person.mov", "color=blue:s=1080x1920:r=30,"
                                 "drawbox=x=80:y=100:w=100:h=100:c=red:t=fill", 90)
    root = project(tmp_path / "project", {**pictures, "cut": plate}, hold_frames=0)
    image = overlay_image(tmp_path / "page.png", "color=lime:s=320x180")
    lane(root, [{**insert(root, image, 3, 36, identifier="broll-page"),
                 "kind": "background", "fit": "cover"}], name="broll")
    layers(root, pictures, monkeypatch, titles=[])
    monkeypatch.setattr(matte_cache, "make_matte", lambda cut, folder, *, joins=(): synthetic_matte(
        cut, folder, box=(80, 100, 100, 100), soft_box=(110, 130, 40, 40),
    ))
    out, _ = render(root)
    assert_colour(synth.picture_pixels(out, 3, box=(120, 140, 20, 20)), (255, 0, 0))
    assert_colour(synth.picture_pixels(out, 3, box=(210, 140, 20, 20)), (0, 255, 0))


def test_background_firm_mask_stays_off_a_separate_behind_graphic(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    plate = synth.picture_source(tmp_path / "person.mov", "color=blue:s=1080x1920:r=30,"
                                 "drawbox=x=80:y=100:w=100:h=100:c=red:t=fill", 90)
    root = project(tmp_path / "project", {**pictures, "cut": plate}, hold_frames=0)
    image = overlay_image(tmp_path / "page.png", "color=lime:s=320x180")
    lane(root, [{**insert(root, image, 3, 36, identifier="broll-page"),
                 "kind": "background", "fit": "cover"}], name="broll")
    lane(root, [{"id": "agent-card", "kind": "graphic", "layer": "behind",
                 "start": {"time": 2}, "hold_s": 1,
                 "template": "text_tag", "slots": {"text": "Alpha"}}])
    synth.picture_source(root / "card.mov", "color=yellow:s=200x100:r=30,format=rgba",
                         30, alpha=True)
    layers(root, pictures, monkeypatch, titles=[
        {"id": "agent-card", "path": "card.mov", "x": 80, "y": 100,
         "width": 200, "height": 100, "start_frame": 60, "frames": 30,
         "layer": "behind", "kind": "graphic"},
    ])
    monkeypatch.setattr(matte_cache, "make_matte", lambda cut, folder, *, joins=(): synthetic_matte(
        cut, folder, box=(80, 100, 100, 100), soft_box=(110, 130, 40, 40),
    ))
    out, _ = render(root)
    # The soft matte value 140 keeps the speaker partly see-through over the card.
    assert_colour(synth.picture_pixels(out, 70, box=(120, 140, 20, 20)), (255, 115, 0),
                  tolerance=16)
    assert_colour(synth.picture_pixels(out, 70, box=(90, 110, 15, 15)), (255, 0, 0))
    assert_colour(synth.picture_pixels(out, 20, box=(120, 140, 20, 20)), (255, 0, 0))


def test_background_holds_still_from_first_to_last_frame(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path / "project", pictures, hold_frames=0)
    # A 9:16 page scales exactly to the canvas; its edge stripe ends at x=100.
    image = overlay_image(tmp_path / "page.png", "color=lime:s=108x192,"
                          "drawbox=x=0:y=0:w=10:h=192:c=magenta:t=fill")
    lane(root, [{**insert(root, image, 3, 36, identifier="broll-page"),
                 "kind": "background", "fit": "cover"}], name="broll")
    layers(root, pictures, monkeypatch, titles=[])
    monkeypatch.setattr(matte_cache, "make_matte", lambda cut, folder, *, joins=(): synthetic_matte(
        cut, folder, box=(80, 100, 100, 100),
    ))
    out, _ = render(root)
    for frame in (3, 35):
        assert_colour(synth.picture_pixels(out, frame, box=(86, 900, 8, 20)), (255, 0, 255))
        assert_colour(synth.picture_pixels(out, frame, box=(108, 900, 8, 20)), (0, 255, 0))


@pytest.mark.parametrize("layout,box", [
    ("bottom", (189, 672, 702, 1248)),
    ("corner_left", (0, 1056, 486, 864)),
    ("corner_right", (594, 1056, 486, 864)),
])
def test_green_screen_layout_wipe_and_determinism(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
    layout: str, box: tuple,
) -> None:
    plate = synth.picture_source(tmp_path / "person.mov", "color=blue:s=1080x1920:r=30,"
                                 "drawbox=x=300:y=300:w=300:h=300:c=red:t=fill", 90)
    root = project(tmp_path / "project", {**pictures, "cut": plate}, hold_frames=0)
    image = overlay_image(tmp_path / "page.png", "color=lime:s=108x192")
    lane(root, [{**insert(root, image, 3, 24, identifier="broll-page"),
                 "kind": "background", "fit": "cover", "layout": layout,
                 "enter": "wipe"}], name="broll")
    layers(root, pictures, monkeypatch, titles=[])
    monkeypatch.setattr(matte_cache, "make_matte", lambda cut, folder: synthetic_matte(
        cut, folder, box=(300, 300, 300, 300),
    ))
    out, result = render(root)
    bx, by, width, _ = box
    scale = width / 1080
    moved = (round(bx + scale * 400), round(by + scale * 400), 20, 20)
    assert_colour(synth.picture_pixels(out, 12, box=moved), (255, 0, 0))
    assert_colour(synth.picture_pixels(out, 12, box=(400, 400, 20, 20)), (0, 255, 0))
    assert_colour(synth.picture_pixels(out, 2, box=(400, 400, 20, 20)), (255, 0, 0))
    assert_colour(synth.picture_pixels(out, 24, box=(400, 400, 20, 20)), (255, 0, 0))
    assert_colour(synth.picture_pixels(out, 3, box=(800, 100, 20, 20)), (0, 255, 0))
    assert_colour(synth.picture_pixels(out, 3, box=(800, 1800, 20, 20)), (0, 0, 255))
    assert result.green_screen[0]["person_box_px"] == list(box)
    assert result.to_dict()["green_screen"] == list(result.green_screen)
    digest = receipt.sha256_file(out)
    again = root / "repeat/picture.mp4"
    stage.render_picture(root, again, work_dir=root / "repeat")
    assert receipt.sha256_file(again) == digest


def test_green_screen_timeout_grows_with_window_and_keeps_legacy_budget(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path / "project", pictures, hold_frames=0)
    layers(root, pictures, monkeypatch, titles=[])
    image = overlay_image(tmp_path / "page.png", "color=lime:s=108x192")
    item = {**insert(root, image, 3, 33, identifier="broll-page"),
            "kind": "background", "fit": "cover"}
    matte = root / "synthetic-matte"
    synthetic_matte(root / "cut/cut.mp4", matte, box=(300, 300, 300, 300))
    monkeypatch.setattr(stage, "ensure_cut_matte", lambda *args, **kwargs: matte / "matte.mkv")
    monkeypatch.setattr(stage.green_screen, "measure_pull",
                        lambda *args: {"offset_rgb": [0, 0, 0]})
    timeouts = []
    run = ffmpeg.run

    class EncodeCaptured(Exception):
        pass

    def capture(command: list[str], **kwargs: Any):
        if command[-1] == str(out):
            timeouts.append(kwargs["timeout_s"])
            raise EncodeCaptured
        return run(command, **kwargs)

    monkeypatch.setattr(ffmpeg, "run", capture)
    look = {"layout": "bottom", "blend": "wrap", "enter": "wipe", "motion": "push"}
    for index, (effects, end) in enumerate((({}, 33), ({}, 63), (look, 33), (look, 63))):
        lane(root, [{**item, **effects, "end": {"time": end / 30}}], name="broll")
        out = root / f"encode-{index}/picture.mp4"
        with pytest.raises(EncodeCaptured):
            stage.render_picture(root, out, work_dir=out.parent)
    assert timeouts[0] == timeouts[1]
    assert timeouts[2] - timeouts[0] == pytest.approx(96)
    assert timeouts[3] - timeouts[1] == pytest.approx(192)


def test_background_refuses_without_person_matte(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path / "project", pictures, hold_frames=0)
    image = overlay_image(tmp_path / "page.png")
    lane(root, [{**insert(root, image, 6, 36, identifier="broll-page"),
                 "kind": "background"}], name="broll")
    layers(root, pictures, monkeypatch, titles=[])

    def unavailable(*args: Any, joins=()) -> None:
        raise matte_cache.CutMatteError("person cutout is unavailable")

    monkeypatch.setattr(matte_cache, "make_matte", unavailable)
    with pytest.raises(matte_cache.CutMatteError,
                       match="Background media needs a usable person matte"):
        render(root)


def test_tall_box_refuses_without_person_cutout(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path / "project", pictures, hold_frames=0)
    image = overlay_image(root / "page.png", "color=white:s=620x2000")
    item = ResolvedElement("broll-page", "overlay", "broll", 0, 90, "front", {
        "placement": "tall_behind", "motion": "none", "box_px": [230, 140, 620, 860],
        "media": {"path": image.name, "sha256": receipt.sha256_file(image)}})
    timeline = ResolvedTimeline((item,), (), (), 90, 0)
    layers(root, pictures, monkeypatch, titles=[])

    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise matte_cache.CutMatteError("synthetic cutout failure")

    monkeypatch.setattr(stage, "ensure_cut_matte", unavailable)
    with pytest.raises(matte_cache.CutMatteError,
                       match="The tall box behind your head needs a usable person cutout: "
                             "synthetic cutout failure"):
        stage.render_picture(root, root / "work/picture.mp4", work_dir=root / "work",
                             timeline=timeline)


@pytest.mark.parametrize("fps,nominal", [(29, 30), (30, 60), (24, 24)])
def test_behind_graphic_refuses_bad_cut_rate_before_matte_work(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
    fps: int, nominal: int,
) -> None:
    root = project(tmp_path / "project", pictures)
    lane(root, [{"id": "agent-card", "kind": "graphic", "layer": "behind",
                 "start": {"time": 0}, "hold_s": 0.5,
                 "template": "text_tag", "slots": {"text": "Alpha"}}])
    timeline = resolve_timeline(root).require_valid()
    layers(root, pictures, monkeypatch, titles=[{
        "id": "agent-card", "path": "title.mov", "x": 80, "y": 100,
        "width": 200, "height": 100, "start_frame": 0, "frames": 15,
        "layer": "behind", "kind": "graphic",
    }])
    probe = ffmpeg.probe

    def changed_rate(source: Path, **kwargs: Any) -> ffmpeg.MediaInfo:
        info = probe(source, **kwargs)
        return replace(info, fps=Fraction(fps), nominal_fps=Fraction(nominal)) \
            if source == root / "cut/cut.mp4" else info

    monkeypatch.setattr(ffmpeg, "probe", changed_rate)
    with pytest.raises(matte_cache.CutMatteError, match="constant-rate 30 fps") as caught:
        stage.render_picture(root, root / "work/picture.mp4", work_dir=root / "work",
                             timeline=timeline)
    assert "Rebuild the cut" in caught.value.suggested_action
    assert not (root / ".yap/cache/matte").exists()


@pytest.mark.parametrize("effect", ["zoom", "insert"])
def test_person_restoration_skips_zoom_and_insert_overlap(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch, effect: str,
) -> None:
    root = project(tmp_path / "project", pictures, hold_frames=0)
    behind = {"id": "agent-card", "kind": "graphic", "layer": "behind",
              "start": {"time": 0.2}, "hold_s": 1,
              "template": "text_tag", "slots": {"text": "Alpha"}}
    if effect == "zoom":
        lane(root, [behind, {"id": "agent-zoom", "kind": "zoom", "start": {"time": 0.4},
                            "end": {"time": 0.6}, "peak_scale": 1.08, "center": [0.5, 0.5]}])
    else:
        lane(root, [behind])
        lane(root, [insert(root, pictures["insert"], 12, 18)], name="broll")
    title = synth.picture_source(tmp_path / "opaque.mov",
                                 "color=yellow:s=200x100:r=30,format=rgba,"
                                 "drawbox=x=20:y=40:w=60:h=8:c=black:t=fill,"
                                 "drawbox=x=20:y=60:w=60:h=8:c=black:t=fill", 30, alpha=True)
    layers(root, {**pictures, "title": title}, monkeypatch, titles=[
        {"id": "agent-card", "path": "title.mov", "x": 80, "y": 100,
         "width": 200, "height": 100, "start_frame": 6, "frames": 30,
         "layer": "behind", "kind": "graphic"},
    ])
    monkeypatch.setattr(matte_cache, "make_matte", lambda cut, folder, *, joins=(): synthetic_matte(
        cut, folder, box=(80, 100, 200, 100),
    ))
    resolved = resolve_timeline(root)
    assert {finding.rule for finding in resolved.findings
            if finding.severity == "violation"} == {"visual_overlap"}
    # Assembly's editorial override can permit overlaps; exercise its picture contract.
    timeline = replace(resolved, findings=tuple(finding for finding in resolved.findings
                                               if finding.severity != "violation"))
    out = root / "work/picture.mp4"
    result = stage.render_picture(root, out, work_dir=root / "work", timeline=timeline)
    for frame in (6, 11, 18, 35):
        assert_colour(synth.picture_pixels(out, frame, box=(100, 120, 20, 20)), (0, 0, 255))
    for frame in (12, 17):
        assert_colour(synth.picture_pixels(out, frame, box=(100, 120, 20, 20)), (255, 255, 0))
    assert result.behind_layers == ({"id": "agent-card", "start_frame": 6, "frames": 30,
                                     "hidden_frames": 24, "hidden_share": 0.8, "flag": True},)
    assert any("skipped during zoom and insert" in warning
               for warning in result.warnings)


def test_person_restoration_skips_transition_overlap(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Synthetic transition frames do not reuse a matte aligned to the verified cut."""
    plate = synth.picture_source(tmp_path / "person.mov", "color=blue:s=1080x1920:r=30,"
                                 "drawbox=x=80:y=100:w=100:h=100:c=red:t=fill", 90)
    root = project(tmp_path / "project", {**pictures, "cut": plate}, hold_frames=0)
    card = synth.picture_source(tmp_path / "opaque.mov",
                                "color=yellow:s=200x100:r=30,format=rgba", 30, alpha=True)
    layers(root, {**pictures, "title": card}, monkeypatch, titles=[{
        "id": "agent-card", "path": "title.mov", "x": 80, "y": 100,
        "width": 200, "height": 100, "start_frame": 6, "frames": 30,
        "layer": "behind", "kind": "graphic",
    }])
    monkeypatch.setattr(matte_cache, "make_matte", lambda cut, folder, *, joins=(): synthetic_matte(
        cut, folder, box=(80, 100, 100, 100), active_windows=((6, 36),),
    ))
    movie = synth.picture_source(root / "transition.mov", "color=lime:s=1080x1920:r=30", 12)
    transition = ResolvedElement("agent-transition", "transition", "agent", 12, 24, "front",
                                 {"look": "whip_pan", "frames": 12, "join_frame": 18})
    behind = ResolvedElement("agent-card", "graphic", "agent", 6, 36, "behind", {})
    timeline = ResolvedTimeline((transition, behind), (), (), 90, 0)
    out = root / "work/picture.mp4"
    result = stage.render_picture(root, out, work_dir=root / "work", timeline=timeline,
                                  transition_layers=[{
                                      "id": transition.id, "path": str(movie),
                                      "start_frame": 12, "frames": 12,
                                  }])
    assert_colour(synth.picture_pixels(out, 11, box=(90, 110, 20, 20)), (255, 0, 0))
    assert_colour(synth.picture_pixels(out, 18, box=(90, 110, 20, 20)), (255, 255, 0))
    assert_colour(synth.picture_pixels(out, 24, box=(90, 110, 20, 20)), (255, 0, 0))
    assert any("skipped during transition" in warning
               for warning in result.warnings)


def test_windows_placement_zoom_and_composited_hold(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path / "project", pictures)
    lane(root, [{"id": "agent-title", "kind": "title", "start": {"time": 0},
                 "hold_s": 0.5, "lines": ["Alpha"]},
                {"id": "agent-zoom", "kind": "zoom", "start": {"time": 0.5},
                 "end": {"time": 1.5}, "peak_scale": 1.08, "center": [0.5, 0.5]},
                {"id": "agent-last", "kind": "title", "start": {"time": 2.5},
                 "end": {"time": 3}, "lines": ["Omega"]}])
    lane(root, [insert(root, pictures["insert"], 45, 60)], name="broll")
    title = {"id": "agent-title", "path": "title.mov", "x": 80, "y": 100,
             "width": 200, "height": 100, "start_frame": 0, "frames": 15,
             "layer": "behind", "kind": "title"}
    layers(root, pictures, monkeypatch, titles=[title, {**title, "id": "agent-last",
                                                      "start_frame": 75}])
    out, result = render(root)
    assert result.total_frames == 93
    assert result.hold_frames == 3
    assert any("rendering zoom from the cut" in warning for warning in result.warnings)
    for frame in (0, 14, 15, 44, 60, 89, 92):
        assert_colour(synth.picture_pixels(out, frame, box=(500, 700, 40, 40)), (0, 0, 255))
    for frame in (45, 59):
        assert_colour(synth.picture_pixels(out, frame, box=(500, 700, 40, 40)), (255, 0, 0))
    assert_colour(synth.picture_pixels(out, 0, box=(120, 120, 40, 40)), (126, 126, 128))
    assert_colour(synth.picture_pixels(out, 15, box=(120, 120, 40, 40)), (0, 0, 255))
    assert_colour(synth.picture_pixels(out, 92, box=(80, 1820, 40, 40)), (0, 255, 0))
    assert_colour(synth.picture_pixels(out, 92, box=(120, 120, 40, 40)), (126, 126, 128))
    assert len(set(synth.picture_hashes(out, 89))) == 1
    info = ffmpeg.probe(out)
    assert (info.width, info.height, info.fps, info.pix_fmt) == (1080, 1920, 30, "yuv420p")
    assert (info.color_primaries, info.color_transfer, info.color_space) == ("bt709",) * 3
    assert not info.has_audio
    encode = next(command for command in result.commands if "-c:v" in command)
    assert encode[encode.index("-crf") + 1] == "16"
    assert encode[encode.index("-preset") + 1] == "slow"
    assert "-nostdin" in encode and "-map" in encode
    json.dumps(result.to_dict())


def test_insert_overrun_refused_before_any_layer_render(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path / "project", pictures)
    lane(root, [insert(root, pictures["insert"], 0, 60, in_s=2)], name="broll")
    findings = resolve_timeline(root)
    assert any(finding.rule == "insert_bounds" for finding in findings.findings)

    def unexpected(_: Path) -> dict[str, Any]:
        pytest.fail("Invalid media reached a layer render")

    monkeypatch.setattr(stage, "graphics", unexpected)
    with pytest.raises(TimelineRefusal, match="exceeds the media"):
        render(root)
    assert not (root / "work/picture.mp4").exists()


@pytest.mark.parametrize("kind", ["zoom", "insert", "still"])
def test_full_range_take_preserves_colours(
    tmp_path: Path, pictures: dict[str, Path], kind: str,
) -> None:
    calibration = synth.calib_source(tmp_path / "calibration.mov", seconds=1,
                                     size="1100x1940" if kind == "zoom" else "320x180")
    source = synth.full_range_source(calibration, tmp_path / "full-range.mov")
    root = project(tmp_path / "project", {**pictures, "take": source}, frames=15, hold_frames=0)
    if kind == "zoom":
        lane(root, [{"id": "agent-zoom", "kind": "zoom", "start": {"time": 0},
                     "end": {"time": 0.5}, "peak_scale": 1.08, "center": [0.5, 0.5]}])
    else:
        if kind == "still":
            image = root / "patches.jpg"
            synth._run(["ffmpeg", "-v", "error", "-i", str(source), "-frames:v", "1",
                        "-q:v", "2", str(image)])
            source = image
        lane(root, [insert(root, source, 0, 15)], name="broll")
    output = root / "work/picture.mp4"
    stage.render_picture(root, output, work_dir=output.parent, title_layers=[], caption_layer={
        "path": str(pictures["title"]), "x": 0, "y": 1800, "width": 200, "height": 100,
        "start_frame": 0, "frames": 15,
    })
    info = synth._stream(output, "video")
    assert (info["pix_fmt"], info["color_range"], info["color_space"]) == (
        "yuv420p", "tv", "bt709",
    )
    np.testing.assert_allclose(synth.patch_colours(output).astype(int),
                               synth.patch_colours(source).astype(int), atol=5)


def test_full_range_cached_still_preserves_colours(tmp_path: Path) -> None:
    calibration = synth.calib_source(tmp_path / "calibration.mov", seconds=1)
    full = synth.full_range_source(calibration, tmp_path / "full.mov")
    source = tmp_path / "still.mkv"
    synth._run(["ffmpeg", "-v", "error", "-i", str(full), "-frames:v", "1",
                "-c:v", "ffv1", "-pix_fmt", "yuv420p", "-color_range", "pc",
                "-c:a", "copy", str(source)])
    owner = ResolvedElement("broll-still", "insert", "broll", 0, 15, "front", {})
    cached = stage._prepare_still(
        stage._source(source, [], [], owner="still"), tmp_path, [], [],
        source_sha256=receipt.sha256_file(source), owner=owner, ffmpeg_version=None,
    )
    graph = stage._Graph()
    branch = graph.filter(graph.input(cached), f"{stage._still_loop(cached)}"
                          f"{stage._prefix(cached)},"
                          "scale=in_color_matrix=bt709:in_range=tv,format=rgb24")
    data = synth._run(["ffmpeg", "-v", "error", "-filter_complex_threads", "1", *graph.inputs,
                       "-filter_complex", ";".join(graph.filters), "-map", f"[{branch}]",
                       "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
    rgb = np.frombuffer(data, dtype=np.uint8).reshape(180, 320, 3)
    patches = np.array([rgb[round(180 * y), 160] for y in (0.2, 0.5, 0.8)])
    np.testing.assert_allclose(patches.astype(int), synth.patch_colours(source).astype(int), atol=3)


def test_native_zoom_peak_centre_and_source_spans(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = synth.picture_source(
        tmp_path / "native.mov", "color=c=black:s=1440x2560:r=30,"
        "drawbox=x=280:y=1160:w=160:h=240:c=red:t=fill,"
        "drawbox=x=1200:y=200:w=100:h=100:c=white:t=fill:enable='gte(t,2)'", 120,
    )
    root = project(tmp_path / "project", {**pictures, "take": source}, hold_frames=0)
    spans = write_document(root / "spans.json", {
        "sources": {"t1": "../takes/t1.mov"}, "fps": "30/1",
        "segments": [{"source": "t1", "start": "2", "end": "7/2"},
                     {"source": "t1", "start": "0", "end": "3/2"}],
    })
    inputs = cut_inputs(ProjectContext(root))
    owned = json.loads((root / "cut/receipt.json").read_bytes())["outputs"]
    with StageRun(root, "cut") as run:
        for name, digest in inputs.items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in owned})
        run.replace_output(root / "cut/spans.json", spans)
        run.finish(extra={"plan": {"total_frames": 90}})
    lane(root, [{"id": "agent-zoom", "kind": "zoom", "start": {"time": 0},
                 "end": {"time": 2}, "peak_scale": 1.08, "center": [0.25, 0.5]}])
    layers(root, pictures, monkeypatch, titles=[])
    out, result = render(root)
    assert not any("from the cut" in warning for warning in result.warnings)
    initial = synth.picture_pixels(out, 0, box=(170, 820, 200, 280))
    peak = synth.picture_pixels(out, 30, box=(170, 820, 200, 280))

    def red_bounds(pixels: np.ndarray) -> tuple[int, int, float, float]:
        """Measure a marker without depending on the renderer's zoom expression."""
        red = (pixels[:, :, 0] > 128) & (pixels[:, :, 1] < 64) & (pixels[:, :, 2] < 64)
        ys, xs = np.where(red)
        return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1), xs.mean(), ys.mean()

    width, height, cx, cy = red_bounds(initial)
    peak_width, peak_height, peak_cx, peak_cy = red_bounds(peak)
    assert peak_width == pytest.approx(width * 1.08, abs=2)
    assert peak_height == pytest.approx(height * 1.08, abs=2)
    assert peak_cx == pytest.approx(cx, abs=1)
    assert peak_cy == pytest.approx(cy, abs=1)
    assert_colour(synth.picture_pixels(out, 0, box=(920, 170, 20, 20)), (255, 255, 255))
    assert_colour(synth.picture_pixels(out, 45, box=(920, 170, 20, 20)), (0, 0, 0))
    assert_colour(synth.picture_pixels(out, 60, box=(250, 900, 20, 20)), (0, 0, 255))


@pytest.mark.parametrize("crop_x", [0.25, 0.75])
def test_native_zoom_uses_take_crop_position(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch, crop_x: float,
) -> None:
    source = synth.picture_source(tmp_path / "square.mov", "color=c=black:s=2160x2160:r=30,"
                                  "drawbox=x=1030:y=1030:w=100:h=100:c=red:t=fill", 120)
    root = project(tmp_path / "project", {**pictures, "take": source}, hold_frames=0)
    data = json.loads((root / "project.json").read_bytes())
    data["takes"][0]["crop_x"] = crop_x
    write_json(root / "project.json", data)
    owned = json.loads((root / "cut/receipt.json").read_bytes())["outputs"]
    inputs = cut_inputs(ProjectContext(root))
    with StageRun(root, "cut") as run:
        for name, digest in inputs.items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in owned})
        run.finish(extra={"plan": {"total_frames": 90}})
    lane(root, [{"id": "agent-zoom", "kind": "zoom", "start": {"time": 0},
                 "end": {"time": 2}, "peak_scale": 1.08, "center": [0.5, 0.5]}])
    layers(root, pictures, monkeypatch, titles=[])
    out, _ = render(root)
    centres = []
    for frame in (0, 30):
        pixels = synth.picture_pixels(out, frame)
        _, xs = np.where((pixels[:, :, 0] > 128) & (pixels[:, :, 1] < 64)
                         & (pixels[:, :, 2] < 64))
        centres.append(xs.mean())
    assert centres[0] == pytest.approx(960 - 840 * crop_x, abs=2)
    assert centres[1] == pytest.approx(540 + (centres[0] - 540) * 1.08, abs=2)


def test_zoom_fallback_needs_no_conversion_of_unused_hdr_take(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = synth.picture_source(tmp_path / "small.mov", "testsrc2=s=160x90:r=30", 120,
                                  transfer="arib-std-b67")
    root = project(tmp_path / "project", {**pictures, "take": source})
    lane(root, [{"id": "agent-zoom", "kind": "zoom", "start": {"time": 0},
                 "end": {"time": 1}, "peak_scale": 1.08, "center": [0.5, 0.5]}])
    layers(root, pictures, monkeypatch, titles=[])

    def unexpected(_: list[str]) -> dict[str, bool]:
        pytest.fail("Fallback zoom should use the already converted cut")

    monkeypatch.setattr(ffmpeg, "has_filters", unexpected)
    out, result = render(root)
    assert result.conversions == ()
    assert any("from the cut" in warning for warning in result.warnings)
    assert_colour(synth.picture_pixels(out, 15, box=(500, 700, 40, 40)), (0, 0, 255))


@pytest.mark.parametrize("fit", ["cover", "contain", "crop"])
def test_insert_fit_and_canvas_blur(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch, fit: str,
) -> None:
    source = synth.picture_source(
        tmp_path / "pattern.mov", "color=c=black:s=1080x960:r=30,"
        "drawbox=x=0:y=0:w=270:h=960:c=red:t=fill,"
        "drawbox=x=270:y=0:w=540:h=960:c=white:t=fill,"
        "drawbox=x=810:y=0:w=270:h=960:c=blue:t=fill,"
        "drawgrid=w=12:h=12:t=2:c=black", 90,
    )
    root = project(tmp_path / "project", pictures, hold_frames=0)
    extra = {"crop": [0, 0, 270, 960]} if fit == "crop" else {}
    lane(root, [insert(root, source, 0, 90, fit=fit, blur=[[470, 840, 120, 120]], **extra)],
         name="broll")
    layers(root, pictures, monkeypatch, titles=[])
    out, _ = render(root)
    inside = synth.picture_pixels(out, 40, box=(490, 860, 80, 80)).astype(float)
    outside = synth.picture_pixels(out, 40, box=(650, 860, 80, 80)).astype(float)
    assert inside.mean(axis=2).std() < outside.mean(axis=2).std() * 0.6
    if fit == "cover":
        swatch = synth.picture_pixels(out, 40, box=(300, 500, 40, 40))
        assert swatch.mean() > 80
        assert swatch[:, :, 0].mean() == pytest.approx(swatch[:, :, 2].mean(), abs=8)
    elif fit == "contain":
        foreground = synth.picture_pixels(out, 40, box=(100, 530, 16, 16))
        fill = synth.picture_pixels(out, 40, box=(100, 200, 16, 16))
        assert foreground[:, :, 0].mean() > foreground[:, :, 2].mean() + 100
        assert fill[:, :, 0].mean() == pytest.approx(fill[:, :, 2].mean(), abs=30)
    else:
        swatch = synth.picture_pixels(out, 40, box=(700, 600, 40, 40))
        assert swatch[:, :, 0].mean() > swatch[:, :, 2].mean() + 100


def test_montage_speed_in_point_and_last_frame_padding(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = synth.picture_source(tmp_path / "sequence.mov",
                                  "color=c=red:s=320x180:r=30,"
                                  "drawbox=w=iw:h=ih:c=lime:t=fill:enable='gte(t,1)'", 90)
    root = project(tmp_path / "project", pictures)
    one = insert(root, source, 0, 30, identifier="broll-one", in_s=0.5, speed=2)
    two = insert(root, pictures["insert"], 30, 60, identifier="broll-two", in_s=2,
                 speed=1)
    lane(root, [{**one, "group": "sequence"}, {**two, "group": "sequence"}], name="broll")
    layers(root, pictures, monkeypatch, titles=[])
    out, _ = render(root)
    for frame, colour in ((0, (255, 0, 0)), (8, (0, 255, 0)), (29, (0, 255, 0)),
                          (30, (255, 0, 0)), (59, (255, 0, 0)), (60, (0, 0, 255))):
        assert_colour(synth.picture_pixels(out, frame, box=(500, 700, 40, 40)), colour)
    # Exercise a decoded stream one frame shorter than its validated metadata duration.
    graph = stage._Graph()
    item = resolve_timeline(root).elements[1]
    short = synth.picture_source(tmp_path / "short.mov", "color=c=red:s=320x180:r=30", 29)
    item = replace(item, start_frame=0, end_frame=30,
                   payload={"media": {"in_s": 0, "speed": 1, "fit": "cover"}})
    branch = stage._insert(graph, stage._Source(short, ffmpeg.probe(short), None), item)
    padded = tmp_path / "padded.mkv"
    ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", *graph.inputs, "-filter_complex",
                ";".join(graph.filters), "-map", f"[{branch}]", "-an", "-c:v", "ffv1",
                str(padded)], timeout_s=60)
    assert ffmpeg.probe(padded).video_frames == 30
    assert len(set(synth.picture_hashes(padded))) == 1


def test_graphics_order_then_captions(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = project(tmp_path / "project", pictures)
    synth.picture_source(root / "behind.mov", "color=c=red:s=200x100:r=30,format=rgba",
                         90, alpha=True)
    synth.picture_source(root / "front.mov", "color=c=blue:s=100x80:r=30,format=rgba",
                         90, alpha=True)
    behind = {"id": "agent-behind", "path": "behind.mov", "x": 80, "y": 100,
              "width": 200, "height": 100, "start_frame": 0, "frames": 90,
              "layer": "behind", "kind": "graphic"}
    title = {**behind, "id": "agent-title", "path": "title.mov", "kind": "title",
             "frames": 15}
    front = {**behind, "id": "agent-front", "path": "front.mov", "layer": "front",
             "x": 130, "width": 100, "height": 80}
    layers(root, pictures, monkeypatch, titles=[front, title, behind])
    monkeypatch.setattr(stage, "captions", lambda _: {"layer": {
        "path": "captions.mov", "x": 180, "y": 140, "width": 100, "height": 80,
        "start_frame": 0, "frames": 90,
    }})
    out, _ = render(root)
    assert_colour(synth.picture_pixels(out, 0, box=(100, 120, 20, 20)), (255, 127, 0))
    assert_colour(synth.picture_pixels(out, 0, box=(150, 110, 20, 20)), (0, 0, 255))
    assert_colour(synth.picture_pixels(out, 0, box=(200, 160, 20, 20)), (0, 255, 0))
    assert_colour(synth.picture_pixels(out, 15, box=(100, 120, 20, 20)), (255, 0, 0))


@pytest.mark.parametrize("transfer", ["arib-std-b67", "smpte2084"])
def test_hdr_off_or_unavailable_refused_before_layer_render(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch, transfer: str,
) -> None:
    source = synth.picture_source(tmp_path / "hdr.mov", "testsrc2=s=160x90:r=30", 90,
                                  transfer=transfer)
    root = project(tmp_path / "project", pictures)
    item = insert(root, source, 0, 30, hdr="off")
    lane(root, [item], name="broll")
    with pytest.raises(TimelineRefusal, match="HDR picture requires"):
        render(root)
    lane(root, [{**item, "media": {**item["media"], "hdr": "auto"}}], name="broll")
    monkeypatch.setattr(hdr.sys, "platform", "linux")
    monkeypatch.setattr(ffmpeg, "has_filters", lambda names: dict.fromkeys(names, False))
    with pytest.raises(hdr.HdrUnsupportedError):
        render(root)
    assert not (root / "work").exists()


@pytest.mark.helpers
@pytest.mark.parametrize("transfer,kind", [("arib-std-b67", "hlg"), ("smpte2084", "pq")])
def test_hdr_insert_shared_conversion_and_receipt(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
    transfer: str, kind: str,
) -> None:
    if not hdr.videotoolbox_available():
        pytest.skip("Shared hardware HDR conversion is unavailable")
    source = synth.picture_source(tmp_path / "hdr.mov", "testsrc2=s=160x90:r=30", 90,
                                  transfer=transfer)
    root = project(tmp_path / "project", pictures)
    lane(root, [insert(root, source, 0, 30, hdr="auto")], name="broll")
    layers(root, pictures, monkeypatch, titles=[])
    out, result = render(root)
    assert result.conversions[0]["kind"] == kind
    assert result.conversions[0]["method"] == "videotoolbox"
    assert result.conversions[0]["output_colour"] == "bt709"
    assert ffmpeg.probe(out).color_transfer == "bt709"


@pytest.mark.parametrize("pixel_format,download", [
    ("yuv420p", "nv12"), ("yuvj420p", "nv12"), ("nv12", "nv12"),
    ("yuv420p10le", "p010le"), ("yuv420p10be", "p010le"),
    ("p010le", "p010le"), ("p010be", "p010le"),
    ("yuv422p", "nv16"), ("yuvj422p", "nv16"), ("nv16", "nv16"),
    ("yuv422p10le", "p210le"), ("yuv422p10be", "p210le"),
    ("p210le", "p210le"), ("p210be", "p210le"),
    ("nv20le", "p210le"), ("nv20be", "p210le"),
    ("yuv444p", "nv24"), ("yuvj444p", "nv24"), ("nv24", "nv24"),
    ("yuv444p10le", "p410le"), ("yuv444p10be", "p410le"),
    ("p410le", "p410le"), ("p410be", "p410le"),
])
def test_hdr_download_format_matches_chroma_and_depth(
    pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
    pixel_format: str, download: str,
) -> None:
    """Select each hardware surface format without starting a hardware decoder."""
    info = replace(ffmpeg.probe(pictures["insert"], count_frames=False),
                   color_transfer="arib-std-b67", pix_fmt=pixel_format)
    monkeypatch.setattr(hdr.sys, "platform", "darwin")
    monkeypatch.setattr(ffmpeg, "has_filters", lambda names: dict.fromkeys(names, True))
    selected = hdr.conversion(info)
    assert selected is not None
    assert selected.method == "videotoolbox"
    assert selected.filter_prefix.endswith(f"hwdownload,format={download},format=yuv420p")


@pytest.mark.parametrize("pixel_format", [None, "", "unknown", "rgb24", "gray",
                                        "yuv420p9le", "yuv422p12le", "yuv444p16le"])
def test_hdr_unknown_or_unsupported_download_format_refused(
    pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch, pixel_format: str | None,
) -> None:
    """Refuse unsupported layouts instead of attempting an incompatible hardware download."""
    info = replace(ffmpeg.probe(pictures["insert"], count_frames=False),
                   color_transfer="smpte2084", pix_fmt=pixel_format)
    monkeypatch.setattr(hdr.sys, "platform", "darwin")

    def unexpected(_: list[str]) -> dict[str, bool]:
        pytest.fail("Unsupported pixel formats must be refused before probing hardware filters")

    monkeypatch.setattr(ffmpeg, "has_filters", unexpected)
    with pytest.raises(hdr.HdrUnsupportedError) as caught:
        hdr.conversion(info)
    assert caught.value.code == "hdr_unsupported"
    assert "download format" in str(caught.value)
    assert "yuv420p10le" in caught.value.suggested_action
    assert "BT.709" in caught.value.suggested_action


@pytest.mark.parametrize("field,value", [
    ("video_frames", 5), ("width", 100), ("height", 100), ("fps", 24),
    ("pix_fmt", "yuv444p"), ("color_primaries", "bt2020"),
    ("color_transfer", "smpte2084"), ("color_space", None), ("has_audio", True),
])
def test_delivery_assertions_refuse_bad_stream(
    pictures: dict[str, Path], field: str, value: Any,
) -> None:
    info = ffmpeg.probe(pictures["cut"])
    good = replace(info, video_frames=93, video_codec="h264", pix_fmt="yuv420p")
    with pytest.raises(stage.PictureError, match="failed verification"):
        stage._assert_picture(replace(good, **{field: value}), 93)


@pytest.mark.render
@pytest.mark.parametrize("move", ["punch", "push", "keys"])
def test_native_keyframed_zoom_marker(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch, move: str,
) -> None:
    source = synth.picture_source(
        tmp_path / "marker.mov", "color=c=black:s=1440x2560:r=30,"
        "drawbox=x=620:y=1160:w=200:h=240:c=red:t=fill", 120)
    cut = synth.picture_source(
        tmp_path / "marker-cut.mov", "color=c=black:s=1080x1920:r=30,"
        "drawbox=x=465:y=870:w=150:h=180:c=red:t=fill", 90)
    root = project(tmp_path / "project", {**pictures, "take": source, "cut": cut}, hold_frames=0)
    item = {"id": "agent-camera", "kind": "zoom", "start": {"time": 0},
            "end": {"time": 2}, "move": move, "scale": 1.3, "center": [.5, .5]}
    if move == "keys":
        item.pop("scale")
        item["keys"] = [{"at": {"time": 0}, "scale": 1, "center": [.4, .5]},
                        {"at": {"time": 1}, "scale": 1.3, "center": [.6, .5],
                         "ease": "linear"}]
    lane(root, [item])
    layers(root, pictures, monkeypatch, titles=[])
    out, _ = render(root)

    def marker(frame):
        pixels = synth.picture_pixels(out, frame, box=(300, 700, 480, 520))
        ys, xs = np.where((pixels[:, :, 0] > 128) & (pixels[:, :, 1] < 64)
                          & (pixels[:, :, 2] < 64))
        return xs.max() - xs.min() + 1, ys.max() - ys.min() + 1, xs.mean(), ys.mean()

    base = marker(0)
    end_sample = 59 if move == "push" else 55
    samples = [(30, 1.3 if move != "push" else math.sqrt(1.3)),
               (end_sample, 1.3 if move != "push" else 1.3 ** (end_sample / 60)), (60, 1)]
    if move == "punch":
        samples += [(frame, 1.3 ** (1 - (1 - frame / 2)**3)) for frame in range(1, 3)]
        samples += [(frame, 1.3 * (1 / 1.3) ** (1 - (1 - (frame - 56) / 4)**3))
                    for frame in range(56, 60)]
    for frame, scale in samples:
        actual = marker(frame)
        assert actual[:2] == pytest.approx([value * scale for value in base[:2]], abs=2)
        expected_x = base[2] - (1080 * .1 * (scale - 1) if move == "keys" else 0)
        assert actual[2:] == pytest.approx((expected_x, base[3]), abs=1)


@pytest.mark.render
def test_opening_push_scale_title_and_repeatable_picture(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.zoom_curve import ZoomCurve

    source = synth.picture_source(
        tmp_path / "opening-markers.mov", "color=c=black:s=1440x2560:r=30,"
        "drawbox=x=400:y=1264:w=32:h=32:c=red:t=fill,"
        "drawbox=x=1000:y=1264:w=32:h=32:c=blue:t=fill", 120)
    cut = synth.picture_source(
        tmp_path / "opening-cut.mov", "color=c=black:s=1080x1920:r=30,"
        "drawbox=x=300:y=948:w=24:h=24:c=red:t=fill,"
        "drawbox=x=750:y=948:w=24:h=24:c=blue:t=fill", 90)
    root = project(tmp_path / "project", {**pictures, "take": source, "cut": cut},
                   hold_frames=0, word_rows=[("First", .1, .4), ("line.", .5, 1.1),
                                             ("Next", 1.1, 2.5)],
                   spans=[{"source": "t1", "start": "0", "end": "1.1"},
                          {"source": "t1", "start": "1.1", "end": "3"}])
    add_timeline(root, write_document(root / "opening.json", {
        "schema": "yap-studio/timeline@1",
        "elements": [{"id": "agent-opening", "kind": "zoom", "lane": "agent",
                      "start": {"word": "t1:0"},
                      "move": "opening", "center": [.5, .5]}],
    }), lane="agent").require_valid()
    layers(root, pictures, monkeypatch)
    out, _ = render(root)
    second = root / "repeat/picture.mp4"
    stage.render_picture(root, second, work_dir=root / "repeat")
    assert out.read_bytes() == second.read_bytes()
    item = resolve_timeline(root).elements[0]
    assert (item.start_frame, item.end_frame) == (0, 33)
    curve = ZoomCurve.from_payload(item.start_frame, item.end_frame, item.payload)
    raw = synth._run(["ffmpeg", "-v", "error", "-i", str(out), "-an", "-vf",
                      "crop=1080:120:0:900,format=rgb24", "-f", "rawvideo", "-"])
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 120, 1080, 3)
    spacings = []
    for pixels in frames:
        red = (pixels[:, :, 0] > 128) & (pixels[:, :, 1] < 64) & (pixels[:, :, 2] < 64)
        blue = (pixels[:, :, 2] > 128) & (pixels[:, :, 0] < 64) & (pixels[:, :, 1] < 64)
        assert red.any() and blue.any()
        spacings.append(np.where(blue)[1].mean() - np.where(red)[1].mean())
    measured = [spacing / spacings[0] for spacing in spacings]
    for frame, expected in enumerate(curve.scales(0, len(frames))):
        assert abs(math.log(measured[frame] / expected)) < .01, (frame, measured[frame], expected)
    assert measured[31] < measured[32] < 1.3
    assert measured[33] == pytest.approx(1, abs=.001)
    raw = synth._run(["ffmpeg", "-v", "error", "-i", str(out), "-an", "-vf",
                      "crop=200:100:80:100,format=rgb24", "-f", "rawvideo", "-"])
    titles = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 100, 200, 3)
    for frame in range(1, 15):
        np.testing.assert_array_equal(titles[frame], titles[0])


@pytest.mark.render
@pytest.mark.parametrize("move,scale,take_size", [
    ("punch", 1.5, (1440, 2560)), ("punch", 2, (1440, 2560)),
    ("push", 2, (1440, 2560)), ("punch", 2, (1080, 1920)),
])
def test_native_zoom_scale_every_frame(
    tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
    move: str, scale: float, take_size: tuple[int, int],
) -> None:
    from yap_studio.zoom_curve import ZoomCurve

    factor = take_size[0] / 1080
    source = synth.picture_source(
        tmp_path / "markers.mov", f"color=c=black:s={take_size[0]}x{take_size[1]}:r=30,"
        f"drawbox=x={round(300 * factor)}:y={round(948 * factor)}:"
        f"w={round(24 * factor)}:h={round(24 * factor)}:c=red:t=fill,"
        f"drawbox=x={round(750 * factor)}:y={round(948 * factor)}:"
        f"w={round(24 * factor)}:h={round(24 * factor)}:c=blue:t=fill", 120)
    cut = synth.picture_source(
        tmp_path / "markers-cut.mov", "color=c=black:s=1080x1920:r=30,"
        "drawbox=x=300:y=948:w=24:h=24:c=red:t=fill,"
        "drawbox=x=750:y=948:w=24:h=24:c=blue:t=fill", 90)
    root = project(tmp_path / "project", {**pictures, "take": source, "cut": cut}, hold_frames=0)
    lane(root, [{"id": "agent-camera", "kind": "zoom", "start": {"time": .2},
                 "end": {"time": 2}, "move": move, "scale": scale, "center": [.5, .5]}])
    layers(root, pictures, monkeypatch, titles=[])
    out, result = render(root)
    fallback = ("agent-camera: take t1 is not larger than the canvas in the "
                "cropped axis; rendering zoom from the cut.")
    assert (fallback in result.warnings) == (take_size == (1080, 1920))
    item = resolve_timeline(root).elements[0]
    curve = ZoomCurve.from_payload(item.start_frame, item.end_frame, item.payload)
    raw = synth._run(["ffmpeg", "-v", "error", "-i", str(out), "-an", "-vf",
                      "crop=1080:120:0:900,format=rgb24", "-f", "rawvideo", "-"])
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 120, 1080, 3)
    spacings = []
    for pixels in frames:
        red = (pixels[:, :, 0] > 128) & (pixels[:, :, 1] < 64) & (pixels[:, :, 2] < 64)
        blue = (pixels[:, :, 2] > 128) & (pixels[:, :, 0] < 64) & (pixels[:, :, 1] < 64)
        assert red.any() and blue.any()
        spacings.append(np.where(blue)[1].mean() - np.where(red)[1].mean())
    measured = [spacing / spacings[0] for spacing in spacings]
    for frame, actual in enumerate(measured):
        expected = curve.at(frame)[0] if curve.start_frame <= frame < curve.end_frame else 1
        assert abs(math.log(actual / expected)) < .01, (frame, actual, expected)
    sample, expected = (7, scale ** .875) if move == "punch" else (33, math.sqrt(scale))
    assert abs(math.log(measured[sample] / expected)) < .01
    from yap_studio.zoom_curve import declared_moves, scale_motion

    assert scale_motion(measured, 0, declared_moves(curve), tolerance=.01) == ()


def test_empty_smooth_layers_is_byte_identical(tmp_path, pictures, monkeypatch) -> None:
    root = project(tmp_path / "project", pictures, hold_frames=0)
    layers(root, pictures, monkeypatch)
    first, _ = render(root)
    second = root / "second/picture.mp4"
    stage.render_picture(root, second, work_dir=root / "second", smooth_layers=[])
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.render
def test_step_in_native_cut_is_instant_and_deterministic(tmp_path, pictures, monkeypatch):
    from types import SimpleNamespace

    from test_step_in import pieces

    from yap_studio.faces.model import FaceBox
    from yap_studio.stages import timeline as timeline_stage

    source = synth.picture_source(
        tmp_path / "step-marker.mov", "color=c=black:s=1440x2560:r=30,"
        "drawbox=x=620:y=1160:w=200:h=240:c=red:t=fill", 120)
    cut = synth.picture_source(
        tmp_path / "step-cut.mov", "color=c=black:s=1080x1920:r=30,"
        "drawbox=x=465:y=870:w=150:h=180:c=red:t=fill", 90)
    root = project(tmp_path / "project", {**pictures, "take": source, "cut": cut}, hold_frames=0,
                   cut_pieces=pieces((30, 60)))
    style_file = root / "picture-style.json"
    style = json.loads(style_file.read_bytes())
    style["moves"]["jump_cuts"] = "step_in"
    write_json(style_file, style)
    measured = SimpleNamespace(frames=tuple(
        (FaceBox(Box(400, 400, 200, 250), 1, "detected"),) for _ in range(90)))
    monkeypatch.setattr(timeline_stage, "_zoom_faces", lambda context, *, measure: measured)
    layers(root, pictures, monkeypatch, titles=[])
    out, _ = render(root)
    second = root / "second/picture.mp4"
    stage.render_picture(root, second, work_dir=second.parent)
    assert out.read_bytes() == second.read_bytes()
    raw = synth._run(["ffmpeg", "-v", "error", "-i", str(out), "-an", "-vf",
                      "crop=700:600:200:600,format=rgb24", "-f", "rawvideo", "-"])
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 600, 700, 3)
    widths = []
    for pixels in frames:
        _, xs = np.where((pixels[:, :, 0] > 128) & (pixels[:, :, 1] < 64)
                         & (pixels[:, :, 2] < 64))
        widths.append(xs.max() - xs.min() + 1)
    for frame, width in enumerate(widths):
        assert width / widths[0] == pytest.approx(1 if frame < 30 else 1.15, abs=.02)


def test_person_restoration_skips_smooth_join_overlap(tmp_path, pictures, monkeypatch) -> None:
    plate = synth.picture_source(tmp_path / "person.mov", "color=blue:s=1080x1920:r=30,"
                                 "drawbox=x=80:y=100:w=100:h=100:c=red:t=fill", 90)
    root = project(tmp_path / "project", {**pictures, "cut": plate}, hold_frames=0)
    card = synth.picture_source(tmp_path / "opaque.mov",
                                "color=yellow:s=200x100:r=30,format=rgba", 30, alpha=True)
    layers(root, {**pictures, "title": card}, monkeypatch, titles=[{
        "id": "agent-card", "path": "title.mov", "x": 80, "y": 100,
        "width": 200, "height": 100, "start_frame": 6, "frames": 30,
        "layer": "behind", "kind": "graphic",
    }])
    monkeypatch.setattr(matte_cache, "make_matte", lambda cut, folder, *, joins=(): synthetic_matte(
        cut, folder, box=(80, 100, 100, 100), active_windows=((6, 36),),
    ))
    movie = synth.picture_source(root / "smooth.mp4", "color=lime:s=1080x1920:r=30", 6)
    behind = ResolvedElement("agent-card", "graphic", "agent", 6, 36, "behind", {})
    timeline = ResolvedTimeline((behind,), (), (), 90, 0)
    out = root / "work/picture.mp4"
    result = stage.render_picture(root, out, work_dir=root / "work", timeline=timeline,
                                  smooth_layers=[{"id": "smooth-0", "path": str(movie),
                                                  "start_frame": 15, "frames": 6,
                                                  "join_frame": 18}])
    assert_colour(synth.picture_pixels(out, 14, box=(90, 110, 20, 20)), (255, 0, 0))
    assert_colour(synth.picture_pixels(out, 18, box=(90, 110, 20, 20)), (255, 255, 0))
    assert_colour(synth.picture_pixels(out, 21, box=(90, 110, 20, 20)), (255, 0, 0))
    assert result.warnings.count("Person restoration for behind graphics is skipped during "
                                 "smooth join windows.") == 1
