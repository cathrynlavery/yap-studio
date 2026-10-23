"""Opt-in translucent chapters, busy picture contrast and deterministic frames."""

import json
import subprocess
import threading
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from test_cards_render import _check_reference
from test_chapters import chapter_style
from test_sync import _plan

from yap_studio import ffmpeg, paths, video
from yap_studio.checks import sync
from yap_studio.checks.caption_contrast import word_contrast
from yap_studio.edl.model import PictureRegion
from yap_studio.gallery.server import GalleryServer
from yap_studio.gallery.still import sample_still
from yap_studio.render import hyperframes
from yap_studio.stages.assemble_picture import _Graph
from yap_studio.stages.captions import (
    LayerPlan,
    chapter_layer_plan,
    chapter_text_box,
    prepare_layer_plan,
    render_layer_plan,
)


def _pixels(path: Path, frames: list[int]) -> np.ndarray:
    selection = "+".join(f"eq(n,{frame})" for frame in frames)
    raw = ffmpeg.run([
        "ffmpeg", "-nostdin", "-i", str(path), "-vf", f"select='{selection}'",
        "-fps_mode", "passthrough", "-frames:v", str(len(frames)),
        "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1",
    ], timeout_s=120, binary=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(len(frames), 1920, 1080, 4)


def _chapter_boxes(plan: LayerPlan, work: Path) -> dict[str, list[float]]:
    composition = prepare_layer_plan(plan, work)
    install = hyperframes.ensure_installed()
    result = subprocess.run([
        "node", str(Path(__file__).parent / "fixtures/chapter_probe.mjs"),
        str(install.prefix / "node_modules"), str(install.browser_path),
        str(composition / "index.html"),
    ], input=json.dumps(plan.variables), text=True, capture_output=True, timeout=60, check=True)
    return json.loads(result.stdout)


@pytest.mark.render
@pytest.mark.parametrize("font", ["Inter", "Gelasio"])
@pytest.mark.parametrize("title", ["Small steps", "How tokens are counted",
                                 "A longer chapter title with several clear words"])
def test_chapter_text_bounds_contain_painted_glyphs(tmp_path, font, title):
    plan = chapter_layer_plan(title, 1, chapter_style(hook={"font": font}))
    bounds = chapter_text_box(plan)
    assert 0 < bounds.y < bounds.y1 < 1920
    with paths.operation():
        rendered = render_layer_plan(plan, tmp_path / "layer", tmp_path / "chapter.mov")
        for frame in _pixels(rendered.path, [5, 9, 12, 17, 24, 40]):
            # The translucent wash is below 71% alpha. Opaque pixels are text or rule.
            rows, columns = np.nonzero(frame[..., 3] >= 250)
            assert rows.size
            assert bounds.x <= columns.min() and bounds.y <= rows.min()
            assert bounds.x1 >= columns.max() + 1 and bounds.y1 >= rows.max() + 1


@pytest.mark.render
@pytest.mark.parametrize("caption", ["#FFFFFF", "#111111"])
def test_chapter_translucency_fade_contrast_and_sync(tmp_path: Path, caption: str) -> None:
    plan = chapter_layer_plan("How tokens are counted", 2,
                              chapter_style(captions={"text_color": caption}),
                              start_frame=2520, frames=48)
    with paths.operation():
        rendered = render_layer_plan(plan, tmp_path / "layer", tmp_path / "chapter.mov")
        boxes = _chapter_boxes(plan, tmp_path / "probe")
        alpha_raw = ffmpeg.run([
            "ffmpeg", "-nostdin", "-i", str(rendered.path), "-vf", "alphaextract",
            "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1",
        ], timeout_s=120, binary=True).stdout
        alpha = np.frombuffer(alpha_raw, dtype=np.uint8).reshape(plan.frames, 1920, 1080)
        # The wash and every glyph share the layer fade. Entrances can delay glyphs.
        clock = np.arange(plan.frames) / 30
        fade = np.minimum(np.minimum(clock / .15, (1.6 - clock) / .2), 1)
        for y, x in ((1880, 40), (40, 1040)):
            np.testing.assert_allclose(alpha[:, y, x] / 255, .45 * fade, atol=.012)
            assert np.all(alpha[:, y, x] < 255)
        assert np.all(alpha[0] == 0)
        assert np.all(np.diff(alpha[:6, 1880, 40].astype(int)) >= 0)
        assert np.all(np.diff(alpha[42:, 1880, 40].astype(int)) <= 0)
        assert np.all(alpha.max(axis=(1, 2)) / 255 <= fade + .012)
        assert np.all(alpha[1:].max(axis=(1, 2)) > 0)
        np.testing.assert_allclose(alpha[42:].max(axis=(1, 2)) / 255, fade[42:], atol=.012)
        samples = [17, 24]  # First frame after every entrance completes, then midpoint.
        rgba = _pixels(rendered.path, samples)
        for frame in rgba:
            for box in boxes.values():
                x, y, w, h = map(round, box)
                assert np.count_nonzero(frame[y:y + h, x:x + w, 3] == 255) > 100
            # No hard band edges and no opaque surface behind the text.
            column = frame[:, 40, 3].astype(int)
            assert column.max() / 255 == pytest.approx(.70, abs=.015)
            assert np.max(np.abs(np.diff(column))) <= 2
        if caption == "#FFFFFF":
            neutral = np.full_like(rgba[1], 255)
            weight = rgba[1, ..., 3:4] / 255
            neutral[..., :3] = np.rint(rgba[1, ..., :3] * weight + 128 * (1 - weight))
            _check_reference("chapter", neutral)
            repeat = render_layer_plan(plan, tmp_path / "repeat", tmp_path / "repeat.mov")
            hashes = [ffmpeg.run([
                "ffmpeg", "-nostdin", "-i", str(path), "-map", "0:v:0", "-an",
                "-f", "framemd5", "pipe:1",
            ], timeout_s=120).stdout for path in (rendered.path, repeat.path)]
            assert hashes[0] == hashes[1]
        wash = np.array([int(plan.variables["background_color"][i:i + 2], 16)
                         for i in (1, 3, 5)])
        composites = []
        for name, pattern in (
            ("moving", "testsrc2=s=1080x1920:r=30:d=1.6"),
            ("checker", "nullsrc=s=1080x1920:r=30:d=1.6,"
             "geq=lum='if(mod(floor(X/32)+floor(Y/32),2),235,16)':cb=128:cr=128"),
        ):
            base = tmp_path / f"{name}.mkv"
            ffmpeg.run([
                "ffmpeg", "-nostdin", "-f", "lavfi", "-i", pattern,
                "-frames:v", "48", "-c:v", "ffv1", str(base),
            ], timeout_s=120)
            graph = _Graph()
            output = graph.overlay(graph.filter("0:v", "format=rgb24"), "1:v", 0, 48)
            final = tmp_path / f"{name}-composite.mkv"
            ffmpeg.run([
                "ffmpeg", "-nostdin", "-i", str(base), "-i", str(rendered.path),
                "-filter_complex", ";".join(graph.filters), "-map", f"[{output}]",
                "-frames:v", "48", "-c:v", "ffv1", str(final),
            ], timeout_s=120)
            picture, ground = _pixels(final, samples), _pixels(base, samples)
            composites.append(picture)
            for index, frame in enumerate(picture):
                for box in boxes.values():
                    ratio = word_contrast(frame[..., :3], rgba[index], tuple(box))
                    assert ratio is not None and ratio >= 3, (name, index, box, ratio)
                for y, x in ((1880, 40), (40, 1040)):
                    expected = .45 * wash + .55 * ground[index, y, x, :3]
                    np.testing.assert_allclose(frame[y, x, :3], expected, atol=4)
            if name == "moving":
                # Sync normalises brightness and contrast. Mask only the text and
                # varying band; the uniform wash still carries the source timing.
                source = video.grey_frames(base)
                composited = video.grey_frames(final)
                valid = ~video.exclusion_mask(source.shape[1:], [(0, 240, 1080, 1320)],
                                              1080, 1920)
                mapping = video.estimate_mapping(source[20:25], composited[20:25],
                                                  render_valid=valid)
                assert mapping.kind == "full" and mapping.score > .95
                landing, distance, second = video.locate_sequence(
                    source[20:29], composited, expect=20, search=8, valid=valid)
                assert landing == 20 and distance < .8 * second, (landing, distance, second)
                (row,) = sync.measure_segments(
                    final, _plan(base, [(0, 1.6)]), {"main": base},
                    regions=[PictureRegion(Fraction(0), Fraction(8, 5), 0, 240, 1080, 1320)],
                )
                assert row.measured and row.masked_region
                assert row.probe_frame == 24 and row.video_vs_plan_frames == 0, row
        for y, x in ((1880, 40), (40, 1040)):
            difference = np.abs(composites[0][:, y, x, :3].astype(int)
                                - composites[1][:, y, x, :3].astype(int))
            assert difference.max() > 40


@pytest.mark.render
def test_chapter_title_contrasts_over_dark_and_light_footage(tmp_path: Path) -> None:
    plan = chapter_layer_plan("How tokens are counted", 2,
                              chapter_style(captions={"text_color": "#111111"},
                                            graphics={"paper_color": "#777777",
                                                      "ink_color": "#777777"}), frames=30)
    with paths.operation():
        rendered = render_layer_plan(plan, tmp_path / "layer", tmp_path / "chapter.mov")
        boxes = _chapter_boxes(plan, tmp_path / "probe")
        rgba = _pixels(rendered.path, [24])[0]
    weight = rgba[..., 3:4] / 255
    for ground in (8, 247):
        picture = np.rint(rgba[..., :3] * weight + ground * (1 - weight)).astype(np.uint8)
        ratio = word_contrast(picture, rgba, tuple(boxes["title"]))
        assert ratio is not None and ratio >= 3, (ground, ratio)


@pytest.mark.render
def test_chapter_gallery_desktop_and_phone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    install = hyperframes.ensure_installed()
    style = chapter_style().data
    style["chapters"]["style"] = "none"
    server = GalleryServer(sample_still(), style, tmp_path / "gallery",
                           install=install, gsap=hyperframes.gsap_file(install))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        script = Path(__file__).parent / "fixtures/chapter_viewport.mjs"
        result = subprocess.run([
            "node", str(script), str(install.prefix / "node_modules"),
            str(install.browser_path), server.url, str(tmp_path),
        ], capture_output=True, text=True, timeout=120, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
