"""Fast source and transcript checks for synthetic finished-video fixtures."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
import taste_finals

from yap_studio import ffmpeg
from yap_studio.captions import CaptionWord, build_phrases
from yap_studio.checks.structure import _loudness
from yap_studio.errors import YapError
from yap_studio.style import SHIPPED_PRESETS, resolve
from yap_studio.style.loader import resolve_style_data
from yap_studio.taste import add_proposals, learn, load_taste
from yap_studio.taste.frames import delta_e, hex_color, interior_components, iter_rgb, lab, rgb
from yap_studio.taste.learn import (
    CAPTION_SIZE_TOLERANCE,
    CAPTION_TOP_TOLERANCE,
    COLOR_HIGHLIGHT_TOLERANCE,
    COLOR_TEXT_TOLERANCE,
    COLOR_TITLE_TOLERANCE,
    CORNER_RADIUS_TOLERANCE,
    DURATION_TOLERANCE,
    LUFS_TOLERANCE,
    OCR_FIELDS,
    PANEL_WIDTH_TOLERANCE,
    PAUSE_TOLERANCE,
    TITLE_DURATION_TOLERANCE,
    TITLE_SIZE_TOLERANCE,
    ZOOM_SCALE_TOLERANCE,
    _highlight_samples,
    aggregate,
    apply_dependencies,
    tolerance,
)
from yap_studio.taste.measure_pacing import (
    audio_envelope,
    loudness,
    pause_envelope,
    punch_in,
    scaled_picture,
    transcript_pacing,
)
from yap_studio.taste.measure_panels import find_panels, panel_present
from yap_studio.taste.measure_shapes import corner_radius, font_class
from yap_studio.taste.measure_text import (
    BLOCK_FRINGE_PX,
    _edge_fill,
    _fringe_allowance,
    caption_band,
    classify_highlight,
    line_pitch,
    panel_text,
)
from yap_studio.taste.measure_words import (
    LIMIT_FIELDS,
    NEEDS_WORDS,
    assign_words,
    caption_windows,
    consistent_limits,
    infer_limits,
)
from yap_studio.transcript.model import Transcript, Word

_SIZE = (180, 320)


def _rounded_card_picture(radii: tuple[int, int, int, int]) -> tuple[np.ndarray, dict]:
    """Draw independently antialiased corners over a gradient with a soft offset shadow."""
    width, height = 1080, 480
    x, y, w, h = 160, 110, 760, 230
    yy, xx = np.mgrid[:height, :width]
    gradient = 65 + 0.03 * xx + 0.04 * yy
    # Smooth shadow extends below the lower edge and around both sides.
    outside_x = np.maximum(np.maximum(x - xx, xx - (x + w)), 0)
    outside_y = np.maximum(np.maximum(y + 10 - yy, yy - (y + h + 10)), 0)
    shadow = 0.18 * np.exp(-(outside_x**2 + outside_y**2) / (2 * 18**2))
    background = np.stack((gradient, gradient + 12, gradient + 6), axis=-1)
    background *= 1 - shadow[:, :, None]
    coverage = np.ones((h, w))
    for radius, dx, dy in zip(radii, (1, -1, 1, -1), (1, 1, -1, -1), strict=True):
        if not radius:
            continue
        offsets = (np.arange(8) + 0.5) / 8
        u = np.arange(radius)[:, None] + offsets
        # Supersampling integrates each quarter disk at the pixel boundary.
        disk = ((u[:, None, :, None] - radius)**2 +
                (u[None, :, None, :] - radius)**2 <= radius**2).mean(axis=(2, 3))
        columns = np.arange(radius) if dx > 0 else w - 1 - np.arange(radius)
        rows = np.arange(radius) if dy > 0 else h - 1 - np.arange(radius)
        coverage[np.ix_(rows, columns)] = disk
    paper = rgb("#F8F4EC")
    region = background[y:y + h, x:x + w]
    region[:] = region * (1 - coverage[:, :, None]) + paper * coverage[:, :, None]
    image = np.clip(np.rint(background), 0, 255).astype(np.uint8)
    card = {"box": {"x": x, "y": y, "width": w, "height": h},
            "start_frame": 10, "end_frame": 40}
    return image, card


@pytest.mark.parametrize("radius", [0, 16, 24, 40])
def test_corner_radius_integrates_antialiasing_gradient_and_shadow(radius: int) -> None:
    image, card = _rounded_card_picture((radius,) * 4)
    value, frames, detail = corner_radius([(20, image), (30, image)], [card], "#F8F4EC")
    assert value == pytest.approx(radius, abs=CORNER_RADIUS_TOLERANCE), detail
    assert frames == detail["frames"] == [20, 30]
    assert len(detail["corners"]) == 8
    assert {row["corner"] for row in detail["corners"]} == {
        "top_left", "top_right", "bottom_left", "bottom_right"
    }
    assert all(row["radius_px"] == pytest.approx(radius, abs=CORNER_RADIUS_TOLERANCE)
               for row in detail["corners"])
    if radius == 0:
        assert value == 0


def test_corner_radius_refuses_disagreeing_corners() -> None:
    image, card = _rounded_card_picture((16, 16, 40, 16))
    value, frames, detail = corner_radius([(20, image)], [card], "#F8F4EC")
    assert value is None and frames == [20]
    assert "corners disagree" in detail["reason"]
    assert detail["spread_margin_px"] < 0


def test_corner_radius_excludes_fades_and_refuses_missing_evidence() -> None:
    image, card = _rounded_card_picture((24,) * 4)
    value, frames, detail = corner_radius(
        [(10, np.zeros_like(image)), (20, image), (40, np.zeros_like(image))],
        [card], "#F8F4EC",
    )
    assert value == pytest.approx(24, abs=CORNER_RADIUS_TOLERANCE)
    assert frames == [20]
    value, frames, detail = corner_radius([(10, image)], [card], "#F8F4EC")
    assert value is None and not frames and "settled interior frames" in detail["reason"]
    value, frames, detail = corner_radius([], [], "#F8F4EC")
    assert value is None and not frames and detail["reason"] == "No card was found."


def test_corner_radius_on_full_size_synthetic_source() -> None:
    style = resolve("preset:clean_minimal")
    raw = taste_finals.source_frame(225, style=style)
    decoded = _ffmpeg([
        "-f", "rawvideo", "-pixel_format", "yuv420p", "-video_size", "1080x1920",
        "-framerate", "30", "-i", "pipe:0", "-frames:v", "1",
        "-vf", "scale=in_color_matrix=bt709:in_range=tv,format=rgb24",
        "-f", "rawvideo", "pipe:1",
    ], data=raw)
    image = np.frombuffer(decoded, dtype=np.uint8).reshape(1920, 1080, 3)
    card = {"box": {"x": 60, "y": 280, "width": 960, "height": 360},
            "start_frame": 195, "end_frame": 269}
    value, frames, detail = corner_radius([(225, image)], [card], style.data["graphics"][
        "paper_color"
    ])
    assert value == pytest.approx(taste_finals.CARD_RADIUS, abs=CORNER_RADIUS_TOLERANCE), detail
    assert frames == [225]


def _stroke_glyphs(stem: int, hairline: int | None = None) -> np.ndarray:
    """Draw six H-shaped masks with optional thin crossbars and small serifs."""
    mask = np.zeros((100, 1080), dtype=bool)
    bar = stem if hairline is None else hairline
    for left in range(100, 700, 100):
        mask[15:85, left:left + stem] = True
        mask[15:85, left + 50:left + 50 + stem] = True
        mask[48:48 + bar, left:left + 50 + stem] = True
        if hairline is not None:
            for edge in (left, left + 50):
                for top in (15, 85 - hairline):
                    mask[top:top + hairline, edge - 8:edge + stem + 8] = True
    return mask


@pytest.mark.parametrize("stem,hairline,expected", [
    (8, None, {"family": "sans", "weight": "regular"}),
    (8, 2, {"family": "serif", "weight": "regular"}),
    (20, None, {"family": "sans", "weight": "heavy"}),
])
def test_font_class_measures_monoline_contrast_and_weight(
    stem: int, hairline: int | None, expected: dict,
) -> None:
    mask = _stroke_glyphs(stem, hairline)
    value, frames, detail = font_class([(7, mask), (29, mask)], size_px=100)
    assert value == expected, detail
    assert frames == detail["frames"] == [7, 29]
    assert detail["size_source"] == "captions.size_px"
    assert detail["median_stem_px"] == stem
    assert all(margin > 0 for margin in detail["clear_margins"].values())
    assert all(sample["glyph_pixels"] > 500 for sample in detail["samples"])


@pytest.mark.parametrize("stem,hairline,reason", [(14, None, "weight"), (11, 5, "family")])
def test_font_class_refuses_ambiguous_statistics(
    stem: int, hairline: int | None, reason: str,
) -> None:
    value, frames, detail = font_class([(17, _stroke_glyphs(stem, hairline))], size_px=100)
    assert value is None and frames == [17]
    assert reason in detail["reason"] and "threshold" in detail["reason"]
    assert detail["clear_margins"]["weight_ratio" if reason == "weight" else
                                   "contrast_ratio"] <= 0


def test_font_class_requires_pixels_and_labels_size_fallback() -> None:
    value, frames, detail = font_class([(1, np.zeros((100, 1080), dtype=bool))], size_px=100)
    assert value is None and not frames and "Too few" in detail["reason"]
    mask = _stroke_glyphs(8)
    value, _, detail = font_class([(3, mask)])
    assert value == {"family": "sans", "weight": "regular"}
    assert detail["size_source"] == "estimated glyph x-height"
    assert detail["size_used_px"] == 70
    mask = np.concatenate((mask, np.zeros((20, 1080), dtype=bool), mask))
    value, _, detail = font_class([(3, mask)])
    assert value == {"family": "sans", "weight": "regular"}
    assert detail["size_source"] == "measured line pitch"


def test_shape_aggregation_requires_equal_classes_and_radius_tolerance() -> None:
    first = {"family": "sans", "weight": "regular"}
    second = {"family": "serif", "weight": "regular"}
    agreed = aggregate("captions.font_class", [{"value": first}, {"value": first.copy()}], "test")
    assert agreed["status"] == "proposed" and agreed["value"] == first
    varied = aggregate("captions.font_class", [{"value": first}, {"value": second}], "test")
    assert varied["status"] == "varies" and varied["value"] is None
    assert tolerance("graphics.corner_radius_px") == CORNER_RADIUS_TOLERANCE == 3
    for difference, expected in ((3, "proposed"), (4, "varies")):
        result = aggregate("graphics.corner_radius_px", [
            {"value": 24}, {"value": 24 + difference}
        ], "test")
        assert result["status"] == expected


def test_shape_measure_decodes_bounded_full_resolution_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from yap_studio.taste import measure_shapes

    ink = _stroke_glyphs(8)
    caption = np.full((*ink.shape, 3), 45, dtype=np.uint8)
    caption[ink] = rgb("#F8F4EC")
    card_picture, card = _rounded_card_picture((24,) * 4)
    calls = []

    def pictures(path, info, indices, *, width):
        assert width == 1080
        calls.append(list(indices))
        for index in indices:
            yield index, caption if len(calls) == 1 else card_picture

    monkeypatch.setattr(measure_shapes, "iter_rgb", pictures)
    observations, missing = measure_shapes.measure(
        Path("synthetic.mp4"), SimpleNamespace(fps=30),
        {"captions.text_color": ("#F8F4EC", list(range(120)), {}),
         "captions.size_px": (100, [1], {}),
         "graphics.paper_color": ("#F8F4EC", [20], {})},
        {"captions.band": ({"top_px": 15, "bottom_px": 85}, [1], {}),
         "graphics.cards": ([card], [10, 40], {})},
    )
    assert not missing
    assert len(calls[0]) == 24 and calls[0][0] == 0 and calls[0][-1] == 119
    assert calls[1] == [18, 25, 32]
    assert observations["captions.font_class"][0] == {"family": "sans", "weight": "regular"}
    assert observations["captions.font_class"][1] == calls[0]
    assert observations["graphics.corner_radius_px"][0] == pytest.approx(24, abs=3)
    assert observations["graphics.corner_radius_px"][1] == calls[1]


def _ffmpeg(args: list[str], *, data: bytes | None = None) -> bytes:
    """Run a bounded media command, with a pipe only for raw input."""
    options = {"input": data} if data is not None else {"stdin": subprocess.DEVNULL}
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        timeout=30,
        check=True,
        **options,
    )
    return result.stdout


def _probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return taste_finals.build_source(
        tmp_path_factory.mktemp("taste-finals") / "source.mov",
        style=resolve("preset:electric"),
        size=_SIZE,
    )


def test_colours_round_trip_through_bt709_encode(tmp_path: Path) -> None:
    colors = [
        resolve(f"preset:{preset}").data["graphics"][key]
        for preset in SHIPPED_PRESETS
        for key in ("paper_color", "ink_color", "accent_color")
    ]
    colors.extend(("#000000", "#FFFFFF"))
    width, height = len(colors) * 32, 32
    yuv = [taste_finals.hex_to_bt709(color) for color in colors]
    raw = b"".join(
        np.concatenate(
            [
                np.full((height // subsample, 32 // subsample), value[component], dtype=np.uint8)
                for value in yuv
            ],
            axis=1,
        ).tobytes()
        for component, subsample in ((0, 1), (1, 2), (2, 2))
    )
    encoded = tmp_path / "colours.mp4"
    _ffmpeg(
        [
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "yuv420p",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            "30",
            "-i",
            "pipe:0",
            "-map",
            "0:v:0",
            "-vf",
            "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
            "-frames:v",
            "1",
            "-c:v",
            "libx264",
            "-crf",
            "4",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
            str(encoded),
        ],
        data=raw,
    )
    decoded = _ffmpeg(
        [
            "-nostdin",
            "-i",
            str(encoded),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    rgb = np.frombuffer(decoded, dtype=np.uint8).reshape(height, width, 3).astype(int)
    for index, color in enumerate(colors):
        expected = np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)])
        interior = rgb[4:-4, index * 32 + 4 : (index + 1) * 32 - 4]
        assert np.max(np.abs(interior - expected)) <= 3, color


def _assert_source(path: Path, size: tuple[int, int], seconds: float, frames: int) -> None:
    info = _probe(path)
    video = next(stream for stream in info["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in info["streams"] if stream["codec_type"] == "audio")
    assert (video["width"], video["height"]) == size
    assert int(video["nb_frames"]) == int(video["nb_read_frames"]) == frames
    assert video["r_frame_rate"] == video["avg_frame_rate"] == "30/1"
    assert video["codec_name"] == "h264"
    assert video["pix_fmt"] == "yuv420p"
    assert video["color_range"] == "tv"
    for tag in ("color_space", "color_transfer", "color_primaries"):
        assert video[tag] == "bt709"
    for stream in (video, audio, info["format"]):
        assert float(stream["duration"]) == pytest.approx(seconds, abs=1e-6)
    assert audio["codec_name"] == "pcm_s16le"
    assert audio["sample_rate"] == "48000"
    assert audio["channels"] == 2
    assert int(audio["duration_ts"]) == round(seconds * 48_000)


def test_source_timeline_and_tags(source: Path) -> None:
    _assert_source(source, _SIZE, seconds=12, frames=360)


def test_source_default_size_in_a_short_clip(tmp_path: Path) -> None:
    clip = taste_finals.build_source(
        tmp_path / "full-size.mov", style=resolve("preset:clean_minimal"), seconds=0.1
    )
    _assert_source(clip, (1080, 1920), seconds=0.1, frames=3)


def test_punch_in_is_a_scaled_continuation_of_b(source: Path) -> None:
    width, height = _SIZE
    decoded = _ffmpeg(
        [
            "-nostdin",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            "select='eq(n,180)'",
            "-frames:v",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    actual = np.frombuffer(decoded[: width * height], dtype=np.uint8).reshape(height, width)
    continued = taste_finals.source_frame(
        180, style=resolve("preset:electric"), size=_SIZE, shot="B"
    )
    cropped = _ffmpeg(
        [
            "-f",
            "rawvideo",
            "-pixel_format",
            "yuv420p",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            "30",
            "-i",
            "pipe:0",
            "-map",
            "0:v:0",
            "-vf",
            f"crop=144:256:18:32,scale={width}:{height}:flags=bilinear",
            "-frames:v",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        data=continued,
    )
    zoomed = np.frombuffer(cropped[: width * height], dtype=np.uint8).reshape(height, width)
    unscaled = np.frombuffer(continued[: width * height], dtype=np.uint8).reshape(height, width)
    zoom_error = np.mean(np.abs(actual.astype(float) - zoomed))
    unscaled_error = np.mean(np.abs(actual.astype(float) - unscaled))
    assert zoom_error < 2
    assert zoom_error < 0.25 * unscaled_error


def test_card_window_and_bars(source: Path) -> None:
    width, height = _SIZE
    raw = _ffmpeg(
        [
            "-nostdin",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            "select='eq(n,194)+eq(n,195)+eq(n,269)+eq(n,270)'",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(4, height, width, 3).astype(int)
    graphics = resolve("preset:electric").data["graphics"]
    for x, y, color in (
        (850, 320, graphics["paper_color"]),
        (300, 374, graphics["ink_color"]),
        (300, 444, graphics["ink_color"]),
        (300, 514, graphics["ink_color"]),
        (200, 584, graphics["accent_color"]),
    ):
        pixel_x, pixel_y = round(x * width / 1080), round(y * height / 1920)
        expected = np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)])
        for index in (1, 2):
            assert np.max(np.abs(frames[index, pixel_y, pixel_x] - expected)) <= 3
    paper_pixel = frames[:, round(320 * height / 1920), round(850 * width / 1080)]
    assert np.linalg.norm(paper_pixel[0] - paper_pixel[1]) > 50
    assert np.linalg.norm(paper_pixel[3] - paper_pixel[2]) > 50
    full_width, full_height = taste_finals.SIZE
    paper_luma = taste_finals.hex_to_bt709(graphics["paper_color"])[0]
    for frame in (195, 269):
        picture = taste_finals.source_frame(
            frame, style=resolve("preset:electric"), size=(full_width, full_height)
        )
        luma = np.frombuffer(picture, dtype=np.uint8, count=full_width * full_height).reshape(
            full_height, full_width
        )
        assert luma[288, 68] == paper_luma
        corner = luma[280:283, 60:63].astype(float)
        background = luma[279, 60:63].astype(float)
        assert np.max(np.abs(corner - background)) <= 5
        assert np.min(np.abs(corner - paper_luma)) > 30


@pytest.mark.parametrize("preset", SHIPPED_PRESETS)
def test_card_corner_missing_paper_area(preset: str, monkeypatch: pytest.MonkeyPatch) -> None:
    style = resolve(f"preset:{preset}")
    width, height = taste_finals.SIZE
    picture = taste_finals.source_frame(200, style=style, size=(width, height))
    luma = np.frombuffer(picture, dtype=np.uint8, count=width * height).reshape(height, width)
    scene = taste_finals._Scene(style.data["graphics"], (width, height))
    # Suppress only the card, preserving the same frame's zoom, gradient and luma noise.
    monkeypatch.setattr(scene, "_card", lambda plane, component: None)
    raw_background = scene.frame(200)
    background = np.frombuffer(raw_background, dtype=np.uint8, count=width * height).reshape(
        height, width
    )
    paper = taste_finals.hex_to_bt709(style.data["graphics"]["paper_color"])[0]
    radius = taste_finals.CARD_RADIUS
    expected = radius**2 * (1 - math.pi / 4)
    for x in (60, 1020 - radius):
        for y in (280, 640 - radius):
            under = background[y : y + radius, x : x + radius].astype(float)
            pixels = luma[y : y + radius, x : x + radius].astype(float)
            coverage = (pixels - under) / (paper - under)
            assert np.any((coverage > 0) & (coverage < 1))
            assert np.sum(1 - coverage) == pytest.approx(expected, rel=0.03)


def test_audio_has_exact_samples_speech_and_pause_floor(source: Path) -> None:
    raw = _ffmpeg(
        [
            "-nostdin",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_s16le",
            "-f",
            "s16le",
            "pipe:1",
        ]
    )
    samples = np.frombuffer(raw, dtype="<i2").reshape(-1, 2).astype(float) / 32767
    assert samples.shape == (576_000, 2)
    assert np.max(np.abs(samples)) < 10 ** (-3 / 20)
    for start, length in ((1.8, 0.3), (3.6, 0.45), (6.0, 0.6), (8.4, 0.3), (9.9, 0.45)):
        pause = samples[round(start * 48_000) : round((start + length) * 48_000)]
        rms = np.sqrt(np.mean(pause**2))
        assert 20 * np.log10(rms) == pytest.approx(-70, abs=0.2)
    speech = samples[round(4.2 * 48_000) : round(5.2 * 48_000)]
    assert np.sqrt(np.mean(speech**2)) > 0.05
    power = np.abs(np.fft.rfft(speech[:, 0])) ** 2
    frequencies = np.fft.rfftfreq(len(speech), 1 / 48_000)
    band_power = power[(frequencies >= 190) & (frequencies <= 4010)].sum()
    assert band_power / power.sum() > 0.98


def test_words_have_fixed_pauses_and_two_wide_phrases(tmp_path: Path) -> None:
    path = taste_finals.build_words(tmp_path / "words.json")
    transcript = Transcript.load(path)
    assert len(transcript.words) == 31
    assert transcript.duration_s == 12
    assert transcript_pacing(transcript, 12)["pacing.words_per_minute"] == 155
    assert transcript.words[0].start == 0
    assert transcript.words[-1].end == 12
    gaps = [
        (previous.end, following.start - previous.end)
        for previous, following in zip(transcript.words, transcript.words[1:], strict=False)
        if following.start > previous.end
    ]
    assert len(gaps) == 5
    assert [start for start, _ in gaps] == pytest.approx([1.8, 3.6, 6.0, 8.4, 9.9])
    assert [length for _, length in gaps] == pytest.approx([0.3, 0.45, 0.6, 0.3, 0.45])
    assert taste_finals.pauses_s() == [
        {"start_s": start, "length_s": length}
        for start, length in ((1.8, 0.3), (3.6, 0.45), (6.0, 0.6), (8.4, 0.3), (9.9, 0.45))
    ]
    for previous, following in zip(transcript.words, transcript.words[1:], strict=False):
        assert previous.start < previous.end <= following.start
        if following.start > previous.end:
            assert previous.text.endswith(".")
    assert transcript.words[-1].text.endswith(".")
    assert " ".join(word.text for word in transcript.words[9:21]) == (
        "Go up do it now go up do it now go up."
    )
    wide_line = "MMMMMMMM WWWWWWWW."
    assert transcript.text.count(wide_line) == 2
    assert len(wide_line) < 21
    words = [
        CaptionWord(f"w{i}", word.text, word.start, word.end)
        for i, word in enumerate(transcript.words)
    ]
    for preset in SHIPPED_PRESETS:
        captions = resolve(f"preset:{preset}").data["captions"]
        phrases = build_phrases(words, captions)
        word_limit = captions["max_words_per_phrase"]
        character_limit = captions["max_characters_per_phrase"]
        for start, end, line in (
            (2.1, 3.6, wide_line),
            (6.6, 8.4, "Unstoppable momentum."),
            (8.7, 9.9, wide_line),
        ):
            tokens = line.split()
            # build_phrases counts spaces and punctuation and keeps overlong single words.
            fits = len(tokens) <= word_limit and len(line) <= character_limit
            expected = [line] if fits else tokens
            assert [phrase.text for phrase in phrases if start <= phrase.start < end] == (
                expected
            ), preset
        counts = [len(phrase.words) for phrase in phrases]
        short_run = [phrase for phrase in phrases if 4.05 <= phrase.start < 6]
        # Only the two occurrences of "now" have enough letters to receive stress.
        assert [
            word.text
            for phrase in short_run
            for word in phrase.words
            if word.id == phrase.stress_id
        ] == ["now", "now"], preset
        expected_unstressed = len(short_run) - 2
        assert sum(phrase.stress_id is None for phrase in short_run) == expected_unstressed
        consistent_max_words = {
            max_words
            for max_words in range(1, 9)
            if any(
                [
                    len(phrase.words)
                    for phrase in build_phrases(
                        words,
                        {
                            **captions,
                            "max_words_per_phrase": max_words,
                            "max_characters_per_phrase": max_chars,
                        },
                    )
                ]
                == counts
                for max_chars in range(8, 49)
            )
        }
        assert consistent_max_words == {captions["max_words_per_phrase"]}


@pytest.mark.parametrize(
    "preset,family,weight",
    [
        ("clean_minimal", "sans", "regular"),
        ("electric", "sans", "heavy"),
        ("warm_editorial", "serif", "regular"),
        ("cobalt", "sans", "regular"),
        ("mono", "sans", "regular"),
        ("headline", "sans", "heavy"),
        ("candy", "sans", "heavy"),
        ("marker", "sans", "heavy"),
        ("hf_bold", "sans", "heavy"),
        ("hf_editorial", "serif", "regular"),
        ("hf_neutral", "sans", "regular"),
    ],
)
def test_final_truth_caption_measurements_without_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preset: str, family: str, weight: str
) -> None:
    from yap_studio.stages import captions as captions_stage

    monkeypatch.setattr(taste_finals, "build_source", lambda path, **kwargs: path)
    monkeypatch.setattr(captions_stage, "caption_clip", lambda *args, **kwargs: None)
    truth = taste_finals.build_final(preset, tmp_path)
    data = json.loads((tmp_path / "truth.json").read_text())
    assert data == truth.to_json()
    assert data["final"] == "final.mp4" and data["words"] == "words.json"
    assert data["font_class"] == {"family": family, "weight": weight}
    assert data["card"]["radius"] == taste_finals.CARD_RADIUS == 24
    phrases = build_phrases(
        [
            CaptionWord(f"w{i}", word.text, word.start, word.end)
            for i, word in enumerate(Transcript.load(truth.words).words)
        ],
        resolve(f"preset:{preset}").data["captions"],
    )
    assert data["phrases"] == [
        {
            "words": len(phrase.words),
            "text": phrase.text,
            "start_s": phrase.start,
            "end_s": phrase.end,
        }
        for phrase in phrases
    ]



@pytest.mark.parametrize("preset,phrases", [
    ("hf_bold", ["Small steady steps", "make real", "progress today.",
                 "MMMMMMMM WWWWWWWW.", "Go up do", "it now go", "up do it", "now go up.",
                 "Unstoppable", "momentum.", "MMMMMMMM WWWWWWWW.",
                 "We find", "our rhythm", "together today."]),
    ("hf_editorial", ["Small steady steps make", "real progress today.",
                      "MMMMMMMM WWWWWWWW.", "Go up do it now", "go up do it", "now go up.",
                      "Unstoppable momentum.", "MMMMMMMM WWWWWWWW.",
                      "We find our rhythm", "together today."]),
    ("hf_neutral", ["Small steady steps", "make real", "progress today.",
                    "MMMMMMMM WWWWWWWW.", "Go up do it now go", "up do it now go up.",
                    "Unstoppable momentum.", "MMMMMMMM WWWWWWWW.",
                    "We find our rhythm", "together today."]),
])
def test_theme_final_phrase_expectations(tmp_path: Path, preset: str, phrases: list[str]) -> None:
    transcript = Transcript.load(taste_finals.build_words(tmp_path / "words.json"))
    words = [CaptionWord(f"w{i}", word.text, word.start, word.end)
             for i, word in enumerate(transcript.words)]
    assert [phrase.text for phrase in build_phrases(
        words, resolve(f"preset:{preset}").data["captions"],
    )] == phrases


@pytest.mark.parametrize("preset", SHIPPED_PRESETS)
def test_shipped_preset_caption_font_has_final_class(preset: str) -> None:
    font = resolve(f"preset:{preset}").data["captions"]["font"]
    assert font in taste_finals.FONT_CLASSES, f"{preset}: unmapped caption font {font!r}"


def test_default_finals_cover_shipped_presets_without_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from yap_studio.stages import captions as captions_stage

    monkeypatch.setattr(taste_finals, "build_source", lambda path, **kwargs: path)
    monkeypatch.setattr(captions_stage, "caption_clip", lambda *args, **kwargs: None)
    assert taste_finals.main([str(tmp_path)]) == 0
    assert {path.parent.name for path in tmp_path.glob("*/truth.json")} == set(SHIPPED_PRESETS)


def test_final_truth_rejects_unmapped_caption_font(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(taste_finals.FONT_CLASSES, "Inter")
    with pytest.raises(ValueError, match="unmapped caption font 'Inter'"):
        taste_finals.build_final("clean_minimal", tmp_path)


CUT_FRAME_TOLERANCE = 1
BOX_TOLERANCE = 8


def test_lab_reference_values_and_delta_e() -> None:
    assert lab([255, 255, 255]) == pytest.approx([100, 0, 0], abs=0.001)
    assert lab([0, 0, 0]) == pytest.approx([0, 0, 0])
    assert lab([255, 0, 0]) == pytest.approx([53.2408, 80.0925, 67.2032], abs=0.001)
    assert delta_e("#FFFFFF", "#FFFFFF") == 0
    assert delta_e("#FFFFFF", "#000000") == pytest.approx(100, abs=0.001)
    assert hex_color(rgb("#FFF4DD")) == "#FFF4DD"


def _glyphs(image: np.ndarray, y: int, color: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(image.shape[:2], dtype=bool)
    for x in range(60, 180, 20):
        mask[y : y + 16, x : x + 4] = True
        mask[y : y + 4, x : x + 12] = True
        mask[y + 7 : y + 11, x : x + 10] = True
    image[mask] = color
    return mask


def _word_window_picture(*, y: int = 508, right: int = 180, lines: int = 1) -> np.ndarray:
    """Draw fixed glyphs on a plain picture without invoking a renderer."""
    image = np.full((640, 360, 3), 45, dtype=np.uint8)
    for row in range(lines):
        _glyphs(image, y - row * 30, (255, 244, 221))
    image[:, right:] = 45
    return image


def test_caption_windows_ignore_background_and_keep_entry_rise() -> None:
    blank = np.full((640, 360, 3), 45, dtype=np.uint8)
    blank[490:540, 80:270] = (255, 244, 221)
    blank[508:516, 300:302] = (255, 244, 221)
    _glyphs(blank, 110, (255, 244, 221))  # Card/title ink outside the caption band.
    pictures = [blank, _word_window_picture(y=512), _word_window_picture(), blank,
                _word_window_picture(), _word_window_picture(), blank]
    windows = caption_windows(
        enumerate(pictures), {"top_px": 1524, "bottom_px": 1572},
        {"captions.text_color": "#FFF4DD"}, reference_frames=[2, 4],
    )
    assert windows == [{"start_frame": 1, "end_frame": 2},
                       {"start_frame": 4, "end_frame": 5}]
    assert caption_windows(enumerate([blank] * 3), {"top_px": 1524, "bottom_px": 1572},
                           {"captions.text_color": "#FFF4DD"}) == []


@pytest.mark.parametrize("change", ["width", "lines"])
def test_caption_windows_split_zero_gap_layout_change(change: str) -> None:
    first = _word_window_picture()
    second = _word_window_picture(right=120) if change == "width" else _word_window_picture(lines=2)
    windows = caption_windows(
        enumerate([first] * 4 + [second] * 4), {"top_px": 1434, "bottom_px": 1572},
        {"captions.text_color": "#FFF4DD"},
    )
    assert windows == [{"start_frame": 0, "end_frame": 3},
                       {"start_frame": 4, "end_frame": 7}]


@pytest.mark.parametrize("fps", [10, 30, 60])
@pytest.mark.parametrize("entry", ["first_frame", "rise"])
def test_caption_windows_use_settled_entry_layout(fps: int, entry: str) -> None:
    from yap_studio.taste.frames import FrameTiming

    entry_frames = max(3, math.ceil(0.15 * fps))
    entry_length = 1 if entry == "first_frame" else entry_frames - 1
    pictures = [_word_window_picture(y=512, right=120)] * entry_length
    pictures += [_word_window_picture()] * (entry_frames + 8)
    windows = caption_windows(
        enumerate(pictures), {"top_px": 1524, "bottom_px": 1572},
        {"captions.text_color": "#FFF4DD"}, fps=fps,
    )
    assert windows == [{"start_frame": 0, "end_frame": len(pictures) - 1}]
    timing = FrameTiming(tuple(i / fps for i in range(len(pictures))),
                         len(pictures) / fps, True, False)
    transcript = Transcript([Word("first", 0, 1 / fps),
                             Word("settled", entry_frames / fps, (entry_frames + 1) / fps)])
    assigned, outside = assign_words(windows, transcript, timing)
    assert [window["words"] for window in assigned] == [2] and outside == []


@pytest.mark.parametrize("component", ["low_mass", "short_height"])
def test_caption_windows_ignore_small_flickering_lines(component: str) -> None:
    from yap_studio.taste.measure_words import _ink_layout

    settled = _word_window_picture()
    stray = settled.copy()
    if component == "low_mass":
        stray[480:496, 145:149] = (255, 244, 221)
        stray[480:496, 157:161] = (255, 244, 221)
    else:
        _glyphs(stray, 475, (255, 244, 221))
        stray[480:491] = 45  # Comparable mass, but only five rows high.
    band = {"top_px": 1410, "bottom_px": 1572}
    colors = {"captions.text_color": "#FFF4DD"}
    assert _ink_layout(stray, band, colors) == _ink_layout(settled, band, colors)
    pictures = [settled] * 8 + [stray] * 5 + [settled] + [stray] * 3 + [settled] * 5
    assert caption_windows(enumerate(pictures), band, colors, fps=30) == [
        {"start_frame": 0, "end_frame": len(pictures) - 1}
    ]


def test_caption_windows_require_three_consecutive_changed_frames() -> None:
    settled = _word_window_picture()
    changed = _word_window_picture(right=120)
    pictures = [settled] * 6 + [changed] * 2 + [settled] + [changed] * 2 + [settled]
    assert caption_windows(
        enumerate(pictures), {"top_px": 1524, "bottom_px": 1572},
        {"captions.text_color": "#FFF4DD"}, fps=30,
    ) == [{"start_frame": 0, "end_frame": len(pictures) - 1}]


def test_caption_windows_merge_short_contiguous_windows_forward() -> None:
    settled = _word_window_picture()
    narrow = _word_window_picture(right=120)
    pictures = [settled] * 6 + [narrow] * 3 + [settled] * 8
    assert caption_windows(
        enumerate(pictures), {"top_px": 1524, "bottom_px": 1572},
        {"captions.text_color": "#FFF4DD"}, fps=30,
    ) == [{"start_frame": 0, "end_frame": 5},
          {"start_frame": 6, "end_frame": len(pictures) - 1}]


@pytest.mark.parametrize("style", ["color", "underline", "moving_block"])
def test_caption_windows_do_not_split_active_highlight_changes(style: str) -> None:
    pictures = []
    for left in (60, 100, 160, 60):
        image = _word_window_picture()
        glyphs = np.all(image == (255, 244, 221), axis=2)
        active = glyphs & (np.indices(glyphs.shape)[1] >= left)
        active &= np.indices(glyphs.shape)[1] < left + 20
        if style == "color":
            image[active] = (210, 255, 45)
        elif style == "underline":
            image[526:528, left : left + 16] = (210, 255, 45)
        else:
            image[504:528, left - 3 : left + 17] = (210, 255, 45)
            image[active] = (17, 17, 17)
        pictures.extend([image] * 3)
    windows = caption_windows(
        enumerate(pictures), {"top_px": 1524, "bottom_px": 1572},
        {"captions.text_color": "#FFF4DD", "captions.highlight_style": style,
         "captions.highlight_color": "#D2FF2D", "captions.highlight_text_color": "#111111"},
    )
    assert windows == [{"start_frame": 0, "end_frame": 11}]


def test_window_word_assignment_uses_presentation_times_and_entry_tolerance() -> None:
    from yap_studio.taste.frames import FrameTiming

    timing = FrameTiming((0, 0.04, 0.13, 0.18, 0.31, 0.4, 0.55, 0.72), 0.9, True, True)
    # Keep this previous-phrase word outside the next window's entry tolerance.
    transcript = Transcript([Word("first", 0, 0.05), Word("still", 0.14, 0.18),
                             Word("second", 0.3, 0.35),
                             Word("late", 0.95, 1)])
    windows, outside = assign_words(
        [{"start_frame": 2, "end_frame": 2}, {"start_frame": 4, "end_frame": 6}],
        transcript, timing,
    )
    assert [window["words"] for window in windows] == [2, 1]
    assert outside == [{"index": 3, "text": "late", "start_s": 0.95}]


@pytest.mark.parametrize("lead_s,expected", [
    pytest.param(11 / 30 - 0.35, [1, 1], id="next-caption-word-at-0.35s"),
    pytest.param(11 / 30 - 0.25, [2, 0], id="current-caption-word-at-0.25s"),
    pytest.param(0.15, [2, 0], id="within-tolerance-before-last-frame"),
    pytest.param(0.1625, [2, 0], id="previous-phrase-word"),
    pytest.param(1 / 30, [1, 1], id="last-frame-boundary"),
    pytest.param(1 / 30 + 0.5e-9, [1, 1], id="last-frame-boundary-epsilon"),
    pytest.param(1 / 30 + 2e-9, [2, 0], id="before-last-frame"),
])
def test_window_word_assignment_handles_entry_over_previous_caption(
    lead_s: float, expected: list[int],
) -> None:
    from yap_studio.taste.frames import FrameTiming

    timing = FrameTiming(tuple(i / 30 for i in range(24)), 24 / 30, True, False)
    start = 11 / 30 - lead_s
    transcript = Transcript([Word("first", 0, 0.1), Word("boundary", start, start + 0.1)])
    windows, outside = assign_words(
        [{"start_frame": 0, "end_frame": 10}, {"start_frame": 11, "end_frame": 23}],
        transcript, timing,
    )
    assert [window["words"] for window in windows] == expected
    assert outside == []


@pytest.mark.parametrize("last_frame,start,expected", [
    pytest.param(3, 0.175, [2, 0], id="before-short-last-frame"),
    pytest.param(3, 0.18, [1, 1], id="short-last-frame-boundary"),
    pytest.param(4, 0.195, [2, 0], id="before-long-last-frame"),
    pytest.param(4, 0.21, [1, 1], id="inside-long-last-frame"),
    pytest.param(2, 0.16, [1, 1], id="entry-tolerance-boundary"),
    pytest.param(2, 0.16 - 0.5e-9, [1, 1], id="entry-tolerance-epsilon"),
    pytest.param(2, 0.16 - 2e-9, [2, 0], id="outside-entry-tolerance"),
])
def test_window_word_assignment_uses_last_frame_duration(
    last_frame: int, start: float, expected: list[int],
) -> None:
    from yap_studio.taste.frames import FrameTiming

    timing = FrameTiming((0, 0.1, 0.15, 0.18, 0.2, 0.26, 0.31, 0.4), 0.5, True, True)
    transcript = Transcript([Word("first", 0, 0.1), Word("boundary", start, start + 0.1)])
    windows, outside = assign_words(
        [{"start_frame": 0, "end_frame": last_frame}, {"start_frame": 6, "end_frame": 7}],
        transcript, timing,
    )
    assert [window["words"] for window in windows] == expected
    assert outside == []


def test_caption_measure_keeps_counts_when_a_word_is_outside_every_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from yap_studio.taste import measure_words
    from yap_studio.taste.frames import FrameTiming

    picture = _word_window_picture()
    blank = np.full_like(picture, 45)
    pictures = [blank] * 2 + [picture] * 3 + [blank] * 5
    monkeypatch.setattr(measure_words, "iter_rgb", lambda *args, **kwargs: enumerate(pictures))
    timing = FrameTiming(tuple(i / 10 for i in range(10)), 1, True, False)
    transcript = Transcript([Word("shown", 0.2, 0.3), Word("outside", 0.8, 0.9)])
    fields, measures, missing = measure_words.measure(
        Path("synthetic.mp4"), SimpleNamespace(video_frames=10),
        {"captions.text_color": ("#FFF4DD", [2], {})},
        {"captions.band": ({"top_px": 1524, "bottom_px": 1572}, [2], {})},
        transcript, timing=timing,
    )
    assert fields == {}
    value, frames, detail = measures["captions.words_per_caption"]
    assert value == {"counts": [1], "windows": [{"start_frame": 2, "end_frame": 4, "words": 1}]}
    assert frames == [2, 3, 4] and detail["observed_counts"] == [1]
    assert len(missing) == 2 and "outside every caption window" in missing[0]["reason"]
    assert "'outside'" in missing[0]["reason"] and "0.8 s" in missing[0]["reason"]
    assert missing[0]["detail"]["observed_counts"] == [1]


@pytest.mark.parametrize("case", ["leading", "trailing", "isolated"])
def test_caption_measure_ignores_windows_without_word_starts(
    case: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from yap_studio.taste import measure_words
    from yap_studio.taste.frames import FrameTiming

    first, second = _word_window_picture(), _word_window_picture(right=120)
    pictures = [first] * 6
    if case == "isolated":
        pictures += [np.full_like(first, 45)] * 6
    pictures += [second] * 6
    monkeypatch.setattr(measure_words, "iter_rgb", lambda *args, **kwargs: enumerate(pictures))
    start = 0.2 if case == "leading" else 0
    timing = FrameTiming(tuple(i / 30 for i in range(len(pictures))),
                         len(pictures) / 30, True, False)
    _, measures, _ = measure_words.measure(
        Path("synthetic.mp4"), SimpleNamespace(fps=30),
        {"captions.text_color": ("#FFF4DD", [3], {})},
        {"captions.band": ({"top_px": 1524, "bottom_px": 1572}, [3], {})},
        Transcript([Word("phrase.", start, start + 0.1)]), timing=timing,
    )
    value, frames, detail = measures["captions.words_per_caption"]
    end = 5 if case == "isolated" else 11
    assert value == {"counts": [1], "windows": [
        {"start_frame": 0, "end_frame": end, "words": 1}
    ]}
    assert frames == list(range(end + 1))
    assert detail["observed_counts"] == [1] and detail["outside_words"] == []
    ignored_start = {"leading": 0, "trailing": 6, "isolated": 12}[case]
    assert detail["ignored_windows"] == [{
        "start_frame": ignored_start, "end_frame": ignored_start + 5, "words": 0,
        "reason": "No word starts in this window.",
        "action": {"leading": "merged_next", "trailing": "merged_previous",
                   "isolated": "dropped"}[case],
    }]


@pytest.mark.parametrize("preset", SHIPPED_PRESETS)
def test_phrase_limits_identify_each_presets_word_limit(tmp_path: Path, preset: str) -> None:
    transcript = Transcript.load(taste_finals.build_words(tmp_path / "words.json"))
    words = [CaptionWord(f"w{i}", w.text, w.start, w.end)
             for i, w in enumerate(transcript.words)]
    captions = resolve(f"preset:{preset}").data["captions"]
    counts = [len(phrase.words) for phrase in build_phrases(words, captions)]
    fields, missing = infer_limits(transcript, counts, [2, 3, 4, 8, 9])
    value, frames, detail = fields[LIMIT_FIELDS[0]]
    assert value == captions["max_words_per_phrase"]
    assert frames == [2, 3, 4, 8, 9]
    assert detail["observed_counts"] == counts
    assert detail["consistent_max_words"] == [value]
    lower, upper = detail["consistent_max_chars_range"]
    assert lower <= captions["max_characters_per_phrase"] <= upper
    assert f"{lower} to {upper} characters" in missing[0]["reason"]
    assert missing[0]["field"] == LIMIT_FIELDS[1]


def test_phrase_limits_explain_ambiguity_and_unbuildable_partition() -> None:
    transcript = Transcript([Word(text, i * 0.2, (i + 1) * 0.2)
                             for i, text in enumerate("bright minds learn fast.".split())])
    fields, missing = infer_limits(transcript, [4], [1, 2])
    assert fields == {}
    assert "Consistent with 4 to 8 words" in missing[0]["reason"]
    assert missing[0]["detail"]["consistent_max_words"] == [4, 5, 6, 7, 8]
    assert missing[0]["field"] == LIMIT_FIELDS[0]
    assert missing[1]["field"] == LIMIT_FIELDS[1]
    assert "Consistent with 24 to 48 characters" in missing[1]["reason"]
    fields, missing = infer_limits(transcript, [1, 3], [1, 2])
    assert fields == {}
    assert "do not match any phrase limits" in missing[0]["reason"]
    assert missing[0]["detail"]["observed_counts"] == [1, 3]


def test_phrase_limits_identify_characters_when_word_limit_varies() -> None:
    lengths = [7, 8, 11, 5, 11, 12, 10, 9, 12, 5, 10, 12]
    transcript = Transcript([
        Word("x" * (length - (i == 5)) + ("." if i == 5 else ""), i * 0.1, (i + 1) * 0.1)
        for i, length in enumerate(lengths)
    ])
    assert [len(word.text) for word in transcript.words] == lengths
    counts = [2, 2, 1, 1, 1, 1, 1, 2, 1]
    words = [CaptionWord(f"w{i}", w.text, w.start, w.end)
             for i, w in enumerate(transcript.words)]
    pairs = [
        (max_words, max_chars)
        for max_words in range(1, 9)
        for max_chars in range(8, 49)
        if [len(phrase.words) for phrase in build_phrases(words, {
            "max_words_per_phrase": max_words, "max_characters_per_phrase": max_chars,
        })] == counts
    ]
    assert pairs == [(max_words, 17) for max_words in range(2, 9)]
    assert consistent_limits(transcript, counts) == pairs
    fields, missing = infer_limits(transcript, counts, [1, 2, 3])
    assert set(fields) == {LIMIT_FIELDS[1]}
    value, evidence, detail = fields[LIMIT_FIELDS[1]]
    assert value == 17 and evidence == [1, 2, 3]
    assert detail["consistent_max_words"] == list(range(2, 9))
    assert detail["consistent_max_chars"] == [17]
    assert len(missing) == 1 and missing[0]["field"] == LIMIT_FIELDS[0]
    assert missing[0]["reason"].startswith("Consistent with 2 to 8 words;")


def test_cached_phrase_limit_search_equals_brute_force() -> None:
    rng = np.random.default_rng(30)
    transcript = Transcript([
        Word("x" * int(length) + ("." if i == 11 else ""), i * 0.1, (i + 1) * 0.1)
        for i, length in enumerate(rng.integers(1, 14, 18))
    ])
    words = [CaptionWord(f"w{i}", w.text, w.start, w.end)
             for i, w in enumerate(transcript.words)]
    partitions: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for max_words in range(1, 9):
        for max_chars in range(8, 49):
            counts = tuple(len(phrase.words) for phrase in build_phrases(
                words, {"max_words_per_phrase": max_words, "max_characters_per_phrase": max_chars}
            ))
            partitions.setdefault(counts, []).append((max_words, max_chars))
    for counts, pairs in partitions.items():
        assert consistent_limits(transcript, counts) == pairs
    unique = next((counts, pairs[0]) for counts, pairs in partitions.items() if len(pairs) == 1)
    counts, (max_words, max_chars) = unique
    fields, missing = infer_limits(transcript, counts, [1, 2, 3])
    assert fields[LIMIT_FIELDS[0]][0] == max_words
    assert fields[LIMIT_FIELDS[1]][0] == max_chars
    assert fields[LIMIT_FIELDS[1]][1] == [1, 2, 3]
    assert missing == []


def test_highlight_samples_recognizes_warm_editorial_stress_highlights() -> None:
    highlighted = [15, 135, 159, 183, 279, 303]
    unhighlighted = [87, 111, 207, 231, 255, 327, 351]
    samples = sorted([(i, "underline") for i in highlighted] +
                     [(i, "none") for i in unhighlighted])
    assert _highlight_samples(samples) == ("underline", highlighted, {})


@pytest.mark.parametrize("looks,expected", [
    pytest.param(["color"] * 12 + ["none"], "color", id="mostly-color"),
    pytest.param(["none"] * 13, "none", id="all-none"),
    pytest.param(["underline"] + ["none"] * 12, "none", id="stray-underline"),
    pytest.param(["underline"] * 2 + ["none"] * 3, "none", id="below-three-sample-minimum"),
    pytest.param(["underline"] * 3 + ["none"] * 9, "underline", id="exact-quarter"),
    pytest.param(["underline"] * 3 + ["none"] * 10, "none", id="quarter-rounded-up"),
    pytest.param(["underline"] * 4 + ["none"] * 9, "underline", id="rounded-up-quarter-met"),
    pytest.param(["underline"] * 4 + ["color"] * 4 + ["none"] * 5, "none", id="tied-looks"),
    pytest.param(["color"] * 4 + ["underline"] * 4 + ["none"] * 5, "none", id="reversed-tie"),
    pytest.param(["underline"] * 5 + ["color"] * 4 + ["none"] * 4, "underline",
                 id="clear-non-none-winner"),
])
def test_highlight_samples_requires_unique_sufficient_support(
    looks: list[str], expected: str,
) -> None:
    samples = [(15 + 24 * i, look) for i, look in enumerate(looks)]
    evidence = [i for i, look in samples if expected == "none" or look == expected]
    assert _highlight_samples(samples) == (expected, evidence, {})


@pytest.mark.parametrize("looks,expected", [
    (["underline", "underline"], "underline"),
    (["underline", "none"], None),
    (["none", "underline"], None),
    (["underline", "color"], None),
    (["none", "none"], "none"),
    (["underline"], "underline"),
    (["none"], "none"),
])
def test_highlight_samples_below_minimum_requires_unanimity(
    looks: list[str], expected: str | None,
) -> None:
    samples = [(15 + 24 * i, look) for i, look in enumerate(looks)]
    observation = _highlight_samples(samples)
    assert observation == (None if expected is None else (expected, [i for i, _ in samples], {}))


@pytest.mark.parametrize("looks,expected", [
    (["underline", "underline"], "underline"),
    (["underline", "none"], None),
    (["none", "none"], "none"),
])
def test_visual_short_caption_highlights(
    monkeypatch: pytest.MonkeyPatch, looks: list[str], expected: str | None,
) -> None:
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module("yap_studio.taste.learn")
    info = SimpleNamespace(fps=30, width=360, height=640, video_frames=50, duration_s=50 / 30)
    pictures = {}
    for index, look in zip((15, 39), looks, strict=True):
        image = np.full((640, 360, 3), 30, dtype=np.uint8)
        _glyphs(image, 508, (255, 244, 221))
        if look == "underline":
            image[526:528, 60:130] = (210, 255, 45)
        pictures[index] = image

    monkeypatch.setattr(module, "iter_rgb", lambda path, info, indices, **kwargs: (
        (index, pictures[index]) for index in indices
    ))
    monkeypatch.setattr(module, "find_panels", lambda *args, **kwargs: [])
    fields, _, missing = module._visual(Path("synthetic.mp4"), info)
    reasons = {row["field"]: row["reason"] for row in missing}
    if expected is None:
        assert "captions.highlight_style" not in fields
        assert reasons["captions.highlight_style"] == (
            "Too few caption samples to separate a highlight from unstressed phrases."
        )
        for field in ("captions.highlight_color", "captions.highlight_text_color"):
            assert field not in fields
            assert reasons[field] == (
                "Highlight style was not measured; cannot identify its colours."
            )
    else:
        assert fields["captions.highlight_style"] == (expected, [15, 39], {})
        assert "captions.highlight_style" not in reasons
        if expected == "underline":
            assert fields["captions.highlight_color"][0] == "#D2FF2D"


@pytest.mark.parametrize("style", ["none", "underline", "color", "moving_block"])
def test_caption_band_and_all_highlights(style: str) -> None:
    image = np.full((480, 270, 3), 30, dtype=np.uint8)
    text = _glyphs(image, 365, (255, 244, 221))
    _glyphs(image, 395, (255, 244, 221))
    accent = np.zeros(image.shape[:2], dtype=bool)
    if style == "underline":
        accent[383:385, 60:130] = True
    elif style == "moving_block":
        accent[360:386, 55:95] = True
    elif style == "color":
        accent = text & (np.indices(text.shape)[1] < 95)
    image[accent] = (210, 255, 45)
    if style == "moving_block":
        image[text & (np.indices(text.shape)[1] < 95)] = (17, 17, 17)
    assert classify_highlight(image, text, accent if accent.any() else None) == style
    sample = caption_band(image)
    assert sample is not None
    assert sample.highlight == style
    assert delta_e(sample.color, "#FFF4DD") <= COLOR_TEXT_TOLERANCE
    assert sample.top >= 360
    assert sample.pitch == pytest.approx(30, abs=1)
    if style != "none":
        assert delta_e(sample.highlight_color, "#D2FF2D") <= COLOR_HIGHLIGHT_TOLERANCE
    if style == "moving_block":
        assert delta_e(sample.highlight_text_color, "#111111") <= COLOR_HIGHLIGHT_TOLERANCE


def test_line_pitch_requires_two_lines() -> None:
    mask = np.zeros((100, 80), dtype=bool)
    mask[10:30, 10:70:3] = True
    assert line_pitch(mask) is None
    mask[50:70, 10:70:3] = True
    assert line_pitch(mask) == 40
    assert caption_band(np.zeros((480, 270, 3), dtype=np.uint8)) is None


def test_line_pitch_measures_three_rows_and_refuses_inconsistent_spacing() -> None:
    mask = np.zeros((160, 80), dtype=bool)
    for top in (10, 50, 90):
        mask[top:top + 20, 10:70:3] = True
    assert line_pitch(mask) == 40
    mask[90:110] = False
    mask[115:135, 10:70:3] = True
    assert line_pitch(mask) is None


def test_equal_height_rows_use_baselines_when_glyph_mass_profiles_differ() -> None:
    mask = np.zeros((100, 80), dtype=bool)
    mask[10:30, 10:14] = True
    mask[10:16, 10:70] = True
    mask[50:70, 10:14] = True
    mask[64:70, 10:70] = True
    assert line_pitch(mask) == 40


def test_moving_block_measures_light_active_ink() -> None:
    image = np.full((480, 270, 3), 30, dtype=np.uint8)
    text = _glyphs(image, 365, (255, 244, 221))
    _glyphs(image, 395, (255, 244, 221))
    image[360:386, 55:95] = (47, 59, 255)
    image[text & (np.indices(text.shape)[1] < 95)] = (255, 255, 255)
    sample = caption_band(image)
    assert sample is not None and sample.highlight == "moving_block"
    assert delta_e(sample.highlight_text_color, "#FFFFFF") < 1
    assert sample.pitch == pytest.approx(30, abs=1)


def test_heavy_glyph_holes_do_not_turn_a_rectangular_block_into_colored_text() -> None:
    accent = np.zeros((120, 160), dtype=bool)
    accent[20:100, 10:150] = True
    for left in range(15, 140, 20):
        accent[25:95, left:left + 15] = False
    image = np.zeros((*accent.shape, 3), dtype=np.uint8)
    assert accent.sum() / (80 * 140) < 0.55
    assert classify_highlight(image, ~accent, accent) == "moving_block"


def test_block_with_one_solid_edge_and_active_glyph_holes_is_moving_block() -> None:
    accent = np.zeros((120, 220), dtype=bool)
    accent[20:56, 20:200] = True
    accent[23:56, 25:190] = False
    accent[23:56, 25:190:3] = True
    accent[23:56, 190:200] = False
    image = np.zeros((*accent.shape, 3), dtype=np.uint8)
    assert classify_highlight(image, ~accent, accent) == "moving_block"


@pytest.mark.parametrize("fringe", ["detached_row", "sparse_rows", "shaded_columns"])
def test_block_with_an_encoding_fringe_is_moving_block(fringe: str) -> None:
    accent = np.zeros((130, 190), dtype=bool)
    accent[20:110, 20:170] = True
    text = np.zeros_like(accent)
    text[23:108, 24:166] = True
    text[23:108, 24:166:3] = False
    text[108:110, 80:102] = True
    accent[text] = False
    if fringe == "detached_row":
        accent[18, 20:140] = True
    elif fringe == "sparse_rows":
        accent[18:20, 20:42] = True
    else:
        accent[20:110, 168:170] = False
        accent[20:33, 168:172] = True
        accent[19, 20:29] = True
    ys, xs = np.nonzero(accent)
    rectangle = accent[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    height, width = rectangle.shape
    assert width < height * 3
    assert 0.25 < rectangle.mean() < 0.55
    assert 0.8 < rectangle[-2:].mean() < 0.9
    assert rectangle[:2].mean() < 0.65
    assert _fringe_allowance(rectangle) == BLOCK_FRINGE_PX
    if fringe == "shaded_columns":
        assert rectangle[:, -2:].mean() < 0.65
        assert _edge_fill(rectangle.T[::-1], BLOCK_FRINGE_PX - 1) < 0.65
        assert _edge_fill(rectangle.T[::-1], BLOCK_FRINGE_PX) > 0.65
    image = np.zeros((*accent.shape, 3), dtype=np.uint8)
    assert classify_highlight(image, text, accent) == "moving_block"


@pytest.mark.parametrize("glyphs", ["serif_stems", "bars", "closed_counters", "middle_bar"])
def test_fringe_scan_keeps_colored_glyphs_as_color(glyphs: str) -> None:
    if glyphs == "serif_stems":
        accent = np.zeros((24, 27), dtype=bool)
        for left in (2, 11, 21):
            accent[:, left:left + 4] = True
            accent[1:3, left - 2:left + 6] = True
            accent[21:23, left - 2:left + 6] = True
        accent[11:13, 6:11] = True
    elif glyphs == "bars":
        accent = np.zeros((80, 80), dtype=bool)
        accent[0, 68:80] = True
        accent[1:7] = True
        accent[73:79] = True
        accent[79, :12] = True
        for row in range(7, 73):
            left = round((72 - row) * 68 / 65)
            accent[row, left:left + 12] = True
    elif glyphs == "closed_counters":
        accent = np.zeros((60, 40), dtype=bool)
        accent[:, :5] = True
        for top in (1, 28, 55):
            accent[top:top + 4, :35] = True
        for center in (16, 43):
            for row in range(center - 12, center + 13):
                right = 35 + round(5 * (1 - ((row - center) / 12)**2))
                accent[row, right - 4:right] = True
    else:
        accent = np.zeros((40, 100), dtype=bool)
        accent[:, ::3] = True
        accent[17:23] = True
    ys, xs = np.nonzero(accent)
    rectangle = accent[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    assert 0.25 < rectangle.mean() < 0.55
    assert rectangle.mean(axis=1).max() > 0.85
    if glyphs == "closed_counters":
        fringe = _fringe_allowance(rectangle)
        assert fringe > 0
        assert min(_edge_fill(edge, fringe) for edge in (
            rectangle, rectangle[::-1],
        )) > 0.8
        assert _edge_fill(rectangle.T[::-1], fringe) < 0.65
    else:
        assert not interior_components(~rectangle).any()
    if glyphs == "serif_stems":
        assert min(_edge_fill(edge, BLOCK_FRINGE_PX) for edge in (
            rectangle, rectangle[::-1], rectangle.T, rectangle.T[::-1],
        )) > 0.65
    if glyphs != "middle_bar":
        assert min(_edge_fill(edge, BLOCK_FRINGE_PX) for edge in (
            rectangle, rectangle[::-1],
        )) > 0.8
    else:
        assert rectangle[:6].mean() < 0.5
        assert rectangle[-6:].mean() < 0.5
    image = np.zeros((*accent.shape, 3), dtype=np.uint8)
    assert classify_highlight(image, accent, accent) == "color"


@pytest.mark.parametrize("height,expected", [(29, 0), (48, 2), (90, 4)])
def test_fringe_allowance_stays_inside_block_padding(height: int, expected: int) -> None:
    rectangle = np.ones((height, 80), dtype=bool)
    rectangle[4:-4, 4:-4] = False
    assert _fringe_allowance(rectangle) == expected
    rectangle[height // 2, -4:] = False
    assert _fringe_allowance(rectangle) == 0


def test_edge_fill_handles_short_lines_and_bounds_fringe() -> None:
    lines = np.array([[True, True, False, False], [False, False, False, False]])
    for fringe in (0, BLOCK_FRINGE_PX):
        assert _edge_fill(lines[:1], fringe) == 0.5
        assert _edge_fill(lines, fringe) == 0.25
    detached = np.zeros((4, 10), dtype=bool)
    detached[0] = True
    detached[2:4] = True
    assert _edge_fill(detached, 0) == detached[:2].mean() == 0.5
    assert _edge_fill(detached, BLOCK_FRINGE_PX) == 1
    limit = np.zeros((BLOCK_FRINGE_PX + 3, 10), dtype=bool)
    limit[BLOCK_FRINGE_PX:BLOCK_FRINGE_PX + 2] = True
    assert _edge_fill(limit, BLOCK_FRINGE_PX) == 1
    assert _edge_fill(limit, BLOCK_FRINGE_PX - 1) == 0.5
    limit[:] = False
    limit[BLOCK_FRINGE_PX + 1:] = True
    assert _edge_fill(limit, BLOCK_FRINGE_PX) == 0.5


def test_final_measurement_limits_cover_presets_and_keep_original_contracts() -> None:
    limits = taste_finals.FINAL_MEASUREMENT_LIMITS
    assert set(limits) == set(SHIPPED_PRESETS)
    for preset in ("clean_minimal", "warm_editorial", "electric"):
        assert limits[preset] == {}
    for preset, fields in limits.items():
        assert all(reason.strip() for reason in fields.values())
        style = resolve(f"preset:{preset}").data
        if "hook.size_px" in fields:
            assert taste_finals.FINAL_TITLE_LINES[preset] == 1
        if "captions.highlight_text_color" in fields:
            assert style["captions"]["highlight_style"] == "none"
        if "captions.max_words_per_phrase" in fields:
            assert "captions.words_per_caption" in fields
            assert "captions.max_characters_per_phrase" in fields
    assert "captions.font_class" in limits["marker"]
    assert "hook.size_px" not in limits["one_word"]  # Three visible title lines.


@pytest.mark.parametrize("direction", [-1, 1], ids=["upward", "downward"])
@pytest.mark.parametrize("pitch", [30, 40, 67])
def test_caption_discovery_keeps_rare_second_line_in_either_direction(
    direction: int, pitch: int,
) -> None:
    from yap_studio.taste.measure_text import CaptionDiscovery

    discovery = CaptionDiscovery(640, 360)
    common = np.full((640, 360, 3), 30, dtype=np.uint8)
    _glyphs(common, 508, (255, 244, 221))
    for _ in range(7):
        discovery.add(common)
    wrapped = common.copy()
    _glyphs(wrapped, 508 + direction * pitch, (255, 244, 221))
    discovery.add(wrapped)
    band = discovery.band()
    assert band is not None
    sample = caption_band(wrapped, band)
    assert sample is not None
    assert sample.pitch == pytest.approx(pitch, abs=1)
    assert sample.top == min(508, 508 + direction * pitch)
    assert sample.bottom == max(508, 508 + direction * pitch) + 16


def _learn_caption_pictures(
    vertical_clip: Path, monkeypatch: pytest.MonkeyPatch, *,
    first_glyph_top: int, last_glyph_top: int | None = None, direction: int = -1,
) -> tuple:
    """Exercise discovery and geometry with recorded pixels and no renderer."""
    import importlib
    from dataclasses import replace

    module = importlib.import_module("yap_studio.taste.learn")
    info = replace(ffmpeg.probe(vertical_clip), width=360, height=640,
                   video_frames=240, duration_s=8)
    common_top = last_glyph_top if last_glyph_top is not None and direction < 0 else first_glyph_top
    common = np.full((640, 360, 3), 30, dtype=np.uint8)
    _glyphs(common, common_top, (255, 244, 221))
    wrapped = common.copy()
    if last_glyph_top is not None:
        _glyphs(wrapped, first_glyph_top, (255, 244, 221))
        _glyphs(wrapped, last_glyph_top, (255, 244, 221))

    def frames(path: Path, info: ffmpeg.MediaInfo, indices: list[int], **kwargs: object):
        for slot, index in enumerate(indices):
            yield index, wrapped if slot == 1 else common

    monkeypatch.setattr(module, "iter_rgb", frames)
    monkeypatch.setattr(module, "find_panels", lambda *args, **kwargs: [])
    return module._visual(vertical_clip, info)


@pytest.mark.parametrize("direction", [-1, 1], ids=["upward", "downward"])
def test_learn_bottom_wrapped_caption_uses_lowest_line_box(
    vertical_clip: Path, monkeypatch: pytest.MonkeyPatch, direction: int,
) -> None:
    fields, _, missing = _learn_caption_pictures(
        vertical_clip, monkeypatch, first_glyph_top=478, last_glyph_top=508, direction=direction,
    )
    assert fields["captions.size_px"][0] == 80
    assert fields["captions.position"][0] == "bottom"
    top, frames, detail = fields["captions.top_px"]
    assert top == 1503
    assert frames == fields["captions.size_px"][1]
    assert detail["first_line_box_top_px"] == pytest.approx(1413)
    assert detail["last_line_box_top_px"] == detail["line_box_top_px"] == pytest.approx(1503)
    assert detail["engine_band_height_px"] == 260
    assert detail["size_source"] == "measured"
    assert "last line" in detail["geometry"] and "upward" in detail["geometry"]
    assert not any(row["field"] in {"captions.size_px", "captions.top_px"} for row in missing)


def test_learn_bottom_single_line_uses_explicit_default_size_for_position(
    vertical_clip: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fields, _, missing = _learn_caption_pictures(
        vertical_clip, monkeypatch, first_glyph_top=508,
    )
    assert "captions.size_px" not in fields
    assert fields["captions.position"][0] == "bottom"
    top, frames, detail = fields["captions.top_px"]
    assert top == 1505
    assert frames == fields["captions.position"][1]
    assert detail["size_source"] == "default" and detail["size_used_px"] == 76
    assert detail["first_line_box_top_px"] == detail["last_line_box_top_px"]
    assert detail["engine_band_height_px"] == 251
    assert "No two-line caption" in detail["size_reason"]
    assert any(row["field"] == "captions.size_px" for row in missing)
    assert not any(row["field"] == "captions.top_px" for row in missing)


def test_learn_bottom_position_uses_last_lines_own_glyph_height(
    vertical_clip: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    from yap_studio.taste.measure_text import TextSample

    module = importlib.import_module("yap_studio.taste.learn")
    sample = TextSample(
        "#FFF4DD", "none", None, None, 478, 530, 30, 478, 16,
        {"band_count": 2, "height_margin": 4, "height_ratio_margin": 0.5,
         "pitch_search_margin": 6},
        last_line_top=508, last_glyph_height=22,
    )
    monkeypatch.setattr(module, "caption_band", lambda *args: sample)
    fields, _, _ = _learn_caption_pictures(
        vertical_clip, monkeypatch, first_glyph_top=478, last_glyph_top=508,
    )
    top, _, detail = fields["captions.top_px"]
    assert top == 1512
    assert detail["first_line_box_top_px"] == 1413
    assert detail["last_line_box_top_px"] == 1512


def test_learn_middle_wrapped_caption_preserves_center_conversion(
    vertical_clip: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fields, _, _ = _learn_caption_pictures(
        vertical_clip, monkeypatch, first_glyph_top=342, last_glyph_top=372, direction=1,
    )
    assert fields["captions.size_px"][0] == 80
    assert fields["captions.position"][0] == "middle"
    top, _, detail = fields["captions.top_px"]
    assert top == 1101
    assert detail["line_box_top_px"] == detail["first_line_box_top_px"] == pytest.approx(1005)
    assert detail["last_line_box_top_px"] == pytest.approx(1095)
    assert detail["engine_band_height_px"] == 260 and detail["padding_px"] == 34
    assert "center" in detail["geometry"]


def test_learn_middle_single_line_keeps_size_and_position_offset_unmeasured(
    vertical_clip: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fields, _, missing = _learn_caption_pictures(
        vertical_clip, monkeypatch, first_glyph_top=342,
    )
    assert fields["captions.position"][0] == "middle"
    assert "captions.size_px" not in fields and "captions.top_px" not in fields
    assert {"captions.size_px", "captions.top_px"} <= {row["field"] for row in missing}


def test_panel_fill_box_and_presence() -> None:
    image = np.full((640, 360, 3), 110, dtype=np.uint8)
    image[100:220, 20:340] = (240, 230, 210)
    for y in (130, 160, 190):
        image[y : y + 8, 45:220] = (30, 30, 30)
    image[205:211, 45:120] = (155, 70, 45)
    panels = find_panels(image)
    assert len(panels) == 1
    panel = panels[0]
    assert panel.box() == {"x": 20, "y": 100, "width": 320, "height": 120}
    assert panel.fill == (240, 230, 210)
    assert panel_present(image, panel, reference_width=360)
    assert not panel_present(np.full_like(image, 110), panel, reference_width=360)
    colors = panel_text(image[100:220, 20:340], np.asarray(panel.fill))
    assert colors["text_color"] == "#1E1E1E"
    assert colors["accent_color"] == "#9B462D"


@pytest.mark.parametrize("ink", [(14, 18, 48), (43, 15, 46)])
def test_panel_text_measures_colored_ink_without_confusing_it_with_accent(ink: tuple) -> None:
    image = np.full((180, 360, 3), 240, dtype=np.uint8)
    for top in (40, 80):
        _glyphs(image, top, ink)
    image[115:120, 60:150] = (47, 59, 255)
    text = panel_text(image, np.array([240, 240, 240]), require_glyphs=True)
    assert text["text_color"] == hex_color(ink)
    assert text["accent_color"] == "#2F3BFF"
    assert text["pitch"] == 40


def test_panel_text_preserves_a_two_pixel_title_underline() -> None:
    image = np.full((180, 360, 3), 240, dtype=np.uint8)
    _glyphs(image, 50, (14, 18, 48))
    image[80:82, 60:150] = (47, 59, 255)
    text = panel_text(image, np.array([240, 240, 240]), require_glyphs=True)
    assert text["text_color"] == "#0E1230"
    assert text["accent_color"] == "#2F3BFF"
    assert "pitch" not in text


def test_punch_in_search_rejects_cuts_and_finds_scale() -> None:
    y, x = np.indices((160, 90))
    image = 80 + 30 * np.sin(x / 10) + 60 * np.cos(y / 25)
    zoom = scaled_picture(image, 1.25)
    mask = np.ones(image.shape, dtype=bool)
    mask[100:120] = False
    result = punch_in(image, zoom, mask)
    assert result is not None and result["scale"] == pytest.approx(1.25, abs=ZOOM_SCALE_TOLERANCE)
    assert punch_in(image, image, mask) is None
    assert punch_in(image, 255 - image, mask) is None


def test_punch_in_ignores_ineligible_scale_with_lower_error() -> None:
    """A scale that masks all changing texture cannot hide an eligible punch-in."""
    y, x = np.indices((228, 128))
    image = np.full((228, 128), 80.0)
    image[190:] += (30 * np.sin(x / 10) + 60 * np.cos(y / 25))[190:]
    zoom = scaled_picture(image, 1.25)
    zoom[209:] += 0.1
    valid = np.ones(image.shape, dtype=bool)
    valid[160:190] = False

    shared = valid & (scaled_picture(valid.astype(float), 1.49) >= 1 - 1e-12)
    assert shared.sum() >= 30
    assert np.mean(np.abs(image[shared] - zoom[shared])) == 0
    assert np.mean(np.abs(scaled_picture(image, 1.49)[shared] - zoom[shared])) == 0

    result = punch_in(image, zoom, valid)
    assert result is not None
    assert result["scale"] == pytest.approx(1.25)
    assert result["error"] == pytest.approx(0.010112359550561478)
    assert result["unscaled_error"] >= 2


@pytest.mark.parametrize("wrapped", [False, True], ids=["single-line", "wrapped"])
@pytest.mark.parametrize("event", ["zoom", "cut", "captions"])
def test_picture_cuts_masks_scaled_electric_captions(
    monkeypatch: pytest.MonkeyPatch, wrapped: bool, event: str,
) -> None:
    from types import SimpleNamespace

    from yap_studio.stages.captions import CAPTION_LINE_HEIGHT
    from yap_studio.taste import measure_pacing

    captions = resolve("preset:electric").data["captions"]
    ratio = 128 / 1080
    y, x = np.indices((228, 128))
    image = 80 + 12 * np.sin(x / 10) + 25 * np.cos(y / 25)
    before = np.repeat(np.rint(image)[..., None], 3, axis=2).astype(np.uint8)
    # Bottom captions keep their last line at top_px; wrapping adds a line above it.
    last_top = round((captions["top_px"] + 20) * ratio)
    pitch = round(captions["size_px"] * CAPTION_LINE_HEIGHT * ratio)
    height = round(captions["size_px"] * 0.75 * ratio)
    rows = [last_top - pitch, last_top] if wrapped else [last_top]
    for row in rows:
        for column in range(17, 110, 8):
            color = (216, 255, 47) if column == 17 else (255, 255, 255)
            before[row : row + height, column : column + 3] = color
            before[row : row + 3, column : column + 6] = color
            before[row + 4 : row + 6, column : column + 5] = color
    following = (
        scaled_picture(image, 1.25) if event == "zoom"
        else 255 - image if event == "cut" else image
    )
    # The caption disappears at the cut, as it does at the acceptance fixture's pause.
    after = np.repeat(np.rint(following)[..., None], 3, axis=2).astype(np.uint8)
    pictures = [before, before, after, after]
    monkeypatch.setattr(measure_pacing, "iter_rgb", lambda *args, **kwargs: enumerate(pictures))
    exclusion = {
        "x": 0, "y": round(rows[0] / ratio) - 20, "width": 1080,
        "height": round((last_top + height - rows[0]) / ratio) + 40,
    }
    cuts, _, punches = measure_pacing.picture_cuts(
        Path("synthetic.mp4"), SimpleNamespace(video_frames=len(pictures)), [exclusion],
    )
    assert cuts == ([] if event == "captions" else [2])
    assert len(punches) == (1 if event == "zoom" else 0)
    if punches:
        assert punches[0]["frame"] == 2
        assert punches[0]["scale"] == pytest.approx(1.25, abs=ZOOM_SCALE_TOLERANCE)
        assert punches[0]["error"] < 1


def test_punch_in_requires_enough_shared_picture_pixels() -> None:
    y, x = np.indices((160, 90))
    image = 80 + 30 * np.sin(x / 10) + 60 * np.cos(y / 25)
    valid = np.zeros(image.shape, dtype=bool)
    valid[0, :35] = True
    # Scaling even 1.05 samples excluded rows below these destination pixels.
    assert punch_in(image, scaled_picture(image, 1.25), valid) is None


def test_pause_envelope_and_transcript_gaps() -> None:
    envelope = np.full(300, 0.1)
    envelope[:10] = envelope[-10:] = 0
    envelope[80:110] = 0.0003
    envelope[160:205] = 0.0003
    pauses = pause_envelope(envelope)
    assert pauses == [{"start_s": 0.8, "length_s": 0.3}, {"start_s": 1.6, "length_s": 0.45}]
    assert pause_envelope(np.zeros(100)) == []
    transcript = Transcript([Word("a", 0, 0.8), Word("b", 1.1, 1.6), Word("c", 2.05, 3)])
    measured = transcript_pacing(transcript, 3)
    assert measured["pacing.words_per_minute"] == 60
    assert [p["length_s"] for p in measured["pacing.word_gap_pauses"]] == pytest.approx([0.3, 0.45])


def test_loudness_parser_uses_final_summary() -> None:
    stderr = (
        "Summary:\nI: -70 LUFS\nSummary:\nI: -14.3 LUFS\nLRA: 2.1 LU\nTrue peak:\nPeak: -1.7 dBFS"
    )
    assert _loudness(stderr) == (-14.3, 2.1, -1.7)
    assert _loudness("Summary:\nI: -inf LUFS") == (None, None, None)


@pytest.mark.parametrize(
    "field,values,expected",
    [
        ("captions.text_color", ["#FFFFFF", "#FDFDFD"], "proposed"),
        ("captions.text_color", ["#FFFFFF", "#111111"], "varies"),
        ("captions.size_px", [76, 80], "proposed"),
        ("captions.size_px", [76, 82], "varies"),
        ("captions.position", ["bottom", "middle"], "varies"),
        ("captions.max_words_per_phrase", [6, 6], "proposed"),
        ("captions.max_words_per_phrase", [6, 7], "varies"),
        ("captions.max_characters_per_phrase", [21, 21], "proposed"),
        ("captions.max_characters_per_phrase", [21, 22], "varies"),
    ],
)
def test_aggregation_keeps_per_video_evidence(field: str, values: list, expected: str) -> None:
    rows = [{"video": i, "value": v, "frames": [2, 4], "detail": {}} for i, v in enumerate(values)]
    result = aggregate(field, rows, "synthetic")
    assert result["status"] == expected
    assert result["per_video"] == rows
    if expected == "varies":
        assert result["value"] is None


@pytest.fixture
def vertical_clip(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    ffmpeg.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=90x160:r=30:d=0.5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        timeout_s=30,
    )
    return path


def test_stream_frame_indices_and_output_validation(
    vertical_clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    info = ffmpeg.probe(vertical_clip)
    with closing(iter_rgb(vertical_clip, info, [1, 7, 9])) as pictures:
        assert [(index, image.shape) for index, image in pictures] == [
            (1, (160, 90, 3)),
            (7, (160, 90, 3)),
            (9, (160, 90, 3)),
        ]
    result = learn([vertical_clip], tmp_path / "out", asr="none")
    resolve_style_data(json.loads(result.style_path.read_text()))
    data = json.loads(result.measurements_path.read_text())
    assert data["videos"][0]["scale"] == 12
    assert data["videos"][0]["frames"] == 15
    assert not (tmp_path / "config").exists()
    assert add_proposals(result.proposals_path)
    assert load_taste().revision == 0 and not load_taste().rules
    for entry in (*data["fields"].values(), *data["measures"].values()):
        for row in entry["per_video"]:
            assert row["frames"] == sorted(set(row["frames"]))
            assert all(0 <= i < 15 for i in row["frames"])
    with pytest.raises(YapError) as refusal:
        learn([vertical_clip], tmp_path / "out", asr="none")
    assert refusal.value.suggested_action
    learn([vertical_clip], tmp_path / "out", asr="none", force=True)
    result.style_path.write_text('{"schema": "other/schema"}')
    original = result.proposals_path.read_bytes()
    with pytest.raises(YapError):
        learn([vertical_clip], tmp_path / "out", asr="none", force=True)
    assert result.proposals_path.read_bytes() == original


def test_refuses_config_symlink_and_input_overwrites(
    vertical_clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    with pytest.raises(YapError):
        learn([vertical_clip], tmp_path / "config" / "styles", asr="none")
    assert not (tmp_path / "config").exists()
    out = tmp_path / "out"
    out.mkdir()
    (out / "proposed-style.json").symlink_to(vertical_clip)
    original = vertical_clip.read_bytes()
    with pytest.raises(YapError):
        learn([vertical_clip], out, asr="none", force=True)
    assert vertical_clip.read_bytes() == original
    (out / "proposed-style.json").unlink()
    os.link(vertical_clip, out / "proposed-style.json")
    with pytest.raises(YapError):
        learn([vertical_clip], out, asr="none", force=True)
    assert vertical_clip.read_bytes() == original


def test_invalid_media_and_missing_inputs(tmp_path: Path) -> None:
    with pytest.raises(YapError):
        learn([], tmp_path / "out", asr="none")
    with pytest.raises(YapError):
        learn([tmp_path / "missing.mp4"], tmp_path / "out", asr="none")
    audio = tmp_path / "audio.wav"
    ffmpeg.run(["ffmpeg", "-f", "lavfi", "-i", "sine=duration=0.1", str(audio)], timeout_s=30)
    with pytest.raises(YapError):
        learn([audio], tmp_path / "out", asr="none")
    assert not (tmp_path / "out").exists()


def test_decoded_pauses_and_loudness_on_short_source(source: Path) -> None:
    pauses = pause_envelope(audio_envelope(source))
    assert len(pauses) == 5
    for actual, expected in zip(pauses, taste_finals.pauses_s(), strict=True):
        assert actual["start_s"] == pytest.approx(expected["start_s"], abs=PAUSE_TOLERANCE)
        assert actual["length_s"] == pytest.approx(expected["length_s"], abs=PAUSE_TOLERANCE)
    assert loudness(source)["loudness.integrated_lufs"] < 0


def _missing_fields_message(missing: set[str], rows: list[dict]) -> str:
    """Name every missing field and share repeated diagnostics in a compact log."""
    details: dict[str, str] = {}
    fields = {}
    for field in sorted(missing):
        evidence = []
        for row in rows:
            if row["field"] != field:
                continue
            encoded = json.dumps(row.get("detail", {}), sort_keys=True, separators=(",", ":"))
            details.setdefault(encoded, f"d{len(details) + 1}")
            evidence.append({"reason": row["reason"], "detail": details[encoded]})
        fields[field] = evidence or [{"reason": "No not-measured row was recorded."}]
    message = "Missing measured fields: " + json.dumps(fields, separators=(",", ":"))
    budget = max(0, 2850 - len(message)) // max(1, len(details))
    for encoded, name in details.items():
        message += f"\n{name}: {encoded[:budget]}"
    return message


def _assert_final(
    preset: str, final: Path, words: Path, truth: dict, out: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(out.parent / f"config-{preset}"))
    started = time.monotonic()
    result = learn([final], out, asr="none", transcripts=[words])
    elapsed = time.monotonic() - started
    data = result.measurements
    actual = {key: row["value"] for key, row in data["fields"].items()}
    style = truth["style"]
    expected = {
        f"{section}.{key}": value
        for section in ("captions", "hook", "graphics")
        for key, value in style[section].items()
        if f"{section}.{key}" in actual
    }
    limits = taste_finals.FINAL_MEASUREMENT_LIMITS[preset]
    # Preset-specific limits are reviewed fixture/design facts, not whatever
    # happened to be missing in this run. Everything else stays mandatory.
    required = {
        "captions.text_color",
        "captions.highlight_style",
        "captions.position",
        "captions.top_px",
        "captions.size_px",
        "captions.max_words_per_phrase",
        "hook.style",
        "hook.position",
        "hook.duration_s",
        "hook.background_color",
        "hook.text_color",
        "hook.accent_color",
        "hook.size_px",
        "graphics.paper_color",
        "graphics.ink_color",
        "graphics.accent_color",
        "graphics.card_width_px",
        "graphics.hold_s",
    }
    if style["captions"]["highlight_style"] != "none":
        required.add("captions.highlight_color")
    if style["captions"]["highlight_style"] == "moving_block":
        required.add("captions.highlight_text_color")
    required -= limits.keys()
    missing = required - actual.keys()
    assert not missing, _missing_fields_message(missing, data["not_measured"])
    required_measures = {
        "captions.words_per_caption", "captions.font_class", "captions.band",
        "graphics.corner_radius_px", "graphics.cards", "hook.box", "hook.window",
    } - limits.keys()
    missing_measures = required_measures - data["measures"].keys()
    assert not missing_measures, _missing_fields_message(missing_measures, data["not_measured"])
    comparisons = []
    for field in sorted(required):
        wanted, measured = expected[field], actual[field]
        if field.endswith("color"):
            limit = (
                COLOR_TEXT_TOLERANCE
                if field in ("captions.text_color", "graphics.paper_color")
                else COLOR_TITLE_TOLERANCE
                if field.startswith("hook.") and field != "hook.background_color"
                else COLOR_HIGHLIGHT_TOLERANCE
            )
            difference = delta_e(wanted, measured)
        elif field.endswith(("style", "position")):
            difference, limit = int(wanted != measured), 0
        else:
            limit = {
                "captions.top_px": CAPTION_TOP_TOLERANCE,
                "captions.size_px": CAPTION_SIZE_TOLERANCE,
                "captions.max_words_per_phrase": 0,
                "hook.size_px": TITLE_SIZE_TOLERANCE,
                "graphics.card_width_px": PANEL_WIDTH_TOLERANCE,
                "hook.duration_s": TITLE_DURATION_TOLERANCE,
                "graphics.hold_s": DURATION_TOLERANCE,
            }[field]
            difference = abs(wanted - measured)
        assert difference <= limit, (preset, field, wanted, measured, difference)
        comparisons.append(
            {"field": field, "expected": wanted, "measured": measured, "delta": difference}
        )
    measures = {key: row["value"] for key, row in data["measures"].items()}
    counts = [phrase["words"] for phrase in truth["phrases"]]
    if "captions.words_per_caption" in required_measures:
        words_per_caption = measures["captions.words_per_caption"]
        assert words_per_caption["counts"] == counts
        assert len(words_per_caption["windows"]) == len(counts)
        for window in words_per_caption["windows"]:
            assert 0 <= window["start_frame"] <= window["end_frame"] < truth["frames"]
        comparisons.append({"field": "captions.words_per_caption", "expected": counts,
                            "measured": words_per_caption["counts"], "delta": 0})
    if "captions.font_class" in required_measures:
        assert measures["captions.font_class"] == truth["font_class"]
        comparisons.append({"field": "captions.font_class", "expected": truth["font_class"],
                            "measured": measures["captions.font_class"], "delta": 0})
    radius_delta = abs(measures["graphics.corner_radius_px"] - truth["card"]["radius"])
    assert radius_delta <= CORNER_RADIUS_TOLERANCE
    comparisons.append({"field": "graphics.corner_radius_px", "expected": truth["card"]["radius"],
                        "measured": measures["graphics.corner_radius_px"], "delta": radius_delta})
    character_field = "captions.max_characters_per_phrase"
    character_limit = style["captions"]["max_characters_per_phrase"]
    if character_field in limits:
        pass  # Recorded explicitly with the other reviewed limits below.
    elif character_field in actual and data["fields"][character_field]["status"] == "proposed":
        assert actual[character_field] == character_limit
        comparisons.append({"field": character_field, "expected": character_limit,
                            "measured": actual[character_field], "delta": 0})
    else:
        character_rows = [row for row in data["not_measured"]
                          if row["field"] == character_field]
        assert character_rows, "Character limit must be proposed or explain its consistent range."
        for row in character_rows:
            lower, upper = row["detail"]["consistent_max_chars_range"]
            assert character_limit in row["detail"]["consistent_max_chars"]
            assert f"Consistent with {lower} to {upper} characters" in row["reason"]
        comparisons.append({"field": character_field, "expected": character_limit,
                            "measured": None, "delta": None,
                            "reason": character_rows[0]["reason"]})
    family, weight = truth["font_class"]["family"], truth["font_class"]["weight"]
    font_class_reason = f"Measured font class: {family}, {weight} weight."
    font_rows = [row for row in data["not_measured"] if row["field"] == "captions.font"]
    assert len(font_rows) == 1
    font_reason = font_rows[0]["reason"]
    if "captions.font_class" in required_measures:
        assert font_class_reason in font_reason
    else:
        # A refusal must survive into the style-page explanation, including marker.
        assert font_reason
        assert "Naming the exact typeface needs text recognition (OCR)." in font_reason
    comparisons.append({"field": "captions.font", "expected": style["captions"]["font"],
                        "measured": None, "delta": None, "reason": font_reason})
    cuts = measures["pacing.cut_frames"]
    assert len(cuts) == len(truth["cut_frames"])
    assert cuts == pytest.approx(truth["cut_frames"], abs=CUT_FRAME_TOLERANCE)
    assert measures["pacing.shot_lengths_s"] == pytest.approx([3, 3, 3, 3], abs=1 / truth["fps"])
    assert measures["pacing.median_shot_length_s"] == pytest.approx(3, abs=1 / truth["fps"])
    zoom = measures["pacing.zoom_punch_ins"]
    assert zoom["count"] == 1 and zoom["frames"] == truth["zoom_frames"]
    assert zoom["punches"][0]["scale"] == pytest.approx(
        truth["zoom_scale"], abs=ZOOM_SCALE_TOLERANCE
    )
    assert zoom["per_minute"] == pytest.approx(5)
    pauses = measures["pacing.pause_s"]
    assert pauses["count"] == 5 and pauses["median_s"] == pytest.approx(0.45, abs=PAUSE_TOLERANCE)
    for observed, wanted in zip(pauses["pauses"], truth["pauses"], strict=True):
        assert observed["start_s"] == pytest.approx(wanted["start_s"], abs=PAUSE_TOLERANCE)
        assert observed["length_s"] == pytest.approx(wanted["length_s"], abs=PAUSE_TOLERANCE)
    card = measures["graphics.cards"]
    assert len(card) == 1
    for key in ("x", "y", "width", "height"):
        assert card[0]["box"][key] == pytest.approx(truth["card"][key], abs=BOX_TOLERANCE)
    assert card[0]["start_frame"] == truth["card"]["start_frame"]
    assert card[0]["end_frame"] == truth["card"]["end_frame"]
    if "captions.band" in required_measures:
        band = measures["captions.band"]
        assert band["height_px"] > 0
        if "captions.size_px" in required:
            assert band["height_px"] > actual["captions.size_px"]
        assert band["bottom_px"] - band["top_px"] == band["height_px"]
    assert measures["hook.box"]["width"] > 600
    hook = style["hook"]
    title_lines = taste_finals.FINAL_TITLE_LINES.get(preset, 2)
    title_height = (
        title_lines * hook["size_px"] * 0.72 * 1.04 + 66
        if hook["style"] == "quiet_serif"
        else title_lines * hook["size_px"] * 1.04 + 102
        if hook["style"] == "paper_title"
        else title_lines * hook["size_px"] * 0.98 + 100
    )
    quiet_width = taste_finals.FINAL_SINGLE_LINE_TITLE_WIDTHS.get(preset, 780)
    expected_title = (
        {"x": (1080 - quiet_width) // 2, "y": 300, "width": quiet_width,
         "height": round(title_height)}
        if hook["style"] == "quiet_serif"
        else {"x": 90, "y": 300, "width": 900, "height": round(title_height)}
        if hook["style"] == "paper_title"
        else {
            "x": 0,
            "y": round(300 - 540 * np.tan(np.radians(2.5))),
            "width": 1080,
            "height": round(title_height + 1080 * np.tan(np.radians(2.5))),
        }
    )
    for key, value in expected_title.items():
        assert measures["hook.box"][key] == pytest.approx(value, abs=BOX_TOLERANCE)
    title_window = measures["hook.window"]
    assert 0 <= title_window["start_frame"] <= round(DURATION_TOLERANCE * truth["fps"])
    assert title_window["end_frame"] == pytest.approx(
        round(hook["duration_s"] * truth["fps"]) - 1,
        abs=round(TITLE_DURATION_TOLERANCE * truth["fps"]),
    )
    independent = ffmpeg.run(
        [
            "ffmpeg",
            "-i",
            str(final),
            "-map",
            "0:a:0",
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        timeout_s=60,
    )
    integrated, lra, peak = _loudness(independent.stderr)
    assert measures["loudness.integrated_lufs"] == pytest.approx(integrated, abs=LUFS_TOLERANCE)
    assert measures["loudness.true_peak_dbtp"] == pytest.approx(peak, abs=LUFS_TOLERANCE)
    assert measures["loudness.lra_lu"] == pytest.approx(lra, abs=LUFS_TOLERANCE)
    resolved = resolve_style_data(json.loads(result.style_path.read_text()))
    assert resolved["name"] == "Learned style (proposed)"
    proposed = json.loads(result.style_path.read_text())
    if "captions.max_words_per_phrase" in required:
        word_limit = "max_words_per_phrase"
        assert proposed["captions"][word_limit] == style["captions"][word_limit]
        sources = data["proposed_style"]["sources"]
        assert sources[f"captions.{word_limit}"]["source"] == "measured"
    node = shutil.which("node")
    assert node is not None, "Node is required for the style page reader check"
    page = Path(__file__).resolve().parents[1] / "docs" / "design" / "style-page"
    prototype_data = json.loads(result.measurements_path.read_text(encoding="utf-8"))
    prototype_data["proposed_style"]["style"].get("broll", {}).pop("generated_pictures", None)
    prototype_data["proposed_style"]["sources"].pop("broll.generated_pictures", None)
    measurements = out.parent / "prototype-measurements.json"
    measurements.write_text(json.dumps(prototype_data), encoding="utf-8")
    script = """
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const [page, measurements] = process.argv.slice(1);
const context = {window: {}};
vm.runInNewContext(fs.readFileSync(path.join(page, 'style-data.js'), 'utf8'), context);
const {readLearned} = require(path.join(page, 'learned.js'));
const result = readLearned(JSON.parse(fs.readFileSync(measurements, 'utf8')),
                           context.window.YAP_STYLE_DATA.schema);
console.log(JSON.stringify({measured: result.rows.measured.map(row => row.field),
                           font_reason: result.rows.base.find(
                             row => row.field === 'captions.font').reason}));
"""
    opened = subprocess.run(
        [node, "-e", script, str(page), str(measurements)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=True,
    )
    reviewed = json.loads(opened.stdout)
    if "captions.max_words_per_phrase" in required:
        assert "captions.max_words_per_phrase" in reviewed["measured"]
    assert reviewed["font_reason"] == font_reason
    assert not load_taste().rules
    assert len(add_proposals(result.proposals_path)) == 3
    assert not load_taste().rules and load_taste().revision == 0
    assert set(OCR_FIELDS) <= {row["field"] for row in data["not_measured"]}
    for entry in (*data["fields"].values(), *data["measures"].values()):
        for row in entry["per_video"]:
            assert row["frames"] == sorted(set(row["frames"]))
            assert row["frames"] and all(0 <= frame < truth["frames"] for frame in row["frames"])
    for field, reason in limits.items():
        comparisons.append({"field": field, "measured": None, "delta": None,
                            "reason": reason, "limited": True,
                            "observed": actual.get(field, measures.get(field)),
                            "not_measured": [row for row in data["not_measured"]
                                             if row["field"] == field]})
    report = {"preset": preset, "runtime_s": elapsed, "fields": comparisons, "measures": measures}
    (out / "acceptance-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


@pytest.mark.parametrize("preset", SHIPPED_PRESETS)
def test_dev_final_acceptance(preset: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = os.environ.get("YAP_STUDIO_TASTE_FINALS")
    if not directory:
        pytest.skip("Set YAP_STUDIO_TASTE_FINALS to pre-rendered finished videos.")
    folder = Path(directory) / preset
    truth = json.loads((folder / "truth.json").read_text())
    _assert_final(
        preset, folder / "final.mp4", folder / truth["words"], truth, tmp_path / preset, monkeypatch
    )


@pytest.mark.render
@pytest.mark.parametrize("preset", SHIPPED_PRESETS)
def test_render_final_acceptance(
    preset: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    truth = taste_finals.build_final(preset, tmp_path / "render")
    _assert_final(preset, truth.final, truth.words, truth.to_json(), tmp_path / preset, monkeypatch)


def test_multiple_videos_propose_only_agreement(
    vertical_clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    module = importlib.import_module("yap_studio.taste.learn")
    second = tmp_path / "second.mp4"
    second.write_bytes(vertical_clip.read_bytes())

    def visual(path: Path, info: ffmpeg.MediaInfo) -> tuple:
        values = {
            "captions.text_color": "#FFFFFF" if path == vertical_clip else "#FDFDFD",
            "captions.size_px": 76 if path == vertical_clip else 90,
            "captions.position": "bottom",
        }
        return {key: (value, [1, 7], {}) for key, value in values.items()}, {}, []

    monkeypatch.setattr(module, "_visual", visual)
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    result = learn([vertical_clip, second], tmp_path / "out", asr="none")
    data = result.measurements
    assert data["fields"]["captions.size_px"]["status"] == "varies"
    assert data["fields"]["captions.text_color"]["value"] == "#FEFEFE"
    assert data["fields"]["captions.position"]["status"] == "proposed"
    proposed = json.loads(result.style_path.read_text())
    assert (
        proposed["captions"]["size_px"]
        == resolve_style_data({"schema_version": 1})["captions"]["size_px"]
    )
    assert len(data["fields"]["captions.size_px"]["per_video"]) == 2
    body = json.loads(result.proposals_path.read_text())["proposals"][0]
    assert "Measured in 2 of 2 finished videos" in body["context"]
    assert len(body["sources"]) == 2
    assert not (tmp_path / "config").exists()


@pytest.mark.parametrize("case", ["agree", "vary", "partial", "missing", "no_captions"])
def test_learn_records_shape_measures_and_explains_exact_font(
    case: str, vertical_clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    module = importlib.import_module("yap_studio.taste.learn")
    second = tmp_path / "second.mp4"
    second.write_bytes(vertical_clip.read_bytes())
    monkeypatch.setattr(module, "_visual", lambda path, info: (
        {} if case == "no_captions" and path == second else {
            "captions.text_color": ("#FFF4DD", [2], {})
        }, {}, [],
    ))
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))

    def shapes(path, info, fields, measures):
        assert measures == {}
        observations = {"graphics.corner_radius_px": (24, [3], {"frames": [3]})}
        if case == "no_captions" and path == second:
            assert fields == {}
            return observations, []
        if case == "missing" or (case == "partial" and path == second):
            return observations, [{"field": "captions.font_class",
                                   "reason": "Too few caption glyph pixels."}]
        value = {"family": "sans" if case == "vary" and path == second else "serif",
                 "weight": "regular"}
        observations["captions.font_class"] = value, [2, 5], {"frames": [2, 5]}
        return observations, []

    monkeypatch.setattr(module.measure_shapes, "measure", shapes)
    result = learn([vertical_clip, second], tmp_path / "shapes", asr="none")
    data = result.measurements
    radius = data["measures"]["graphics.corner_radius_px"]
    assert radius["value"] == 24 and radius["tolerance"] == 3
    assert all(row["frames"] == [3] for row in radius["per_video"])
    assert "captions.font_class" not in data["fields"]
    assert "graphics.corner_radius_px" not in data["fields"]
    source = data["proposed_style"]["sources"]["captions.font"]
    assert source["source"] == "base" and "OCR" in source["reason"]
    font_rows = [row for row in data["not_measured"] if row["field"] == "captions.font"]
    assert len(font_rows) == 1 and font_rows[0]["reason"] == source["reason"]
    assert "graphics.corner_radius_px" not in {row["field"] for row in data["not_measured"]}
    if case == "missing":
        assert "captions.font_class" not in data["measures"]
        assert "Too few caption glyph pixels" in source["reason"]
    else:
        font = data["measures"]["captions.font_class"]
        assert font["status"] == {
            "agree": "proposed", "vary": "varies", "partial": "partial",
            "no_captions": "proposed",
        }[case]
        assert all(row["frames"] == [2, 5] for row in font["per_video"])
        if case in {"agree", "no_captions"}:
            assert source["reason"] == (
                "Measured font class: serif, regular weight. "
                "Naming the exact typeface needs text recognition (OCR)."
            )
            assert font["coverage"] == {
                "measured": 1 if case == "no_captions" else 2,
                "showing": 1 if case == "no_captions" else 2, "videos": 2,
            }
        elif case == "vary":
            assert "vary across videos" in source["reason"]
        else:
            assert "Too few caption glyph pixels" in source["reason"]
            assert font["coverage"] == {"measured": 1, "showing": 2, "videos": 2}


def test_learn_asr_uses_transcript_evidence(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    from yap_studio.transcript.backends import ScriptedBackend

    module = importlib.import_module("yap_studio.taste.learn")
    words = taste_finals.build_words(tmp_path / "words.json")
    transcript = Transcript.load(words)
    transcribed = []

    def transcribe(path: Path) -> Transcript:
        transcribed.append(path)
        return transcript

    monkeypatch.setattr(
        module, "resolve_backend", lambda name: ScriptedBackend(transcribe)
    )
    original_measure = module.measure_words.measure
    caption_transcripts = []

    def measure(*args, **kwargs):
        caption_transcripts.append(args[4])
        return original_measure(*args, **kwargs)

    monkeypatch.setattr(module.measure_words, "measure", measure)
    monkeypatch.setattr(module, "_visual", lambda path, info: ({}, {}, []))
    result = learn([source], tmp_path / "out", asr="auto")
    data = result.measurements["measures"]
    assert data["pacing.words_per_minute"]["value"] == pytest.approx(155)
    gaps = data["pacing.word_gap_pauses"]["value"]
    assert len(gaps) == 5
    assert data["pacing.words_per_minute"]["per_video"][0]["frames"] == [0, 359]
    assert data["pacing.words_per_minute"]["per_video"][0]["detail"]["engine"] == "scripted"
    assert "pacing.words_per_minute" not in {
        r["field"] for r in result.measurements["not_measured"]
    }
    assert transcribed == [source.resolve()]
    assert caption_transcripts == [transcript]


def test_learn_words_file_measures_pacing_and_caption_limits_with_asr_none(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    from yap_studio.taste import measure_words

    module = importlib.import_module("yap_studio.taste.learn")
    words_file = taste_finals.build_words(tmp_path / "words.json")
    transcript = Transcript.load(words_file)
    phrases = build_phrases(
        [CaptionWord(f"w{i}", w.text, w.start, w.end) for i, w in enumerate(transcript.words)],
        resolve("preset:warm_editorial").data["captions"],
    )
    counts = [len(phrase.words) for phrase in phrases]
    monkeypatch.setattr(module, "_visual", lambda *args: (
        {"captions.text_color": ("#FFF4DD", [5, 20, 70], {})},
        {"captions.band": ({"top_px": 1524, "bottom_px": 1572, "height_px": 48},
                           [5, 20, 70], {})}, [],
    ))
    monkeypatch.setattr(module, "resolve_backend", lambda name: pytest.fail("unneeded ASR"))
    picture = _word_window_picture()
    blank = np.full_like(picture, 45)

    def pictures(path: Path, info: ffmpeg.MediaInfo, indices, *, width: int):
        assert width == 360
        assert list(indices) == list(range(360))
        for index in indices:
            present = any(phrase.start + 0.06 <= index / 30 < phrase.end for phrase in phrases)
            yield index, picture if present else blank

    monkeypatch.setattr(measure_words, "iter_rgb", pictures)
    result = learn([source], tmp_path / "with-words", asr="none", transcripts=[words_file])
    measures = result.measurements["measures"]
    assert measures["pacing.words_per_minute"]["value"] == pytest.approx(155)
    assert len(measures["pacing.word_gap_pauses"]["value"]) == 5
    assert measures["pacing.words_per_minute"]["per_video"][0]["detail"]["engine"] == "words file"
    measured = measures["captions.words_per_caption"]["value"]
    assert measured["counts"] == counts
    field = result.measurements["fields"][LIMIT_FIELDS[0]]
    assert field["status"] == "proposed" and field["value"] == 5
    assert field["tolerance"] == 0
    assert field["per_video"][0]["frames"] == [
        frame for window in measured["windows"]
        for frame in range(window["start_frame"], window["end_frame"] + 1)
    ]
    assert json.loads(result.style_path.read_text())["captions"]["max_words_per_phrase"] == 5
    assert result.to_dict()["proposed_style"]["sources"][LIMIT_FIELDS[0]]["source"] == "measured"
    assert LIMIT_FIELDS[0] not in {row["field"] for row in result.measurements["not_measured"]}


def test_learn_words_files_validate_length_and_input_paths(
    vertical_clip: Path, tmp_path: Path
) -> None:
    for files in ([], [None, None]):
        with pytest.raises(YapError, match="one words file or None per video"):
            learn([vertical_clip], tmp_path / "out", asr="none", transcripts=files)
    with pytest.raises(YapError):
        learn([vertical_clip], tmp_path / "out", asr="none",
              transcripts=[tmp_path / "missing-words.json"])
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("words,transcripts,asr", [
    pytest.param([], None, "none", id="no-words"),
    pytest.param(["first.json", "second.json"], [Path("first.json"), Path("second.json")],
                 "none", id="words-files"),
    pytest.param(["a.json", "-"], [Path("a.json"), None], "auto", id="mixed-file-and-asr"),
    pytest.param(["-", "-"], [None, None], "auto", id="asr-for-each-video"),
])
def test_taste_learn_cli_forwards_words_in_video_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    words: list[str], transcripts: list[Path | None] | None, asr: str,
) -> None:
    import importlib
    from unittest.mock import Mock

    from yap_studio import cli
    from yap_studio.taste.learn import LearnResult

    module = importlib.import_module("yap_studio.taste.cli")
    result = LearnResult(
        tmp_path / "proposed-style.json", tmp_path / "taste-proposals.json",
        tmp_path / "measurements.json", {
            "videos": [], "fields": {}, "measures": {}, "not_measured": [],
            "proposed_style": {"base": {"name": "Schema defaults"}, "sources": {}},
        },
    )
    measure = Mock(return_value=result)
    monkeypatch.setattr(module, "learn", measure)
    args = ["taste", "learn", "first.mp4", "second.mp4", "--out", str(tmp_path),
            "--asr", asr, "--json"]
    for filename in words:
        args.extend(["--words", filename])
    assert cli.main(args) == 0
    assert capsys.readouterr().err == ""
    measure.assert_called_once_with(
        [Path("first.mp4"), Path("second.mp4")], tmp_path, asr=asr, force=False, base=None,
        transcripts=transcripts,
    )


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("words", [["first.json"], ["first.json", "second.json", "third.json"]])
def test_taste_learn_cli_rejects_words_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    json_output: bool, words: list[str],
) -> None:
    import importlib
    from unittest.mock import Mock

    from yap_studio import cli

    module = importlib.import_module("yap_studio.taste.cli")
    measure = Mock(wraps=module.learn)
    monkeypatch.setattr(module, "learn", measure)
    args = ["taste", "learn", "first.mp4", "second.mp4", "--out", str(tmp_path / "out")]
    for filename in words:
        args.extend(["--words", filename])
    if json_output:
        args.append("--json")
    assert cli.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "provide one words file or None per video, in video order" in captured.err
    if json_output:
        assert json.loads(captured.err)["error"]["code"] == "invalid_parameter"
    else:
        assert captured.err.startswith("error: Bad value for transcripts:")
    assert measure.call_args.kwargs["transcripts"] == [Path(filename) for filename in words]
    assert not (tmp_path / "out").exists()


def test_style_without_words_keeps_phrase_limits_unmeasured(
    vertical_clip: Path, tmp_path: Path
) -> None:
    from yap_studio.taste.learn import measure_style

    result = measure_style(vertical_clip, LIMIT_FIELDS)
    assert result == {field: {"value": None, "reason": NEEDS_WORDS} for field in LIMIT_FIELDS}
    learned = learn([vertical_clip], tmp_path / "out", asr="none")
    missing = {row["field"]: row["reason"] for row in learned.measurements["not_measured"]}
    assert all(missing[field] == NEEDS_WORDS for field in LIMIT_FIELDS)


def test_large_regular_and_irregular_frame_selections(tmp_path: Path) -> None:
    clip = tmp_path / "many-frames.mp4"
    ffmpeg.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=90x160:r=300:d=2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        timeout_s=30,
    )
    info = ffmpeg.probe(clip)
    for selected in (list(range(0, 600, 4)), [3 * i + i % 2 for i in range(150)]):
        with closing(iter_rgb(clip, info, selected, width=45)) as pictures:
            assert [index for index, _ in pictures] == selected


def test_caption_strokes_ignore_wall_skin_and_shirt() -> None:
    from yap_studio.taste.measure_text import CaptionDiscovery

    discovery = CaptionDiscovery(640, 360)
    pictures = []
    for shift in range(8):
        image = np.full((640, 360, 3), 35, dtype=np.uint8)
        image[295:430, :180] = (245, 245, 245)
        image[430:530, 40 + shift : 260 + shift] = (218, 166, 130)
        image[470:620, 270:] = (235, 232, 229)
        _glyphs(image, 555, (248, 234, 202))
        image[575:578, 60:135] = (176, 83, 50)
        discovery.add(image)
        pictures.append(image)
    band = discovery.band()
    assert band is not None
    for image in pictures:
        sample = caption_band(image, band)
        assert sample is not None
        assert sample.top == 555 and sample.bottom == 571
        assert sample.highlight == "underline"
        assert delta_e(sample.highlight_color, "#B05332") < 1
        assert sample.pitch is None


def test_translucent_panel_over_opposing_backgrounds() -> None:
    image = np.zeros((640, 360, 3), dtype=np.uint8)
    image[:, 180:] = 255
    fill = np.array([225, 210, 190])
    image[100:200, 30:330] = np.rint(fill * 0.94 + image[100:200, 30:330] * 0.06).astype(np.uint8)
    _glyphs(image, 130, (25, 25, 25))
    panels = find_panels(image)
    assert len(panels) == 1
    panel = panels[0]
    assert panel.box() == {"x": 30, "y": 100, "width": 300, "height": 100}
    reconstructed = (np.asarray(panel.fill) - 0.06 * 127.5) / 0.94
    assert delta_e(reconstructed, fill) < COLOR_HIGHLIGHT_TOLERANCE


def test_rotated_dark_panel_with_dark_picture_above() -> None:
    image = np.full((640, 360, 3), 120, dtype=np.uint8)
    for x in range(360):
        top = round(110 - 0.045 * (x - 180))
        image[top : top + 100, x] = (12, 16, 28)
        image[top + 94 : top + 100, x] = (200, 235, 50)
    _glyphs(image, 140, (245, 245, 245))
    panels = find_panels(image)
    assert len(panels) == 1 and panels[0].rotated
    assert delta_e(np.asarray(panels[0].fill), np.array([12, 16, 28])) < 1


def test_dark_full_width_band_survives_a_nearby_subject_palette_color() -> None:
    image = np.full((640, 360, 3), 120, dtype=np.uint8)
    image[180:600, 120:240] = 12
    for x in range(360):
        top = round(110 - 0.045 * (x - 180))
        image[top:top + 100, x] = (6, 4, 16)
        image[top + 94:top + 100, x] = (123, 90, 254)
    _glyphs(image, 140, (252, 247, 254))
    panels = find_panels(image)
    assert len(panels) == 1 and panels[0].rotated
    assert delta_e(np.asarray(panels[0].fill), np.array([6, 4, 16])) < 1
    assert panels[0].width == 360


def test_cut_confirmation_ignores_overlays_and_smooth_motion() -> None:
    from yap_studio.taste.measure_pacing import cut_evidence

    y, x = np.indices((80, 48))
    base = 100 + 35 * np.sin(x / 4) + 40 * np.cos(y / 6)
    frames = np.stack([base.copy() for _ in range(45)])
    frames[8:16, 5:20] = 240
    valid = np.ones(base.shape, dtype=bool)
    valid[5:20] = False
    assert not any(row["accepted"] for row in cut_evidence(frames, valid))
    for n in range(20):
        # A smooth gesture has several adjacent fast frames.
        displacement = 4 * np.exp(-(((n - 10) / 3) ** 2))
        frames[n] = 100 + 35 * np.sin((x - displacement) / 4) + 40 * np.cos(y / 6)
    frames[25:] = 170 + 30 * np.cos(x / 3) - 25 * np.sin(y / 8)
    rows = cut_evidence(frames, np.ones(base.shape, dtype=bool))
    assert [row["frame"] for row in rows if row["accepted"]] == [25]
    assert any(not row["accepted"] for row in rows)


def test_pause_minimum_applies_to_audio_and_words() -> None:
    from yap_studio.taste.measure_pacing import MIN_PAUSE_S

    assert MIN_PAUSE_S == 0.15
    envelope = np.full(150, 0.1)
    envelope[15:29] = envelope[50:65] = 0
    assert pause_envelope(envelope) == [{"start_s": 0.5, "length_s": 0.15}]
    transcript = Transcript([Word("a", 0, 0.1), Word("b", 0.24, 0.3), Word("c", 0.45, 1)])
    gaps = transcript_pacing(transcript, 1)["pacing.word_gap_pauses"]
    assert len(gaps) == 1 and gaps[0]["length_s"] == pytest.approx(MIN_PAUSE_S)


def test_evidence_events_endpoints_and_even_sampling(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    from yap_studio.taste.learn import compact_evidence

    module = importlib.import_module("yap_studio.taste.learn")
    monkeypatch.setattr(module, "_visual", lambda path, info: ({}, {}, []))
    result = learn([source], tmp_path / "event-output", asr="none")
    measures = result.measurements["measures"]
    cuts = measures["pacing.cut_frames"]["per_video"][0]
    assert cuts["frames"] == cuts["value"]
    assert measures["pacing.median_shot_length_s"]["per_video"][0]["frames"] == cuts["value"]
    pauses = measures["pacing.pause_s"]["per_video"][0]
    assert pauses["frames"] == [round(p["start_s"] * 30) for p in pauses["value"]["pauses"]]
    loud = measures["loudness.integrated_lufs"]["per_video"][0]
    assert loud["frames"] == loud["frame_range"] == [0, 359]
    proposals = json.loads(result.proposals_path.read_text())["proposals"]
    for body in proposals:
        for source_row in body["sources"]:
            assert source_row["frames"] == measures[source_row["measure"]]["per_video"][0]["frames"]
    detail = {}
    sampled = compact_evidence(list(range(15000)), detail)
    assert len(sampled) == 10000 and sampled[0] == 0 and sampled[-1] == 14999
    assert detail["evidence_sampling"]["event_frames"] == 15000


def test_partial_fields_keep_named_missing_rows_without_global_duplicates(
    vertical_clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    module = importlib.import_module("yap_studio.taste.learn")
    other = tmp_path / "other.mp4"
    other.write_bytes(vertical_clip.read_bytes())
    monkeypatch.setattr(
        module,
        "_visual",
        lambda path, info: (
            ({"captions.text_color": ("#F8EACA", [3], {})}, {}, [])
            if path == vertical_clip
            else ({}, {}, [{"field": "captions.text_color", "reason": "No caption."}] * 2)
        ),
    )
    result = learn([vertical_clip, other], tmp_path / "partial", asr="none")
    rows = [r for r in result.measurements["not_measured"] if r["field"] == "captions.text_color"]
    assert rows == [
        {"field": "captions.text_color", "reason": "No caption.", "video": 1, "name": "other.mp4"}
    ]
    assert result.measurements["fields"]["captions.text_color"]["status"] == "proposed"


def test_panel_window_matches_ink_when_background_keeps_the_fill() -> None:
    image = np.full((640, 360, 3), 80, dtype=np.uint8)
    image[100:200, 30:330] = (225, 210, 190)
    _glyphs(image, 130, (25, 25, 25))
    panel = find_panels(image)[0]
    reference = image[panel.y : panel.y + panel.height, panel.x : panel.x + panel.width].copy()
    assert panel_present(image, panel, reference_width=360, reference=reference)
    background = image.copy()
    background[100:200, 30:330] = (225, 210, 190)
    assert not panel_present(background, panel, reference_width=360, reference=reference)


def test_dark_custom_caption_ink_is_not_its_bright_background() -> None:
    image = np.full((640, 360, 3), 240, dtype=np.uint8)
    _glyphs(image, 550, (20, 20, 20))
    sample = caption_band(image)
    assert sample is not None
    assert sample.color == "#141414"
    assert sample.top == 550 and sample.bottom == 566
    assert sample.pitch is None


@pytest.mark.parametrize(
    ("positions", "offsets", "status"),
    [
        ([(0, "bottom"), (1, "middle")], [(1, 950)], "varies"),
        ([(0, "middle"), (1, "middle")], [(1, 950)], "proposed"),
        ([(0, "middle")], [(1, 950)], "varies"),
        ([], [(0, 950)], "varies"),
    ],
)
def test_caption_offset_requires_its_measured_position(
    positions: list[tuple[int, str]], offsets: list[tuple[int, int]], status: str
) -> None:
    fields = {
        "captions.top_px": aggregate(
            "captions.top_px", [{"video": i, "value": value} for i, value in offsets], "test"
        )
    }
    if positions:
        fields["captions.position"] = aggregate(
            "captions.position", [{"video": i, "value": value} for i, value in positions], "test"
        )
    apply_dependencies(fields)
    offset = fields["captions.top_px"]
    assert offset["status"] == status
    assert offset["per_video"] == [{"video": i, "value": value} for i, value in offsets]
    if status == "varies":
        assert offset["value"] is None
        assert "depends on a position" in offset["detail"]["reason"]


def test_partial_caption_offset_is_omitted_when_positions_vary(
    vertical_clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    module = importlib.import_module("yap_studio.taste.learn")
    second = tmp_path / "second.mp4"
    second.write_bytes(vertical_clip.read_bytes())
    monkeypatch.setattr(
        module,
        "_visual",
        lambda path, info: (
            ({"captions.position": ("bottom", [2], {})}, {}, [])
            if path == vertical_clip
            else (
                {"captions.position": ("middle", [2], {}), "captions.top_px": (950, [2], {})},
                {},
                [],
            )
        ),
    )
    result = learn([vertical_clip, second], tmp_path / "dependent", asr="none")
    assert result.measurements["fields"]["captions.top_px"]["status"] == "partial"
    assert result.measurements["fields"]["captions.top_px"]["reason"] == (
        "measured in 1 of 2 videos that show captions"
    )
    style = json.loads(result.style_path.read_text())
    assert result.measurements["proposed_style"]["sources"]["captions.top_px"]["source"] == "base"
    assert (
        style["captions"]["top_px"]
        == resolve_style_data({"schema_version": 1})["captions"]["top_px"]
    )


@pytest.mark.parametrize("power", [1, 2, 3])
def test_panel_fade_window_reaches_zero_opacity(power: int) -> None:
    from yap_studio.taste.measure_panels import fade_end

    samples = [(i, 0.03 + min(1, max(0, (36 - i) / 16)) ** power) for i in range(51)]
    visible_end = max(i for i, amplitude in samples if amplitude > 0.21)
    assert fade_end(samples, visible_end) == 35
    abrupt = [(i, 1.03 if i < 20 else 0.03) for i in range(40)]
    assert fade_end(abrupt, 19) == 19


def test_panel_edge_amplitude_retains_faint_ink() -> None:
    from yap_studio.taste.measure_panels import panel_edge_match

    background = np.full((640, 360, 3), 80, dtype=np.uint8)
    settled = background.copy()
    settled[100:200, 30:330] = (225, 210, 190)
    _glyphs(settled, 130, (25, 25, 25))
    panel = find_panels(settled)[0]
    reference = settled[panel.y : panel.y + panel.height, panel.x : panel.x + panel.width]
    faint = np.rint(background * 0.99 + settled * 0.01).astype(np.uint8)
    amplitude, _ = panel_edge_match(faint, panel, reference_width=360, reference=reference)
    assert amplitude == pytest.approx(0.01, abs=0.003)
    assert not panel_present(faint, panel, reference_width=360, reference=reference)


def test_seconds_are_rounded_in_style_measurements_and_summary(
    vertical_clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    module = importlib.import_module("yap_studio.taste.learn")
    monkeypatch.setattr(
        module,
        "_visual",
        lambda path, info: (
            {"hook.duration_s": (5 / 3, [2], {}), "graphics.hold_s": (7 / 3, [2], {})},
            {},
            [],
        ),
    )
    monkeypatch.setattr(
        module.measure_pacing,
        "measure",
        lambda *args, **kwargs: {
            "pacing.shot_lengths_s": ([1 / 3, 2 / 3], [2], {}),
            "pacing.median_shot_length_s": (0.16000000000000014, [2], {}),
            "pacing.pause_s": (
                {
                    "pauses": [{"start_s": 1 / 3, "length_s": 0.16000000000000014}],
                    "median_s": 0.16000000000000014,
                    "count": 1,
                },
                [2],
                {},
            ),
            "pacing.word_gap_pauses": (
                [{"start_s": 2 / 3, "length_s": 0.16000000000000014}],
                [2],
                {},
            ),
        },
    )
    result = learn([vertical_clip], tmp_path / "rounded", asr="none")
    style = json.loads(result.style_path.read_text())
    assert style["hook"]["duration_s"] == 1.667
    assert style["graphics"]["hold_s"] == 2.333
    measures = result.to_dict()["measures"]
    assert measures["pacing.shot_lengths_s"]["value"] == [0.333, 0.667]
    assert measures["pacing.median_shot_length_s"]["value"] == 0.16
    assert measures["pacing.pause_s"]["value"]["pauses"] == [{"start_s": 0.333, "length_s": 0.16}]
    assert measures["pacing.word_gap_pauses"]["value"] == [{"start_s": 0.667, "length_s": 0.16}]
    assert "333333333333" not in result.measurements_path.read_text()
    assert "0.16000000000000014" not in result.measurements_path.read_text()
    proposals = json.loads(result.proposals_path.read_text())["proposals"]
    assert all("1 finished video;" in body["context"] for body in proposals)


def test_no_card_keeps_ocr_and_other_unmeasured_reasons(
    vertical_clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    module = importlib.import_module("yap_studio.taste.learn")
    monkeypatch.setattr(
        module,
        "_visual",
        lambda *args: (
            {},
            {},
            [
                {"field": f"graphics.{key}", "reason": "No stable visible evidence."}
                for key in ("paper_color", "ink_color", "accent_color", "card_width_px", "hold_s")
            ],
        ),
    )
    result = learn([vertical_clip], tmp_path / "no-card", asr="none")
    rows = {row["field"]: row["reason"] for row in result.measurements["not_measured"]}
    assert "OCR" in rows["graphics.font"]
    assert rows["graphics.style"] == "Panel appearance does not identify a graphics style."
    assert rows["graphics.corner_radius_px"] == "No card was found."
    assert "graphics.corner_radius" not in rows
    for key in ("paper_color", "ink_color", "accent_color", "card_width_px", "hold_s"):
        assert rows[f"graphics.{key}"] == "No card was found."


def test_cut_detail_keeps_decision_scores_and_near_misses(
    vertical_clip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from yap_studio.taste import measure_pacing
    from yap_studio.taste.measure_pacing import CUT_JUMP_RATIO

    candidates = [
        {"frame": 2, "score": 2.4, "accepted": True},
        {"frame": 4, "score": 1.2, "accepted": False},
        {"frame": 6, "score": CUT_JUMP_RATIO / 3, "accepted": False},
    ]
    monkeypatch.setattr(measure_pacing, "picture_cuts", lambda *args: ([2], candidates, []))
    data = measure_pacing.measure(vertical_clip, ffmpeg.probe(vertical_clip), [])
    detail = data["pacing.cut_frames"][2]
    assert detail["detector"] == "adjacent-motion jump ratio outside overlay boxes"
    assert detail["candidates"] == candidates[:2]
    assert detail["weakest_accepted_score"] == 2.4
    assert detail["strongest_rejected_score"] == 1.2
    assert all("shared_detector" not in row for row in detail["candidates"])


def test_missing_field_log_names_each_field_and_shared_diagnostics() -> None:
    missing = {
        f"hook.{key}"
        for key in (
            "style",
            "position",
            "duration_s",
            "background_color",
            "text_color",
            "accent_color",
            "size_px",
        )
    }
    detail = {
        "discovery_frames_examined": 12,
        "frames_with_panel_candidate": 0,
        "rejections": {"fill": 10, "slope_or_residual": 2},
    }
    rows = [{"field": name, "reason": "No title was found.", "detail": detail} for name in missing]
    message = _missing_fields_message(missing, rows)
    assert len(message.encode()) < 3000
    assert all(name in message for name in missing)
    assert "No title was found." in message
    assert "slope_or_residual" in message
    assert message.count("discovery_frames_examined") == 1


def test_pitch_diagnostics_explain_band_count_and_height_ratio() -> None:
    mask = np.zeros((120, 90), dtype=bool)
    mask[12:32, 15:75:3] = True
    detail = {}
    assert line_pitch(mask, detail=detail) is None
    assert detail["reason"] == "band_count" and detail["band_count"] == 1
    mask[60:115, 15:75:3] = True
    assert line_pitch(mask, detail=detail) is None
    assert detail["reason"] == "height_ratio" and detail["height_ratio"] > 2


def test_unmeasured_size_and_title_include_evidence_counts(
    vertical_clip: Path, tmp_path: Path
) -> None:
    result = learn([vertical_clip], tmp_path / "diagnostics", asr="none")
    missing = {row["field"]: row for row in result.measurements["not_measured"]}
    for name in ("captions.size_px", "captions.top_px"):
        detail = missing[name]["detail"]
        assert detail["samples_examined"] > 0
        assert detail["samples_with_pitch"] == 0
    for key in (
        "style",
        "position",
        "duration_s",
        "background_color",
        "text_color",
        "accent_color",
        "size_px",
    ):
        detail = missing[f"hook.{key}"]["detail"]
        assert detail["discovery_frames_examined"] > 0
        assert detail["panels_with_text"] == 0
        assert detail["title_frames_with_pitch"] == 0
        assert "rejections" in detail


@pytest.mark.parametrize("soften", [False, True])
def test_dense_rotated_title_fill_excludes_glyph_holes(soften: bool) -> None:
    image = np.full((1280, 720, 3), 100, dtype=np.uint8)
    y, x = np.indices(image.shape[:2])
    top = 230 - np.rint(np.tan(np.radians(2.5)) * (x - 360))
    panel = (y >= top) & (y < top + 200)
    image[panel] = (18, 24, 30)
    glyphs = panel & (x >= 60) & (x < 660) & (x % 24 < 14)
    glyphs &= (y >= top + 36) & (y < top + 168)
    image[glyphs] = (240, 240, 240)
    if soften:
        padded = np.pad(image.astype(float), ((1, 1), (1, 1), (0, 0)), mode="edge")
        weights = np.array([1, 2, 1])
        image = np.rint(
            sum(
                padded[a : a + 1280, b : b + 720] * weights[a] * weights[b]
                for a in range(3)
                for b in range(3)
            )
            / 16
        ).astype(np.uint8)
    found = [
        p for p in find_panels(image) if np.linalg.norm(np.asarray(p.fill) - (18, 24, 30)) < 10
    ]
    assert len(found) == 1 and found[0].rotated
    assert found[0].margins["raw_fill_coverage"] < 0.72
    assert found[0].margins["fill_coverage"] > 0
    assert found[0].margins["excluded_ink_fraction"] > 0.2


def test_panel_text_rejects_a_broad_picture_colour_without_glyph_strokes() -> None:
    image = np.full((240, 360, 3), 180, dtype=np.uint8)
    image[20:220, 70:290] = (25, 25, 25)
    detail = {}
    assert panel_text(image, np.array([180, 180, 180]), detail=detail, require_glyphs=True) == {}
    assert detail["reason"] == "text_colour"
    image[20:220, 70:290] = 180
    image[100:112, 40:320] = (25, 25, 25)
    assert panel_text(image, np.array([180, 180, 180]), detail=detail, require_glyphs=True) == {}
    image[100:112, 40:320] = 180
    for row in (65, 115):
        _glyphs(image, row, (25, 25, 25))
    found = panel_text(image, np.array([180, 180, 180]), detail=detail, require_glyphs=True)
    assert found["text_color"] == "#191919"
    assert found["pitch"] == 50
    assert found["colour_margins"]["stroke_support"] > 0


def test_later_panel_refinement_cannot_replace_the_selected_opening_title(
    vertical_clip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib
    from dataclasses import replace

    from yap_studio.taste.measure_panels import Panel

    module = importlib.import_module("yap_studio.taste.learn")
    info = replace(ffmpeg.probe(vertical_clip), video_frames=75, duration_s=2.5)
    later_sample = round(float(info.fps) * 0.5) + round(float(info.fps) * 0.8)
    title = Panel(5, 25, 80, 14, (24, 32, 40))
    later = Panel(5, 9, 80, 28, (210, 205, 190))

    def frames(path: Path, info: ffmpeg.MediaInfo, indices: list[int], **kwargs: object):
        for index in indices:
            yield index, np.full((160, 90, 3), index, dtype=np.uint8)

    class Discovery:
        def __init__(self, *args: object) -> None:
            pass

        def add(self, image: np.ndarray) -> None:
            pass

        def band(self) -> None:
            return None

    monkeypatch.setattr(module, "iter_rgb", frames)
    monkeypatch.setattr(module, "CaptionDiscovery", Discovery)
    monkeypatch.setattr(
        module,
        "find_panels",
        lambda image, **kwargs: [title, later] if image[0, 0, 0] >= later_sample else [title],
    )
    monkeypatch.setattr(
        module,
        "panel_text",
        lambda image, fill, **kwargs: {
            "text_color": "#FFFFFF",
            "accent_color": "#AA4433",
            "fill": fill,
        },
    )
    monkeypatch.setattr(module, "_window", lambda *args: (1, 70, [1, 70]))
    fields, _, _ = module._visual(vertical_clip, info)
    assert fields["hook.position"][0] == "headroom"
    assert fields["hook.background_color"][0] == "#182028"


def test_panel_rejects_a_straight_scenery_edge_without_a_parallel_lower_edge() -> None:
    image = np.full((640, 360, 3), 110, dtype=np.uint8)
    y, x = np.indices(image.shape[:2])
    top = 110 - np.rint(np.tan(np.radians(2.5)) * (x - 180))
    bottom = np.full(x.shape, 280)
    image[(y >= top) & (y < bottom)] = (25, 30, 35)
    detail = {}
    assert find_panels(image, detail=detail) == []
    assert detail["rejections"]["slope_or_residual"] > 0
    image[(y >= top) & (y < bottom)] = 110
    image[(y >= top) & (y < top + 170)] = (25, 30, 35)
    assert any(panel.rotated for panel in find_panels(image))


def test_panel_interior_keeps_ink_and_removes_border_connected_picture() -> None:
    from yap_studio.taste.frames import interior_components

    mask = np.zeros((80, 90), dtype=bool)
    mask[:8, :14] = True
    mask[8:40, 4:6] = True
    mask[25:28, 4:24] = True
    mask[20:45, 40:70] = True
    retained = interior_components(mask)
    assert not retained[:, :24].any()
    assert retained[20:45, 40:70].all()
    assert retained.sum() == 25 * 30
    assert mask[:8, :14].all()


def test_long_overlays_keep_evidence_without_aborting_style_output(
    vertical_clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    module = importlib.import_module("yap_studio.taste.learn")
    monkeypatch.setattr(
        module,
        "_visual",
        lambda *args: (
            {"hook.duration_s": (9.4, [1, 5], {}), "graphics.hold_s": (12.0, [1, 7], {})},
            {},
            [],
        ),
    )
    result = learn([vertical_clip], tmp_path / "long", asr="none")
    for field, value in (("hook.duration_s", 9.4), ("graphics.hold_s", 12.0)):
        row = result.measurements["fields"][field]
        assert row["status"] == "outside_style" and row["value"] == value
        assert "above the style's 8 s limit" in row["reason"]
        assert row["per_video"][0]["frames"] in ([1, 5], [1, 7])
        assert result.measurements["proposed_style"]["sources"][field]["source"] == "base"
    style = json.loads(result.style_path.read_text())
    resolve_style_data(style)
    assert style["hook"]["duration_s"] <= 8 and style["graphics"]["hold_s"] <= 8
    assert all(
        p.is_file() for p in (result.style_path, result.proposals_path, result.measurements_path)
    )


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("captions.size_px", 1, "below"),
        ("hook.style", "unknown", "not an allowed choice"),
    ],
)
def test_schema_leaf_rejections_are_explained(field: str, value: object, reason: str) -> None:
    import importlib

    module = importlib.import_module("yap_studio.taste.learn")
    assert reason in module._outside_style(field, value)


def test_invalid_position_cannot_apply_its_dependent_offset() -> None:
    import importlib

    module = importlib.import_module("yap_studio.taste.learn")
    fields = {
        field: aggregate(field, [{"video": 0, "value": value, "frames": [2], "detail": {}}], "test")
        for field, value in (("captions.position", "top"), ("captions.top_px", 1400))
    }
    base = resolve_style_data({"schema_version": 1})
    proposal = module._proposed_style(
        fields, base, {"kind": "schema_defaults", "name": "Schema defaults"}, []
    )
    assert fields["captions.position"]["status"] == "outside_style"
    assert fields["captions.top_px"]["status"] == "varies"
    assert proposal["style"]["captions"]["top_px"] == base["captions"]["top_px"]


@pytest.mark.parametrize(
    "measured,showing,total,status",
    [
        (1, 2, 2, "partial"),
        (2, 2, 2, "proposed"),
        (1, 1, 1, "proposed"),
        (1, 1, 2, "proposed"),
    ],
)
def test_feature_coverage_controls_style_proposals(
    measured: int, showing: int, total: int, status: str
) -> None:
    rows = [{"video": i, "value": 80, "frames": [2], "detail": {}} for i in range(measured)]
    result = aggregate("captions.size_px", rows, "pitch", showing=showing, total=total)
    assert result["coverage"] == {"measured": measured, "showing": showing, "videos": total}
    assert result["status"] == status
    if status == "partial":
        assert result["reason"] == "measured in 1 of 2 videos that show captions"


@pytest.mark.parametrize("base", [None, "preset:electric", "file"])
def test_complete_style_uses_explicit_base_and_marks_each_leaf(
    vertical_clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base: str | None
) -> None:
    import importlib

    from yap_studio.style import add

    module = importlib.import_module("yap_studio.taste.learn")
    expected = (
        resolve("preset:electric").data if base else resolve_style_data({"schema_version": 1})
    )
    spec = base
    if base == "file":
        candidate = tmp_path / "base.json"
        candidate.write_text(json.dumps(expected))
        spec = str(candidate)
    monkeypatch.setattr(
        module, "_visual", lambda *args: ({"captions.text_color": ("#E8EDF2", [2, 4], {})}, {}, [])
    )
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    result = learn([vertical_clip], tmp_path / "based", asr="none", base=spec)
    proposal = result.measurements["proposed_style"]
    style = json.loads(result.style_path.read_text())
    assert style == proposal["style"] and "internal" not in style
    assert style["captions"]["font"] == expected["captions"]["font"]
    assert proposal["base"]["kind"] == (
        "schema_defaults" if base is None else "file" if base == "file" else "preset"
    )
    if base == "file":
        assert proposal["base"]["file_name"] == "base.json"
    leaves = {
        f"{section}.{key}"
        for section in ("captions", "hook", "chapters", "graphics", "broll", "music")
        for key in style[section]
    }
    assert set(proposal["sources"]) == leaves
    assert proposal["sources"]["captions.text_color"] == {
        "source": "measured",
        "status": "proposed",
        "reason": None,
    }
    for field, row in proposal["sources"].items():
        if field != "captions.text_color":
            assert row["source"] == "base" and row["reason"]
    resolve_style_data(style)
    assert add(result.style_path).name == "Learned style (proposed)"


def test_partial_measurement_uses_base_when_all_videos_show_captions(
    vertical_clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    module = importlib.import_module("yap_studio.taste.learn")
    second = tmp_path / "other.mp4"
    second.write_bytes(vertical_clip.read_bytes())
    band = {"top_px": 1200, "bottom_px": 1400, "height_px": 200}
    monkeypatch.setattr(
        module,
        "_visual",
        lambda path, info: (
            {"captions.size_px": (84, [2], {})} if path == vertical_clip else {},
            {"captions.band": (band, [2], {})},
            []
            if path == vertical_clip
            else [{"field": "captions.size_px", "reason": "No two-line caption."}],
        ),
    )
    result = learn(
        [vertical_clip, second], tmp_path / "partial-feature", asr="none", base="preset:electric"
    )
    entry = result.measurements["fields"]["captions.size_px"]
    assert entry["status"] == "partial"
    assert entry["coverage"] == {"measured": 1, "showing": 2, "videos": 2}
    proposal = result.measurements["proposed_style"]
    assert proposal["sources"]["captions.size_px"]["reason"] == entry["reason"]
    assert (
        proposal["style"]["captions"]["size_px"]
        == resolve("preset:electric").data["captions"]["size_px"]
    )


@pytest.mark.parametrize("colors", [((255, 255, 255), (0, 0, 0)), ((255, 0, 0), (0, 76, 0))])
def test_flat_colour_cut_uses_rgb_even_when_luma_matches(colors: tuple) -> None:
    from yap_studio.taste.measure_pacing import cut_evidence

    pictures = np.array([colors[0]] * 5 + [colors[1]] * 5, dtype=np.float32)
    grey = pictures @ np.array([0.2126, 0.7152, 0.0722])
    frames = np.broadcast_to(grey[:, None, None], (10, 12, 8))
    candidates = cut_evidence(frames, np.ones((12, 8), dtype=bool), pictures)
    assert [row["frame"] for row in candidates if row["accepted"]] == [5]


def test_slow_flat_colour_fade_has_no_cut() -> None:
    from yap_studio.taste.measure_pacing import cut_evidence

    pictures = np.linspace((255, 0, 0), (0, 76, 0), 30, dtype=np.float32)
    grey = pictures @ np.array([0.2126, 0.7152, 0.0722])
    frames = np.broadcast_to(grey[:, None, None], (30, 12, 8))
    assert not cut_evidence(frames, np.ones((12, 8), dtype=bool), pictures)


def test_variable_frame_timestamps_measure_actual_shot_lengths(tmp_path: Path) -> None:
    from yap_studio.taste.frames import presentation_timing
    from yap_studio.taste.measure_pacing import measure

    video = tmp_path / "variable.mp4"
    ffmpeg.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=white:s=90x160:r=30:d=1",
            "-f",
            "lavfi",
            "-i",
            "color=black:s=90x160:r=30:d=1",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0,setpts='if(lt(N,30),N,30+(N-30)/30)/TB'[v]",
            "-map",
            "[v]",
            "-fps_mode",
            "vfr",
            "-enc_time_base",
            "1:30000",
            "-c:v",
            "libx264",
            "-bf",
            "0",
            "-video_track_timescale",
            "30000",
            str(video),
        ],
        timeout_s=30,
    )
    info = ffmpeg.probe(video)
    timing = presentation_timing(video, info)
    assert len(timing.starts) == 60 and timing.from_timestamps and timing.variable_rate
    assert timing.starts[30] == pytest.approx(30, abs=0.001)
    measured = measure(video, info, [], timing=timing)
    assert measured["pacing.cut_frames"][0] == [30]
    assert measured["pacing.shot_lengths_s"][0] == pytest.approx([30, 1], abs=0.001)
    assert timing.frame_at(20.4) == 20
    assert timing.frame_at(30.5) == 45
    result = learn([video], tmp_path / "timed-output", asr="none")
    assert result.measurements["videos"][0]["timing"] == {
        "from_timestamps": True,
        "variable_rate": True,
    }
    assert result.measurements["measures"]["pacing.shot_lengths_s"]["value"] == [30, 1]


def test_textured_cut_scores_do_not_use_rgb_fallback() -> None:
    from yap_studio.taste.measure_pacing import cut_evidence

    frames = np.random.default_rng(7).uniform(0, 255, (12, 24, 16)).astype(np.float32)
    valid = np.ones((24, 16), dtype=bool)
    arbitrary_colors = np.arange(12)[:, None] * np.array([[10, 4, 9]])
    assert cut_evidence(frames, valid, arbitrary_colors) == cut_evidence(frames, valid)


def test_constant_frame_timestamps_preserve_rate_based_values(vertical_clip: Path) -> None:
    from yap_studio.taste.frames import presentation_timing

    info = ffmpeg.probe(vertical_clip)
    timing = presentation_timing(vertical_clip, info)
    assert timing.from_timestamps and not timing.variable_rate
    assert timing.starts == pytest.approx([i / 30 for i in range(15)], abs=0.000001)
    assert timing.end_s == pytest.approx(0.5, abs=0.000001)
    assert timing.window_s(3, 8) == pytest.approx(6 / 30, abs=0.000001)


def test_variable_rate_flag_detects_small_changes_above_timestamp_precision(
    vertical_clip: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from yap_studio.taste.frames import presentation_timing

    info = replace(ffmpeg.probe(vertical_clip), video_frames=4)
    payload = {
        "frames": [
            {"pts_time": str(t), "duration_time": "0.033333"} for t in (0, 0.033333, 0.0665, 0.1)
        ],
        "streams": [{"time_base": "1/90000"}],
    }
    monkeypatch.setattr(
        ffmpeg, "run", lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(payload))
    )
    assert presentation_timing(vertical_clip, info).variable_rate


def test_overlay_and_audio_evidence_use_presentation_timestamps() -> None:
    import importlib

    from yap_studio.taste.frames import FrameTiming

    module = importlib.import_module("yap_studio.taste.learn")
    timing = FrameTiming((0, 1, 2, 2.1, 2.2, 2.3), 2.4, True, True)
    fields = {"hook.duration_s": (0.1, [0, 3], {}), "graphics.hold_s": (0.1, [2, 5], {})}
    measures = {
        "hook.window": ({"start_frame": 0, "end_frame": 3}, [0, 3], {}),
        "graphics.cards": ([{"start_frame": 2, "end_frame": 5}], [2, 5], {}),
    }
    module._timed_visual(fields, measures, timing)
    assert fields["hook.duration_s"][0] == pytest.approx(2.2)
    assert fields["graphics.hold_s"][0] == pytest.approx(0.4)
    assert timing.frame_at(2.19) == 4 and timing.frame_at(100) == 5
