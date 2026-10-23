"""Build deterministic finished vertical videos and their measurement ground truth.

Run with ``python tests/taste_finals.py OUT_DIR [PRESET ...]``. Rendering the
finals needs the pinned HyperFrames renderer; source and word helpers do not.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
import threading
import wave
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from yap_studio.captions import CaptionWord, build_phrases
from yap_studio.style import SHIPPED_PRESETS, ResolvedStyle, resolve
from yap_studio.transcript.model import Transcript, Word

SIZE = (1080, 1920)
FPS = 30
FRAMES = 360
SECONDS = FRAMES / FPS
RATE = 48_000
ZOOM_SCALE = 1.25
CARD_RADIUS = 24  # Canvas pixels at 1080 wide.
FONT_CLASSES = {
    "Inter": ("sans", "regular"),
    "Archivo Black": ("sans", "heavy"),
    # HyperFrames Editorial substitutes Gelasio for Georgia/Iowan Old Style.
    "Gelasio": ("serif", "regular"),
    "Instrument Serif": ("serif", "regular"),
    "Alata": ("sans", "regular"),
    # Monospace is not a measured family; Space Mono has low-contrast, regular stems.
    "Space Mono": ("sans", "regular"),
    # HyperFrames Bold substitutes Anton SC for Impact; Neutral keeps bundled Inter.
    "Anton SC": ("sans", "heavy"),
    "Titan One": ("sans", "heavy"),
    # Cursive is not a measured family; marker strokes are thick without serif contrast.
    "Permanent Marker": ("sans", "heavy"),
}
TITLE = "Opening words|that stay"
# Reviewed limits of pixel-only learning on this deterministic final. Keep the
# original three acceptance contracts intact. These are field-level exceptions,
# never preset skips; titles, cards, pacing and sound remain required separately.
FINAL_MEASUREMENT_LIMITS: dict[str, dict[str, str]] = {
    "clean_minimal": {},
    "warm_editorial": {},
    "electric": {},
    "editorial_serif": {
        "hook.size_px": "Instrument Serif fits the fixture title on one line; no line pitch.",
        "captions.font_class": "Instrument Serif's contrasting strokes straddle the family "
        "classifier's refusal band; naming the font requires OCR.",
        "captions.words_per_caption": "Italic active words change glyph extents within a phrase; "
        "pixel layout changes do not uniquely identify phrase boundaries.",
        "captions.max_words_per_phrase": "Depends on the unidentifiable italic caption partition.",
        "captions.max_characters_per_phrase": "Depends on the unidentifiable italic partition.",
    },
    "cobalt": {
        "hook.size_px": "Alata at 84 px fits the fixture title on one line; no line pitch.",
    },
    "mono": {
        "captions.size_px": "Space Mono at 62 px and a 26-character limit keeps captions on "
        "one line; no line pitch identifies the em size.",
        "captions.top_px": "Without measured size, the bottom line box uses a default-size "
        "estimate, not a measured anchor.",
        "captions.words_per_caption": "The sliding block leaves dark active glyphs over the "
        "dark subject during its travel; ink windows can split within a phrase.",
        "captions.max_words_per_phrase": "Depends on the unidentifiable sliding-block partition.",
        "captions.max_characters_per_phrase": "Depends on the sliding-block partition.",
    },
    "headline": {
        "captions.font_class": "Anton SC's condensed heavy glyphs straddle the stroke classifier's "
        "weight/family refusal bands; geometric stem ratios do not identify its class.",
        "captions.words_per_caption": "The sliding block leaves dark active glyphs over the "
        "dark subject during its travel; ink windows can split within a phrase.",
        "captions.max_words_per_phrase": "Depends on the unidentifiable sliding-block partition.",
        "captions.max_characters_per_phrase": "Depends on the sliding-block partition.",
    },
    "candy": {},
    "marker": {
        "captions.font_class": "Permanent Marker's irregular brush strokes straddle the "
        "weight classifier's refusal band.",
    },
    "one_word": {
        "captions.size_px": "One word per caption provides no repeated line pitch.",
        "captions.top_px": "Without measured size, the bottom line box uses a default-size "
        "estimate, not a measured anchor.",
        "captions.font_class": "One-line Archivo Black has no measured em size; the x-height "
        "fallback straddles the weight classifier's refusal band.",
        "captions.highlight_color": "Highlight is disabled; its configured color is invisible.",
        "captions.highlight_text_color": "Highlight is disabled; active ink is invisible.",
    },
    "hf_bold": {
        "hook.size_px": "Condensed Anton SC at 94 px fits the fixture title on one line.",
        "captions.font_class": "Anton SC's condensed heavy glyphs straddle the stroke classifier's "
        "weight/family refusal bands; geometric stem ratios do not identify its class.",
    },
    "hf_editorial": {
        "captions.text_color": "Dark ungrounded base ink blends with the fixture's dark subject; "
        "the visible teal stress color cannot identify the base ink.",
        "captions.highlight_style": "Only teal stress glyphs are visible over the dark subject; "
        "their color alone cannot identify editorial weight and italic emphasis.",
        "captions.highlight_color": "With base ink hidden, visible teal glyphs cannot be assigned "
        "a highlight role independently of the preset.",
        "captions.size_px": "Hidden base glyphs leave isolated stress words, not complete rows.",
        "captions.top_px": "Incomplete rows and default size cannot identify the line anchor.",
        "captions.words_per_caption": "Isolated stress glyphs do not expose whole phrase windows.",
        "captions.max_words_per_phrase": "Depends on the hidden full caption partition.",
        "captions.max_characters_per_phrase": "Depends on the hidden full caption partition.",
        "captions.font_class": "Visible bold stress glyphs cannot identify the base font weight.",
    },
    "hf_neutral": {
        "captions.text_color": "Dark ungrounded ink blends with the fixture's dark subject.",
        "captions.highlight_style": "No contrasting glyph band establishes caption presence.",
        "captions.position": "The invisible caption band cannot establish position.",
        "captions.top_px": "The invisible caption band cannot establish an anchor.",
        "captions.size_px": "The invisible glyph rows provide no line pitch.",
        "captions.max_words_per_phrase": "The invisible caption band provides no phrase windows.",
        "captions.max_characters_per_phrase": "The invisible band provides no caption partition.",
        "captions.words_per_caption": "The invisible caption band provides no phrase windows.",
        "captions.font_class": "The invisible caption band provides no glyph masks.",
        "captions.band": "Dark ungrounded ink has no contrasting band on the fixture's subject.",
        "captions.highlight_color": "Highlight is disabled; its configured color is invisible.",
        "captions.highlight_text_color": "Highlight is disabled; active ink is invisible.",
    },
}
# Independent expected layout of TITLE in the pinned bundled fonts. All other
# titles wrap to two lines; these short/condensed titles have a different layout.
FINAL_TITLE_LINES = {"editorial_serif": 1, "cobalt": 1, "hf_bold": 1, "one_word": 3}
FINAL_SINGLE_LINE_TITLE_WIDTHS = {"editorial_serif": 668, "cobalt": 746}
_SEED = 20_261_005
_WIDE_LINE = "MMMMMMMM WWWWWWWW."
# Integer centiseconds keep the run boundaries and pause lengths exact.
# Run 3's short words exercise the word limit; run 4 exercises the character limit.
_RUNS = (
    (0, 180, "Small steady steps make real progress today."),
    (210, 360, _WIDE_LINE),
    (405, 600, "Go up do it now go up do it now go up."),
    (660, 840, "Unstoppable momentum."),
    (870, 990, _WIDE_LINE),
    (1035, 1200, "We find our rhythm together today."),
)
_BACKGROUNDS = {
    "A": ("#66716D", "#ACB5AD"),
    "B": ("#8A7971", "#BBB0A5"),
    "C": ("#8A7971", "#BBB0A5"),
    "D": ("#6A7387", "#A5ACB9"),
}


@dataclass(frozen=True)
class FinalTruth:
    """Known events and resolved settings; card end_frame is inclusive."""

    preset: str
    final: Path
    words: Path
    frames: int
    fps: int
    cut_frames: tuple[int, ...]
    zoom_frames: tuple[int, ...]
    zoom_scale: float
    card: dict[str, int | str]
    phrases: list[dict[str, Any]]
    font_class: dict[str, str]
    pauses: list[dict[str, float]]
    title: dict[str, str]
    style: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        """Return independent JSON data with portable file names for paths."""
        data = asdict(self)
        data["final"] = self.final.name
        data["words"] = self.words.name
        data["cut_frames"] = list(self.cut_frames)
        data["zoom_frames"] = list(self.zoom_frames)
        return data


def hex_to_bt709(color: str) -> tuple[int, int, int]:
    """Convert #RRGGBB to rounded BT.709 limited-range 8-bit Y, Cb, Cr.

    RGB is the supplied display value, without an extra transfer-function step.
    Neutral black and white have Y values 16 and 235; chroma is centred on 128.
    """
    if len(color) != 7 or not color.startswith("#"):
        raise ValueError("expected a colour in #RRGGBB form")
    red, green, blue = (int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    cb = (blue - luma) / (2 * (1 - 0.0722))
    cr = (red - luma) / (2 * (1 - 0.2126))
    return round(16 + 219 * luma), round(128 + 224 * cb), round(128 + 224 * cr)


def pauses_s() -> list[dict[str, float]]:
    """Return the five internal pauses shared by the audio and transcript."""
    return [
        {"start_s": previous[1] / 100, "length_s": (following[0] - previous[1]) / 100}
        for previous, following in zip(_RUNS, _RUNS[1:], strict=False)
    ]


def build_words(path: Path) -> Path:
    """Save 31 touching words in six speech runs, including two wide phrases."""
    words = []
    for start, end, sentence in _RUNS:
        tokens = sentence.split()
        edges = [
            round((start + (end - start) * i / len(tokens)) / 100, 6)
            for i in range(len(tokens) + 1)
        ]
        words.extend(Word(text, edges[i], edges[i + 1]) for i, text in enumerate(tokens))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return Transcript(words, language="en", backend="synthetic", duration_s=SECONDS).save(path)


def speech_audio(seconds: float = SECONDS) -> np.ndarray:
    """Return deterministic stereo float audio with gated, modulated speech noise."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be positive and finite")
    count = round(seconds * RATE)
    if count < 1:
        raise ValueError("seconds must include at least one audio sample")
    rng = np.random.default_rng(_SEED)
    white = rng.standard_normal((count, 2))
    spectrum = np.fft.rfft(white, axis=0)
    frequencies = np.fft.rfftfreq(count, 1 / RATE)
    spectrum[(frequencies < 200) | (frequencies > 4000)] = 0
    band = np.fft.irfft(spectrum, n=count, axis=0)
    time = np.arange(count) / RATE
    speech = band * (0.7 + 0.3 * np.sin(2 * np.pi * 4 * time))[:, None]
    speech *= 0.55 / max(float(np.abs(speech).max()), 1e-12)
    audio = rng.standard_normal((count, 2)) * 10 ** (-70 / 20)
    for start, end, _ in _RUNS:
        first, last = start * RATE // 100, min(count, end * RATE // 100)
        if first < last:
            audio[first:last] += speech[first:last]
    return audio.astype(np.float32)


def _write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = np.rint(samples * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(RATE)
        output.writeframes(pcm.tobytes())


def _quarter_circle_area(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Integrate a radius-R quarter disk over the rectangle from (0, 0) to (x, y)."""
    split = np.minimum(x, np.sqrt(np.maximum(CARD_RADIUS**2 - y**2, 0)))

    def integral(edge: np.ndarray) -> np.ndarray:
        return (
            edge * np.sqrt(np.maximum(CARD_RADIUS**2 - edge**2, 0))
            + CARD_RADIUS**2 * np.arcsin(edge / CARD_RADIUS)
        ) / 2

    return y * split + integral(x) - integral(split)


class _Scene:
    """Evaluate a continuous picture on full-resolution and subsampled grids."""

    def __init__(self, graphics: Mapping[str, str], size: tuple[int, int]) -> None:
        width, height = size
        if width < 2 or height < 2 or width % 2 or height % 2:
            raise ValueError("yuv420p requires positive even dimensions")
        self.size = size
        self.colors = {
            name: hex_to_bt709(graphics[f"{name}_color"]) for name in ("paper", "ink", "accent")
        }
        self.backgrounds = {
            name: (hex_to_bt709(top), hex_to_bt709(bottom))
            for name, (top, bottom) in _BACKGROUNDS.items()
        }
        self.card_masks = {
            subsample: self._card_coverage(width // subsample, height // subsample)
            for subsample in (1, 2)
        }
        self.grids = {}
        for scale in (1.0, ZOOM_SCALE):
            for subsample in (1, 2):
                x = np.arange(width // subsample, dtype=np.float32) + 0.5
                y = np.arange(height // subsample, dtype=np.float32) + 0.5
                x = (x * SIZE[0] * subsample / width - SIZE[0] / 2) / scale + SIZE[0] / 2
                y = (y * SIZE[1] * subsample / height - SIZE[1] / 2) / scale + SIZE[1] / 2
                self.grids[scale, subsample] = (x[None, :], y[:, None])

    def _card_coverage(self, width: int, height: int) -> tuple[slice, slice, np.ndarray]:
        """Measure corner coverage in canvas units, keeping the other edges hard."""
        dx, dy = SIZE[0] / width, SIZE[1] / height
        left, right = math.floor(60 / dx), math.ceil(1020 / dx)
        top, bottom = math.floor(280 / dy), math.ceil(640 / dy)
        columns, rows = np.arange(left, right), np.arange(top, bottom)
        x0, x1 = columns * dx, (columns + 1) * dx
        y0, y1 = rows * dy, (rows + 1) * dy
        rectangle_area = (
            np.maximum(np.minimum(y1, 640) - np.maximum(y0, 280), 0)[:, None]
            * np.maximum(np.minimum(x1, 1020) - np.maximum(x0, 60), 0)[None, :]
        )
        missing_area = np.zeros(rectangle_area.shape)
        corners = np.zeros(rectangle_area.shape, dtype=bool)
        for cx, direction_x in ((60 + CARD_RADIUS, -1), (1020 - CARD_RADIUS, 1)):
            u0, u1 = (cx - x1, cx - x0) if direction_x < 0 else (x0 - cx, x1 - cx)
            u0, u1 = np.clip(u0, 0, CARD_RADIUS), np.clip(u1, 0, CARD_RADIUS)
            for cy, direction_y in ((280 + CARD_RADIUS, -1), (640 - CARD_RADIUS, 1)):
                v0, v1 = (cy - y1, cy - y0) if direction_y < 0 else (y0 - cy, y1 - cy)
                v0, v1 = np.clip(v0, 0, CARD_RADIUS), np.clip(v1, 0, CARD_RADIUS)
                indices = np.ix_(np.flatnonzero(v1 > v0), np.flatnonzero(u1 > u0))
                low_x, high_x = u0[indices[1]], u1[indices[1]]
                low_y, high_y = v0[indices[0]], v1[indices[0]]
                disk_area = (
                    _quarter_circle_area(high_x, high_y)
                    - _quarter_circle_area(low_x, high_y)
                    - _quarter_circle_area(high_x, low_y)
                    + _quarter_circle_area(low_x, low_y)
                )
                missing_area[indices] += (high_x - low_x) * (high_y - low_y) - disk_area
                corners[indices] = True
        coverage = (
            (rows[:, None] >= round(280 / dy))
            & (rows[:, None] < round(640 / dy))
            & (columns[None, :] >= round(60 / dx))
            & (columns[None, :] < round(1020 / dx))
        ).astype(float)
        coverage[corners] = (rectangle_area[corners] - missing_area[corners]) / (dx * dy)
        return slice(top, bottom), slice(left, right), np.clip(coverage, 0, 1)

    def _alpha(self, x: np.ndarray, y: np.ndarray, frame: int, pixel: float) -> np.ndarray:
        centre = 540 + 40 * math.sin(2 * math.pi * frame / 90)
        head = (np.sqrt(((x - centre) / 135) ** 2 + ((y - 830) / 185) ** 2) - 1) * 135
        body = (
            np.hypot(np.maximum(np.abs(x - centre) - 95, 0), np.maximum(np.abs(y - 1550) - 395, 0))
            - 155
        )
        return np.clip(0.5 - np.minimum(head, body) / pixel, 0, 1)

    def _card(self, plane: np.ndarray, component: int) -> None:
        height, width = plane.shape

        def rectangle(x: int, y: int, w: int, h: int, color: str) -> None:
            left, right = round(x * width / SIZE[0]), round((x + w) * width / SIZE[0])
            top, bottom = round(y * height / SIZE[1]), round((y + h) * height / SIZE[1])
            plane[top:bottom, left:right] = self.colors[color][component]

        rows, columns, coverage = self.card_masks[1 if component == 0 else 2]
        region = plane[rows, columns]
        region[:] = np.rint(
            region * (1 - coverage) + self.colors["paper"][component] * coverage
        ).astype(np.uint8)
        for y in (360, 430, 500):
            rectangle(120, y, 600, 28, "ink")
        rectangle(120, 570, 200, 28, "accent")

    def frame(self, frame: int, *, shot: str | None = None) -> bytes:
        """Return planar YUV; override shot with B to continue it past frame 179."""
        shot = shot or "ABCD"[min(frame // 90, 3)]
        scale = ZOOM_SCALE if shot == "C" else 1.0
        top, bottom = self.backgrounds[shot]
        rng = np.random.default_rng(_SEED + frame)
        planes = []
        for subsample, components in ((1, (0,)), (2, (1, 2))):
            x, y = self.grids[scale, subsample]
            shape = (y.shape[0], x.shape[1])
            alpha = self._alpha(x, y, frame, SIZE[0] * subsample / self.size[0] / scale)
            for component in components:
                background = top[component] + (bottom[component] - top[component]) * y / SIZE[1]
                silhouette = (28, 128, 128)[component]
                plane = np.broadcast_to(background, shape) * (1 - alpha) + silhouette * alpha
                if component == 0:
                    plane += 0.9 * rng.standard_normal(shape, dtype=np.float32)
                plane = np.clip(np.rint(plane), 0, 255).astype(np.uint8)
                if 195 <= frame <= 269:
                    self._card(plane, component)
                planes.append(plane.tobytes())
        return b"".join(planes)


def source_frame(
    frame: int, *, style: ResolvedStyle, size: tuple[int, int] = SIZE, shot: str | None = None
) -> bytes:
    """Evaluate one source frame, optionally continuing a chosen shot for comparison."""
    if frame < 0 or frame >= FRAMES or shot is not None and shot not in _BACKGROUNDS:
        raise ValueError("choose a frame from 0 through 359 and a shot from A through D")
    return _Scene(style.data["graphics"], size).frame(frame, shot=shot)


def _pipe_frames(command: list[str], frames: Iterable[bytes], *, timeout: float) -> None:
    """Stream frames while communicate drains stderr and enforces the timeout."""
    failures: list[Exception] = []
    with subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    ) as process:
        pipe = process.stdin
        assert pipe is not None
        # The writer owns stdin; communicate must not flush or close it concurrently.
        process.stdin = None

        def write() -> None:
            try:
                with pipe:
                    for frame in frames:
                        pipe.write(frame)
            except Exception as exc:
                failures.append(exc)

        writer = threading.Thread(target=write, daemon=True)
        writer.start()
        try:
            _, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.communicate(timeout=10)
            raise RuntimeError(f"ffmpeg timed out after {timeout} seconds") from exc
        finally:
            writer.join(timeout=10)
        if writer.is_alive():
            raise RuntimeError("source frame writer did not stop")
        if process.returncode:
            message = stderr.decode(errors="replace")
            raise RuntimeError(f"ffmpeg failed ({process.returncode}):\n{message}")
        if failures:
            raise RuntimeError("source frame writer failed") from failures[0]


def build_source(
    path: Path,
    *,
    style: ResolvedStyle,
    size: tuple[int, int] = SIZE,
    seconds: float = SECONDS,
) -> Path:
    """Pipe numpy frames into an H.264/PCM MOV, with shorter sizes for fast tests.

    A shorter duration truncates the same 30 fps timeline, preserving shot events.
    The default has 360 frames and exactly 576000 stereo audio samples.
    """
    if not math.isfinite(seconds) or not 0 < seconds <= SECONDS:
        raise ValueError("seconds must be positive and at most 12")
    count = round(seconds * FPS)
    if count < 1 or not math.isclose(count / FPS, seconds, abs_tol=1e-9):
        raise ValueError("seconds must describe a whole number of 30 fps frames")
    scene = _Scene(style.data["graphics"], size)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="taste-source-", dir=path.parent) as directory:
        audio = Path(directory) / "audio.wav"
        _write_wav(audio, speech_audio(count / FPS))
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "yuv420p",
            "-video_size",
            f"{size[0]}x{size[1]}",
            "-framerate",
            str(FPS),
            "-i",
            "pipe:0",
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-vf",
            "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
            "-frames:v",
            str(count),
            "-t",
            str(count / FPS),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "4",
            "-threads",
            "2",
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
            "-video_track_timescale",
            "30000",
            "-c:a",
            "pcm_s16le",
            "-ar",
            str(RATE),
            "-ac",
            "2",
            "-f",
            "mov",
            str(path),
        ]
        _pipe_frames(command, (scene.frame(i) for i in range(count)), timeout=120 + 20 * seconds)
    return path


def build_final(preset: str, out_dir: Path) -> FinalTruth:
    """Build a captioned final with the engine renderer and write its truth.json."""
    from yap_studio.stages.captions import caption_clip

    style = resolve(f"preset:{preset}")
    captions = style.data["captions"]
    try:
        family, weight = FONT_CLASSES[captions["font"]]
    except KeyError as exc:
        raise ValueError(
            f"unmapped caption font {captions['font']!r}; add it to FONT_CLASSES"
        ) from exc
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = build_source(out_dir / "source.mov", style=style)
    words = build_words(out_dir / "words.json")
    phrases = build_phrases(
        [
            CaptionWord(f"w{i}", word.text, word.start, word.end)
            for i, word in enumerate(Transcript.load(words).words)
        ],
        captions,
    )
    final = out_dir / "final.mp4"
    caption_clip(source, style=style, out=final, words=words, title=TITLE, srt=False, force=True)
    graphics = style.data["graphics"]
    truth = FinalTruth(
        preset=preset,
        final=final,
        words=words,
        frames=FRAMES,
        fps=FPS,
        cut_frames=(90, 180, 270),
        zoom_frames=(180,),
        zoom_scale=ZOOM_SCALE,
        card={
            "x": 60,
            "y": 280,
            "width": 960,
            "height": 360,
            "radius": CARD_RADIUS,
            "start_frame": 195,
            "end_frame": 269,
            "fill": graphics["paper_color"],
            "ink": graphics["ink_color"],
            "accent": graphics["accent_color"],
        },
        phrases=[
            {
                "words": len(phrase.words),
                "text": phrase.text,
                "start_s": phrase.start,
                "end_s": phrase.end,
            }
            for phrase in phrases
        ],
        font_class={"family": family, "weight": weight},
        pauses=pauses_s(),
        title=dict(zip(("plain", "key"), TITLE.split("|"), strict=True)),
        style=style.data,
    )
    (out_dir / "truth.json").write_text(
        json.dumps(truth.to_json(), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return truth


def main(argv: Sequence[str] | None = None) -> int:
    """Build requested presets, defaulting to all shipped presets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("presets", nargs="*", metavar="PRESET", help=", ".join(SHIPPED_PRESETS))
    args = parser.parse_args(argv)
    presets = args.presets or SHIPPED_PRESETS
    for preset in presets:
        if preset not in SHIPPED_PRESETS:
            parser.error(f"unknown preset {preset!r}; choose from {', '.join(SHIPPED_PRESETS)}")
    for preset in presets:
        truth = build_final(preset, args.out_dir / preset)
        print(f"{preset}: {truth.final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
