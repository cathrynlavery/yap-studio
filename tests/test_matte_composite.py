"""Synthetic ffmpeg composites checked at every decoded YUV pixel."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import synth
from test_matte_chroma import VISION_BACKEND, mint_screen_clip
from test_matte_make import PASSTHROUGH
from test_matte_timing import timestamp_plate

from yap_studio import ffmpeg, receipt
from yap_studio.errors import OutputPathError, ProcessingError, ValidationError
from yap_studio.faces.model import Box, VideoIdentity
from yap_studio.matte import composite
from yap_studio.matte.composite import (
    CARDS_SCHEMA,
    DEFAULT_PLACEMENT,
    PLACEMENTS,
    CardOverlay,
    composite_graph,
    composite_video,
)
from yap_studio.matte.make import make_matte as make_person_matte
from yap_studio.matte.model import Cleanup, MatteData

WIDTH, HEIGHT, FRAMES = 64, 96, 12
FPS = Fraction(10, 1)
PERSON = Box(24, 8, 16, 32)


def make_plate(path: Path, *, audio: bool = False, fps: Fraction = FPS) -> Path:
    """Encode a varied plate losslessly, optionally with a real audio stream."""
    command = [
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        f"testsrc2=size={WIDTH}x{HEIGHT}:rate={fps}",
    ]
    if audio:
        command.extend(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"])
    command.extend(["-map", "0:v:0", "-frames:v", str(FRAMES), "-c:v", "libx264", "-qp", "0"])
    if audio:
        command.extend(["-map", "1:a:0", "-c:a", "aac", "-t", str(float(FRAMES / fps))])
    command.extend(["-pix_fmt", "yuv420p", str(path)])
    ffmpeg.run(command, timeout_s=30)
    return path


def make_layer(
    path: Path,
    *,
    color: str = "red",
    frames: int = FRAMES,
    alpha: bool = True,
    first_color: str | None = None,
    fps: Fraction = FPS,
    offset: int = 0,
) -> Path:
    """Make a box-sized FFV1 layer, optionally with a distinct first frame."""
    fmt = "yuva420p" if alpha else "yuv420p"
    filters = f"color=c={color}:size=48x48:rate={fps},format={fmt}"
    if first_color:
        filters += f",drawbox=x=0:y=0:w=iw:h=ih:color={first_color}:t=fill:enable='eq(n,0)'"
    filters += f",setpts=PTS+{offset}/TB"
    ffmpeg.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", filters, "-map", "0:v:0",
         "-frames:v", str(frames), "-c:v", "ffv1", "-pix_fmt", fmt, str(path)],
        timeout_s=30,
    )
    return path


def make_matte(
    plate: Path, folder: Path, *, unavailable: bool = False, frames: int = FRAMES,
    empty_frames: tuple[int, ...] = (),
) -> MatteData:
    """Write a valid matte document and a rectangle of white alpha on black."""
    folder.mkdir()
    info = ffmpeg.probe(plate, count_frames=True)
    assert info.fps is not None
    identity = VideoIdentity(
        plate.name, receipt.sha256_file(plate), FRAMES, info.fps, WIDTH, HEIGHT
    )
    cleanup = Cleanup(median=False, low=0, high=255, feather_sigma=0)
    if unavailable:
        data = MatteData("unavailable", "no segmentation backend", identity, {}, cleanup,
                         None, None, None, None)
    else:
        raw = folder / "mask.gray"
        pixels = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        pixels[8:40, 24:40] = 255
        raw.write_bytes(b"".join(
            bytes(WIDTH * HEIGHT) if i in empty_frames else pixels.tobytes()
            for i in range(frames)
        ))
        matte = folder / "matte.mkv"
        ffmpeg.run(
            ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "gray",
             "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(info.fps), "-i", str(raw),
             "-map", "0:v:0", "-c:v", "ffv1", "-pix_fmt", "gray", str(matte)],
            timeout_s=30,
        )
        raw.unlink()
        data = MatteData(
            "measured", None, identity, {"name": "synthetic-test-matte"}, cleanup,
            {"file": "matte.mkv", "sha256": receipt.sha256_file(matte),
             "codec": "ffv1", "pix_fmt": "gray"},
            tuple(float(i / info.fps) for i in range(FRAMES)), float(FRAMES / info.fps),
            tuple(0.0 if i in empty_frames else float(PERSON.w * PERSON.h / (WIDTH * HEIGHT))
                  for i in range(FRAMES)),
            empty_frames=empty_frames,
        )
    (folder / "matte.json").write_text(json.dumps(data.to_dict()), encoding="utf-8")
    return data


def decode(path: Path, width: int = WIDTH, height: int = HEIGHT) -> tuple[np.ndarray, ...]:
    """Decode native YUV planes, avoiding display conversions from colour tags."""
    raw = ffmpeg.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "rawvideo",
         "-fps_mode", "passthrough", "-pix_fmt", "yuv420p", "pipe:1"],
        timeout_s=30, binary=True,
    ).stdout
    frame_size = width * height * 3 // 2
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, frame_size)
    pixels = width * height
    return (
        frames[:, :pixels].reshape(-1, height, width),
        frames[:, pixels:pixels * 5 // 4].reshape(-1, height // 2, width // 2),
        frames[:, pixels * 5 // 4:].reshape(-1, height // 2, width // 2),
    )


def assert_planes(actual: tuple[np.ndarray, ...], expected: tuple[np.ndarray, ...]) -> None:
    for plane, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(plane, reference)


def expected_card(
    expected: tuple[np.ndarray, ...],
    plate: tuple[np.ndarray, ...],
    card: CardOverlay,
    *,
    keyed: bool,
) -> None:
    """Place the decoded opaque layer, then restore the person in that box only."""
    layers = decode(card.layer, card.width, card.height)
    for index, (target, original, layer) in enumerate(zip(expected, plate, layers, strict=True)):
        scale = 1 if index == 0 else 2
        x, y, w, h = (getattr(card, key) // scale for key in ("x", "y", "width", "height"))
        for frame in range(card.start_frame, card.end_frame):
            target[frame, y:y + h, x:x + w] = layer[frame - card.start_frame]
            if keyed:
                px0, py0 = max(x, 24 // scale), max(y, 8 // scale)
                px1, py1 = min(x + w, 40 // scale), min(y + h, 40 // scale)
                target[frame, py0:py1, px0:px1] = original[frame, py0:py1, px0:px1]


@dataclass
class SyntheticMedia:
    plate: Path
    matte: Path
    layer: Path
    card: CardOverlay


@pytest.fixture
def synthetic(tmp_path: Path) -> SyntheticMedia:
    plate = make_plate(tmp_path / "plate.mp4")
    matte = tmp_path / "matte"
    make_matte(plate, matte)
    layer = make_layer(tmp_path / "red.mkv")
    card = CardOverlay("top", layer, 8, 0, 48, 48, 3, 9, "behind_head")
    return SyntheticMedia(plate, matte, layer, card)


@pytest.mark.parametrize("placement", ["behind_head", "front"])
def test_card_pixels_boxes_windows_and_color_tags(
    synthetic: SyntheticMedia, tmp_path: Path, placement: str
) -> None:
    card = replace(synthetic.card, placement=placement)
    before = synthetic.plate.read_bytes()
    result = composite_video(synthetic.plate, [card], tmp_path / "new" / "out.mp4",
                             matte_dir=synthetic.matte, lossless=True)
    plate = decode(synthetic.plate)
    expected = tuple(plane.copy() for plane in plate)
    expected_card(expected, plate, card, keyed=placement == "behind_head")
    assert_planes(decode(result.path), expected)
    assert synthetic.plate.read_bytes() == before
    assert result.frames == FRAMES and result.cards == (card,) and result.warnings == ()
    assert result.matte_used == (placement == "behind_head")
    assert result.to_dict() == {"path": str(result.path), "frames": FRAMES,
                                "cards": [card.to_dict()], "warnings": [],
                                "matte_used": result.matte_used}
    info = ffmpeg.probe(result.path)
    assert (info.color_space, info.color_primaries, info.color_transfer, info.color_range) == (
        "bt709", "bt709", "bt709", "tv"
    )


@pytest.mark.parametrize("backend", ["chroma-key", "vision-person-segmentation"])
def test_card_cutout_despill_respects_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    plate = mint_screen_clip(tmp_path)
    folder = tmp_path / "matte"
    data = make_person_matte(plate, folder, cleanup=PASSTHROUGH)
    assert data.backend["name"] == "chroma-key"
    if backend != "chroma-key":
        data = replace(data, backend=VISION_BACKEND.copy())
        (folder / "matte.json").write_text(json.dumps(data.to_dict()), encoding="utf-8")
    layer = synth.picture_source(tmp_path / "card.mov",
                                 "color=gray:s=1080x1920:r=30,format=rgba", 1, alpha=True)
    card = CardOverlay("screen", layer, 0, 0, 1080, 1920, 0, 1, "behind_head")
    out = tmp_path / "out.mp4"
    composite_video(plate, [card], out, matte_dir=folder, lossless=True)
    graph = composite.composite_graph

    def without_despill(*args, **kwargs):
        kwargs.pop("cutout", None)
        return graph(*args, **kwargs)

    monkeypatch.setattr(composite, "composite_graph", without_despill)
    plain = tmp_path / "plain.mp4"
    composite_video(plate, [card], plain, matte_dir=folder, lossless=True)
    if backend != "chroma-key":
        assert_planes(decode(out, 1080, 1920), decode(plain, 1080, 1920))
        return
    original = synth.picture_pixels(plate, 0).astype(int)
    before, after = (synth.picture_pixels(path, 0).astype(int) for path in (plain, out))
    np.testing.assert_allclose(after[900:1060, 460:620], original[900:1060, 460:620], atol=2)
    assert (before[900:1060, 760:800, 1] - after[900:1060, 760:800, 1]).max() > 3


def test_front_card_is_on_top_even_when_listed_first(
    synthetic: SyntheticMedia, tmp_path: Path
) -> None:
    blue = make_layer(tmp_path / "blue.mkv", color="blue")
    front = replace(synthetic.card, id="front", layer=blue, placement="front", y=16,
                    start_frame=5, end_frame=11)
    result = composite_video(synthetic.plate, [front, synthetic.card], tmp_path / "out.mp4",
                             matte_dir=synthetic.matte, lossless=True)
    plate = decode(synthetic.plate)
    expected = tuple(plane.copy() for plane in plate)
    expected_card(expected, plate, synthetic.card, keyed=True)
    expected_card(expected, plate, front, keyed=False)
    assert_planes(decode(result.path), expected)


def test_multiple_behind_cards_use_untouched_plate(
    synthetic: SyntheticMedia, tmp_path: Path
) -> None:
    blue = make_layer(tmp_path / "blue.mkv", color="blue")
    second = replace(synthetic.card, id="second", layer=blue, y=16, start_frame=5, end_frame=11)
    result = composite_video(synthetic.plate, [synthetic.card, second], tmp_path / "out.mp4",
                             matte_dir=synthetic.matte, lossless=True)
    plate = decode(synthetic.plate)
    expected = tuple(plane.copy() for plane in plate)
    for card in (synthetic.card, second):
        expected_card(expected, plate, card, keyed=True)
    assert_planes(decode(result.path), expected)


@pytest.mark.parametrize("unavailable", [False, True])
def test_missing_matte_renders_in_front_with_warning(
    synthetic: SyntheticMedia, tmp_path: Path, unavailable: bool
) -> None:
    folder = tmp_path / "unavailable"
    if unavailable:
        make_matte(synthetic.plate, folder, unavailable=True)
    result = composite_video(synthetic.plate, [synthetic.card], tmp_path / "out.mp4",
                             matte_dir=folder if unavailable else None, lossless=True)
    plate = decode(synthetic.plate)
    expected = tuple(plane.copy() for plane in plate)
    expected_card(expected, plate, synthetic.card, keyed=False)
    assert_planes(decode(result.path), expected)
    assert result.matte_used is False
    assert result.warnings == (
        "person cutout unavailable; card top renders in front of the person",
    )


@pytest.mark.parametrize("fps", [FPS, Fraction(30000, 1001)])
def test_layer_first_frame_lands_at_start_frame(tmp_path: Path, fps: Fraction) -> None:
    plate = make_plate(tmp_path / "plate.mp4", fps=fps)
    layer = make_layer(tmp_path / "layer.mkv", first_color="blue", fps=fps, offset=7)
    card = CardOverlay("later", layer, 8, 0, 48, 48, 3, 9)
    result = composite_video(plate, [card], tmp_path / "out.mp4", matte_dir=None, lossless=True)
    original = decode(plate)
    expected = tuple(plane.copy() for plane in original)
    expected_card(expected, original, card, keyed=False)
    assert_planes(decode(result.path), expected)


@pytest.mark.parametrize("fps", [Fraction(30, 1), Fraction(30000, 1001)])
def test_matte_frames_follow_plate_at_fractional_timestamps(
    tmp_path: Path, fps: Fraction
) -> None:
    plate = make_plate(tmp_path / "plate.mp4", fps=fps)
    layer = make_layer(tmp_path / "layer.mkv", fps=fps)
    matte = tmp_path / "matte"
    make_matte(plate, matte, empty_frames=tuple(range(1, FRAMES, 2)))
    card = CardOverlay("top", layer, 8, 0, 48, 48, 3, 9, "behind_head")
    result = composite_video(plate, [card], tmp_path / "out.mp4", matte_dir=matte, lossless=True)
    original = decode(plate)
    expected = tuple(plane.copy() for plane in original)
    expected_card(expected, original, card, keyed=False)
    for index, (target, pixels) in enumerate(zip(expected, original, strict=True)):
        scale = 1 if index == 0 else 2
        for frame in range(4, 9, 2):
            target[frame, 8 // scale:40 // scale, 24 // scale:40 // scale] = (
                pixels[frame, 8 // scale:40 // scale, 24 // scale:40 // scale]
            )
    assert_planes(decode(result.path), expected)


@pytest.mark.parametrize("has_card", [False, True])
def test_front_only_does_not_open_matte(
    synthetic: SyntheticMedia, tmp_path: Path, has_card: bool
) -> None:
    card = replace(synthetic.card, placement="front")
    result = composite_video(synthetic.plate, [card] if has_card else [], tmp_path / "out.mp4",
                             matte_dir=tmp_path / "missing-matte", lossless=True)
    original = decode(synthetic.plate)
    expected = tuple(plane.copy() for plane in original)
    if has_card:
        expected_card(expected, original, card, keyed=False)
    assert_planes(decode(result.path), expected)
    assert result.matte_used is False and result.warnings == ()


def test_audio_is_copied(tmp_path: Path) -> None:
    plate = make_plate(tmp_path / "audio.mp4", audio=True)
    result = composite_video(plate, [], tmp_path / "out.mp4", matte_dir=None)
    streams = []
    for source in (plate, result.path):
        streams.append(ffmpeg.run(
            ["ffmpeg", "-v", "error", "-i", str(source), "-map", "0:a:0", "-c:a", "copy",
             "-f", "adts", "pipe:1"], timeout_s=30, binary=True,
        ).stdout)
    assert streams[0] and streams[0] == streams[1]
    assert ffmpeg.probe(result.path).audio_codec == "aac"
    assert result.warnings == ()


@pytest.mark.parametrize("placement", ["front", "behind_head"])
@pytest.mark.parametrize("start", ["2.025", "2.075"])
@pytest.mark.parametrize("fps,ext", [(FPS, "mp4"), (Fraction(30000, 1001), "mkv")])
def test_positive_plate_timestamp_preserves_card_and_matte_frames(
    tmp_path: Path, placement: str, start: str, fps: Fraction, ext: str
) -> None:
    original = make_plate(tmp_path / "original.mp4", audio=True, fps=fps)
    plate = tmp_path / f"offset.{ext}"
    ffmpeg.run(["ffmpeg", "-v", "error", "-itsoffset", start, "-i", str(original),
                "-i", str(original), "-map", "0:v:0", "-map", "1:a:0", "-c", "copy",
                str(plate)], timeout_s=30)
    assert ffmpeg.probe(plate).video_start_s == pytest.approx(float(start))
    matte = tmp_path / "matte"
    make_matte(plate, matte, empty_frames=(4, 6, 8))
    layer = make_layer(tmp_path / "layer.mkv", first_color="blue", offset=7, fps=fps)
    card = CardOverlay("offset", layer, 8, 0, 48, 48, 3, 9, placement)
    result = composite_video(plate, [card], tmp_path / "out.mp4",
                             matte_dir=matte, lossless=True)
    pixels = decode(plate)
    expected = tuple(plane.copy() for plane in pixels)
    expected_card(expected, pixels, card, keyed=False)
    if placement == "behind_head":
        for index, (target, source) in enumerate(zip(expected, pixels, strict=True)):
            scale = 1 if index == 0 else 2
            for frame in (3, 5, 7):
                target[frame, 8 // scale:40 // scale, 24 // scale:40 // scale] = (
                    source[frame, 8 // scale:40 // scale, 24 // scale:40 // scale]
                )
    assert_planes(decode(result.path), expected)
    output_info = ffmpeg.probe(result.path)
    input_info = ffmpeg.probe(plate)
    assert output_info.video_start_s == pytest.approx(input_info.video_start_s, abs=0.001)
    assert output_info.audio_start_s == pytest.approx(input_info.audio_start_s, abs=0.001)


def test_60fps_layer_advances_one_frame_per_plate_frame(tmp_path: Path) -> None:
    plate = make_plate(tmp_path / "plate.mp4", fps=Fraction(30))
    colors = ("red", "green", "blue", "yellow", "magenta", "cyan")
    filters = "color=black:size=48x48:rate=60,format=yuva420p"
    for index, color in enumerate(colors):
        filters += f",drawbox=color={color}:t=fill:enable='eq(n,{index})'"
    layer = tmp_path / "numbered.mkv"
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", filters, "-frames:v", "6",
                "-c:v", "ffv1", str(layer)], timeout_s=30)
    assert ffmpeg.probe(layer).fps == Fraction(60)
    card = CardOverlay("numbered", layer, 8, 0, 48, 48, 3, 9)
    result = composite_video(plate, [card], tmp_path / "out.mp4", matte_dir=None, lossless=True)
    pixels = decode(plate)
    expected = tuple(plane.copy() for plane in pixels)
    expected_card(expected, pixels, card, keyed=False)
    assert_planes(decode(result.path), expected)


def test_composite_refuses_irregular_plate_with_matching_rates(tmp_path: Path) -> None:
    plate = timestamp_plate(tmp_path / "irregular.mkv")
    info = ffmpeg.probe(plate)
    assert info.fps == info.nominal_fps == Fraction(30)
    with pytest.raises(ValidationError, match=
                       "composite needs a constant-rate plate; run it on the cut"):
        composite_video(plate, [], tmp_path / "out.mp4", matte_dir=None)
    assert not (tmp_path / "out.mp4").exists()


def audio_plate(tmp_path: Path, codec: str) -> Path:
    video = make_plate(tmp_path / "silent.mp4")
    plate = tmp_path / "audio.mkv"
    ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "lavfi", "-i",
                "sine=frequency=440:sample_rate=48000", "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", codec, "-t", "1.2", str(plate)], timeout_s=30)
    return plate


@pytest.mark.parametrize("ext", ["mp4", "mov", "mkv"])
def test_pcm_audio_uses_container_policy(tmp_path: Path, ext: str) -> None:
    plate = audio_plate(tmp_path, "pcm_s16le")
    result = composite_video(plate, [], tmp_path / f"out.{ext}", matte_dir=None)
    assert ffmpeg.probe(result.path).audio_codec == ("pcm_s16le" if ext == "mkv" else "aac")
    assert result.warnings == (() if ext == "mkv" else (f"audio re-encoded to AAC for {ext}",))


@pytest.mark.parametrize("codec", ["aac", "alac", "mp3", "ac3", "eac3"])
@pytest.mark.parametrize("ext", ["mp4", "mov"])
def test_supported_audio_codecs_are_copied(tmp_path: Path, codec: str, ext: str) -> None:
    plate = audio_plate(tmp_path, codec)
    result = composite_video(plate, [], tmp_path / f"out.{ext}", matte_dir=None)
    assert ffmpeg.probe(result.path).audio_codec == codec
    assert result.warnings == ()
    hashes = []
    for source in (plate, result.path):
        packets = ffmpeg.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                              "-show_packets", "-show_data_hash", "sha256", "-of", "json",
                              str(source)], timeout_s=30)
        hashes.append([packet["data_hash"] for packet in json.loads(packets.stdout)["packets"]])
    assert hashes[0] and hashes[0] == hashes[1]


@pytest.mark.parametrize("ext", ["mp4", "mov"])
def test_opus_audio_obeys_muxer_support(tmp_path: Path, ext: str) -> None:
    plate = audio_plate(tmp_path, "libopus")
    supported = False
    if ext == "mp4":
        try:
            ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(plate), "-map", "0:a:0",
                        "-c:a", "copy", "-frames:a", "1", "-f", "mp4", "-movflags",
                        "empty_moov", "pipe:1"], timeout_s=30, binary=True)
            supported = True
        except ProcessingError:
            pass
    result = composite_video(plate, [], tmp_path / f"out.{ext}", matte_dir=None)
    assert ffmpeg.probe(result.path).audio_codec == ("opus" if supported else "aac")
    assert result.warnings == (() if supported else (f"audio re-encoded to AAC for {ext}",))


def test_unsupported_mp4_opus_is_reencoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plate = audio_plate(tmp_path, "libopus")
    run = ffmpeg.run

    def unsupported(command: list[str], **kwargs: Any):
        if "-f" in command and command[command.index("-f") + 1] == "mp4":
            raise ProcessingError("ffmpeg", 1, "muxer does not support Opus")
        return run(command, **kwargs)

    monkeypatch.setattr(ffmpeg, "run", unsupported)
    result = composite_video(plate, [], tmp_path / "out.mp4", matte_dir=None)
    assert ffmpeg.probe(result.path).audio_codec == "aac"
    assert result.warnings == ("audio re-encoded to AAC for mp4",)


@pytest.mark.parametrize("tag,value", [
    ("color_primaries", "bt2020"), ("color_space", "smpte170m"),
    ("color_transfer", "smpte2084"), ("color_range", "pc"),
])
def test_incompatible_plate_color_tags_are_refused(tmp_path: Path, tag: str, value: str) -> None:
    original = make_plate(tmp_path / "original.mp4")
    plate = tmp_path / "tagged.mkv"
    option = {"color_space": "colorspace", "color_transfer": "color_trc"}.get(tag, tag)
    ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(original), "-c:v", "copy",
                f"-{option}", value, str(plate)], timeout_s=30)
    assert getattr(ffmpeg.probe(plate), tag) == value
    with pytest.raises(ValidationError) as caught:
        composite_video(plate, [], tmp_path / "out.mp4", matte_dir=None)
    assert tag in str(caught.value) and value in str(caught.value)
    assert "convert to BT.709 limited range first; the cut already is" in str(caught.value)
    assert not (tmp_path / "out.mp4").exists()


@pytest.mark.parametrize("tagged", [False, True])
def test_untagged_and_bt709_limited_plates_pass(tmp_path: Path, tagged: bool) -> None:
    original = make_plate(tmp_path / "original.mp4")
    plate = original
    if tagged:
        plate = tmp_path / "tagged.mkv"
        ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(original), "-c:v", "copy",
                    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                    "-color_range", "tv", str(plate)], timeout_s=30)
    else:
        assert ffmpeg.probe(plate).color_primaries in (None, "unknown")
    result = composite_video(plate, [], tmp_path / "out.mp4", matte_dir=None, lossless=True)
    assert_planes(decode(result.path), decode(plate))


@pytest.mark.parametrize("field,value", [
    ("id", ""), ("x", 3), ("y", -2), ("width", 0), ("height", 7),
    ("start_frame", -1), ("start_frame", FRAMES), ("end_frame", 3),
    ("end_frame", FRAMES + 1), ("x", WIDTH), ("y", HEIGHT), ("placement", "back"),
])
def test_card_validation_paths(
    synthetic: SyntheticMedia, tmp_path: Path, field: str, value: Any
) -> None:
    card = replace(synthetic.card, **{field: value})
    if field == "start_frame" and value == FRAMES:
        card = replace(card, end_frame=FRAMES + 1)
    with pytest.raises(ValidationError) as caught:
        composite_video(synthetic.plate, [card], tmp_path / "out.mp4",
                        matte_dir=synthetic.matte, lossless=True)
    assert caught.value.parameter == f"cards[0].{field}"
    assert not (tmp_path / "out.mp4").exists()


@pytest.mark.parametrize("kind", ["missing", "no-alpha", "short", "dimensions"])
def test_invalid_layers(synthetic: SyntheticMedia, tmp_path: Path, kind: str) -> None:
    layer = tmp_path / "bad.mkv"
    if kind != "missing":
        make_layer(layer, frames=2 if kind == "short" else FRAMES, alpha=kind != "no-alpha")
    card = replace(synthetic.card, layer=layer, width=46 if kind == "dimensions" else 48)
    with pytest.raises(ValidationError) as caught:
        composite_video(synthetic.plate, [card], tmp_path / "out.mp4", matte_dir=None)
    assert caught.value.parameter == "cards[0].layer"


def test_duplicate_ids(synthetic: SyntheticMedia, tmp_path: Path) -> None:
    with pytest.raises(ValidationError) as caught:
        composite_video(synthetic.plate, [synthetic.card, synthetic.card], tmp_path / "out.mp4",
                        matte_dir=None)
    assert caught.value.parameter == "cards[1].id"


@pytest.mark.parametrize("kind", ["identity", "document-count", "file-count", "hash"])
def test_matte_refusals(synthetic: SyntheticMedia, tmp_path: Path, kind: str) -> None:
    document = synthetic.matte / "matte.json"
    data = json.loads(document.read_text())
    if kind == "identity":
        data["video"]["sha256"] = "0" * 64
    elif kind == "document-count":
        data["video"]["frames"] -= 1
        data["times"].pop()
        data["coverage"].pop()
    elif kind == "hash":
        data["matte"]["sha256"] = "0" * 64
    else:
        other = tmp_path / "short-matte"
        data = make_matte(synthetic.plate, other, frames=FRAMES - 1).to_dict()
        (synthetic.matte / "matte.mkv").write_bytes((other / "matte.mkv").read_bytes())
    document.write_text(json.dumps(data))
    with pytest.raises(ValidationError) as caught:
        composite_video(synthetic.plate, [synthetic.card], tmp_path / "out.mp4",
                        matte_dir=synthetic.matte)
    assert caught.value.parameter == {
        "identity": "matte.video.sha256", "document-count": "matte.video.frames",
        "file-count": "matte.matte.frames", "hash": "matte.matte.sha256",
    }[kind]
    if kind == "identity":
        assert "matte was made from a different video" in str(caught.value)
        assert caught.value.suggested_action == (
            f"yap-studio matte make {synthetic.plate} --out {synthetic.matte}"
        )
    assert not (tmp_path / "out.mp4").exists()


@pytest.mark.parametrize("target", ["unowned", "plate", "layer", "matte", "document"])
def test_output_refusals(synthetic: SyntheticMedia, tmp_path: Path, target: str) -> None:
    out = {
        "unowned": tmp_path / "keep.mp4", "plate": synthetic.plate, "layer": synthetic.layer,
        "matte": synthetic.matte / "matte.mkv", "document": synthetic.matte / "matte.json",
    }[target]
    if target == "unowned":
        out.write_bytes(b"unowned")
    before = out.read_bytes()
    with pytest.raises(OutputPathError):
        composite_video(synthetic.plate, [synthetic.card], out, matte_dir=synthetic.matte)
    assert out.read_bytes() == before
    assert not list(out.parent.glob(".yap_tmp_*"))


def test_failed_frame_count_does_not_publish(
    synthetic: SyntheticMedia, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = ffmpeg.probe

    def wrong_count(path: Path, *, count_frames: bool = True) -> ffmpeg.MediaInfo:
        info = probe(path, count_frames=count_frames)
        return replace(info, video_frames=FRAMES - 1) if path.name.startswith(".yap_tmp_") else info

    monkeypatch.setattr(ffmpeg, "probe", wrong_count)
    with pytest.raises(ProcessingError) as caught:
        composite_video(synthetic.plate, [], tmp_path / "out.mp4", matte_dir=None)
    assert caught.value.code == "composite_frame_count"
    assert not (tmp_path / "out.mp4").exists()
    assert not list(tmp_path.glob(".yap_tmp_*"))


def test_graph_deterministic_order_and_untouched_crops(tmp_path: Path) -> None:
    front = CardOverlay("front", tmp_path / "front.mkv", 0, 0, 48, 48, 0, 12)
    behind = replace(front, id="behind", x=8, start_frame=3, end_frame=9, placement="behind_head")
    second = replace(behind, id="second", y=16)
    cards = [front, behind, second]
    graph = composite_graph(cards, fps=Fraction(30000, 1001), matte_input=1, first_layer_input=2)
    assert graph == composite_graph(cards, fps=Fraction(30000, 1001),
                                    matte_input=1, first_layer_input=2)
    text, label, warnings = graph
    assert label == "out" and warnings == ()
    assert text.startswith(
        "[0:v]split=3[plate][person1][person2];"
        "[1:v]settb=AVTB,setpts=N/(30000/1001)/TB+(0/1)/TB,"
        "settb=expr=1001/30000,split=2[matte1][matte2];"
    )
    assert (
        "[3:v]settb=AVTB,setpts=(N+3)/(30000/1001)/TB+(0/1)/TB,"
        "settb=expr=1001/30000[layer1]"
    ) in text
    assert "[person1]crop=48:48:8:0[pixels1]" in text
    assert "[matte1]crop=48:48:8:0,format=gray[alpha1]" in text
    assert "[pixels1][alpha1]alphamerge=ts_sync_mode=nearest[key1]" in text
    assert "overlay=x=8:y=0:eof_action=pass:ts_sync_mode=nearest:enable='between(n,3,8)'" in text
    assert text.index("[layer1]overlay") < text.index("[key1]overlay")
    assert text.index("[key1]overlay") < text.index("[layer2]overlay")
    assert text.index("[key2]overlay") < text.index("[layer0]overlay")


def test_graph_passthrough_and_fallback(tmp_path: Path) -> None:
    assert composite_graph([], fps=FPS, matte_input=None, first_layer_input=1) == (
        "[0:v]null[out]", "out", ()
    )
    card = CardOverlay("one", tmp_path / "card.mkv", 0, 0, 48, 48, 0, 12, "behind_head")
    text, _, warnings = composite_graph([card], fps=FPS, matte_input=None, first_layer_input=1)
    assert "split=" not in text and "alphamerge" not in text
    assert warnings == ("person cutout unavailable; card one renders in front of the person",)
    assert composite_graph([replace(card, placement="front")], fps=FPS,
                           matte_input=None, first_layer_input=1)[0] == text


@pytest.mark.parametrize("field,value", [
    ("id", " "), ("layer", 7), ("layer", ""), ("x", True), ("y", 2.0),
    ("width", "48"), ("height", 0), ("start_frame", False), ("end_frame", None),
    ("placement", "behind"), ("extra", 1),
])
def test_strict_card_records(tmp_path: Path, field: str, value: Any) -> None:
    card = CardOverlay("one", tmp_path / "card.mkv", 0, 0, 48, 48, 0, 12)
    record = {**card.to_dict(), field: value}
    with pytest.raises(ValidationError) as caught:
        CardOverlay.from_dict(record, index=4)
    assert caught.value.parameter == f"cards[4].{field}"


def test_card_round_trip_default_and_relative_paths(tmp_path: Path) -> None:
    card = CardOverlay("one", Path("card.mkv"), 0, 0, 48, 48, 0, 12)
    assert CardOverlay.from_dict(card.to_dict()) == card
    assert card.box() == Box(0, 0, 48, 48)
    assert DEFAULT_PLACEMENT == "front" and PLACEMENTS == ("front", "behind_head")
    record = card.to_dict()
    del record["placement"]
    assert CardOverlay.from_dict(record).placement == "front"
    document = tmp_path / "cards.json"
    document.write_text(json.dumps({"schema": CARDS_SCHEMA, "cards": [record]}))
    assert composite.load_cards(document) == (replace(card, layer=tmp_path / "card.mkv"),)
