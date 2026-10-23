"""Calibration tests for video primitives and picture mapping."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import synth

from yap_studio import ffmpeg, video
from yap_studio.checks import structure
from yap_studio.errors import ValidationError


@pytest.mark.parametrize("rate,offset", [(Fraction(30), 0), (None, 0), (Fraction(30), 7)])
def test_rgb_frames_seeks_to_exact_output_frames(tmp_path, monkeypatch, rate, offset):
    raw = tmp_path / "unique.rgb"
    raw.write_bytes(np.random.default_rng(240).integers(
        0, 256, (75, 96, 64, 3), dtype=np.uint8).tobytes())
    source = tmp_path / "unique.mp4"
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", "64x96", "-framerate", "30", "-i", str(raw),
                "-c:v", "libx264", "-crf", "18", "-g", "15", "-bf", "2",
                "-sc_threshold", "0", "-pix_fmt", "yuv420p", "-output_ts_offset",
                str(offset), str(source)], timeout_s=30)

    def selected(frames):
        select = "+".join(f"eq(n,{frame})" for frame in frames)
        clock = f"fps={rate}:start_time=0:round=near," if rate is not None else ""
        command = ["ffmpeg", "-v", "error", "-i", str(source), "-an", "-vf",
                   f"setpts=PTS-STARTPTS,{clock}select='{select}',"
                   "scale=90:160:flags=area:in_color_matrix=bt709:in_range=tv,format=rgb24",
                   "-fps_mode", "passthrough", "-frames:v", str(len(frames)),
                   "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
        result = ffmpeg.run(command, timeout_s=30, binary=True)
        return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, 160, 90, 3)

    reference = selected(list(range(75)))
    assert np.unique(reference.reshape(75, -1), axis=0).shape[0] == 75
    frames = [0, 1, 2, 13, 14, 15, 16, 28, 29, 30, 31, 44, 45, 46, 59, 60, 61, 73, 74]
    expected = selected(frames)
    np.testing.assert_array_equal(expected, reference[frames])
    commands = []
    original = ffmpeg.run_stream

    def stream(command, **kwargs):
        commands.append(command)
        return original(command, **kwargs)

    monkeypatch.setattr(ffmpeg, "run_stream", stream)
    np.testing.assert_array_equal(video.rgb_frames(source, frames, rate=rate), expected)
    assert len(commands) == len(frames)
    for command, frame in zip(commands, frames, strict=True):
        assert command.index("-ss") < command.index("-i")
        assert float(command[command.index("-ss")+1]) == pytest.approx(frame/30, abs=1e-6)
        assert command[command.index("-frames:v")+1] == "1"


def _direct_first_frame(path: Path, *, w: int, h: int) -> np.ndarray:
    result = ffmpeg.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            f"scale={w}:{h}:flags=area,format=gray",
            "-frames:v",
            "1",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "-",
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
        binary=True,
    )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(h, w).astype(np.float32)


def _put_cover_before_tracks(path: Path) -> None:
    data = path.read_bytes()

    def boxes(start: int, end: int) -> list[tuple[int, int, bytes]]:
        found: list[tuple[int, int, bytes]] = []
        position = start
        while position < end:
            size = int.from_bytes(data[position : position + 4], "big")
            assert size >= 8
            found.append((position, size, data[position + 4 : position + 8]))
            position += size
        assert position == end
        return found

    moov_start, moov_size, _ = next(box for box in boxes(0, len(data)) if box[2] == b"moov")
    children = boxes(moov_start + 8, moov_start + moov_size)
    cover = next(box for box in children if box[2] == b"udta")
    first_track = next(index for index, box in enumerate(children) if box[2] == b"trak")
    children.remove(cover)
    children.insert(first_track, cover)
    content = b"".join(data[start : start + size] for start, size, _ in children)
    path.write_bytes(
        data[: moov_start + 8] + content + data[moov_start + moov_size :]
    )


def _leading_cover_video(path: Path) -> None:
    ffmpeg.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:s=16x16:r=1:d=1",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=32x18:r=10:d=1",
            "-map",
            "0:v:0",
            "-map",
            "1:v:0",
            "-c:v:0",
            "mjpeg",
            "-frames:v:0",
            "1",
            "-disposition:v:0",
            "attached_pic",
            "-c:v:1",
            "libx264",
            "-pix_fmt:v:1",
            "yuv420p",
            str(path),
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    _put_cover_before_tracks(path)


def _filtered_video(synth_media: object, key: str, vf: str) -> Path:
    def build(out: Path) -> None:
        ffmpeg.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(synth_media.path("clean")),  # type: ignore[attr-defined]
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(out),
            ],
            timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
        )

    return synth_media.cached(key, build)  # type: ignore[attr-defined]


@pytest.mark.parametrize("fixture", ["perclip", "clean"])
def test_grey_frames_count_matches_video_packets(synth_media: object, fixture: str) -> None:
    path = synth_media.path(fixture)  # type: ignore[attr-defined]
    assert len(video.grey_frames(path)) == ffmpeg.probe(path).video_frames


def test_grey_frames_decode_selected_video_after_leading_cover(tmp_path: Path) -> None:
    path = tmp_path / "leading-cover.mp4"
    _leading_cover_video(path)

    info = ffmpeg.probe(path)
    frames = video.grey_frames(path, w=32, h=18)
    assert info.video_index == 1
    assert info.audio_index is None
    assert info.video_frames == len(frames) == 10

    cover = _direct_first_frame(path, w=32, h=18)
    assert not np.array_equal(frames[0], cover)
    assert structure.check_structure(path).frames == 10


def test_vfr_grey_frames_use_the_requested_grid(synth_media) -> None:
    path = synth_media.cached("vfr", synth.vfr_source, suffix=".mov")
    info = ffmpeg.probe(path)
    assert video.source_rate(info, Fraction(30)) == 30
    conformed = video.grey_frames(path, rate=Fraction(30))
    assert len(conformed) == round(info.video_duration_s * 30) == 900
    # Each missing source frame holds its predecessor on the cut's grid.
    for dropped in (45, 75, 105, 135):
        np.testing.assert_array_equal(conformed[dropped], conformed[dropped - 1])
        assert not np.array_equal(conformed[dropped], conformed[dropped + 1])
    default = video.grey_frames(path)
    np.testing.assert_array_equal(default, video.grey_frames(path, rate=info.fps))
    assert len(default) == round(info.video_duration_s * float(info.fps)) == 896
    blocks = list(video.iter_grey_frames(path, rate=Fraction(30), block=127))
    assert [first for first, _ in blocks] == list(range(0, len(conformed), 127))
    np.testing.assert_array_equal(np.concatenate([frames for _, frames in blocks]), conformed)
    np.testing.assert_array_equal(
        video.grey_frames(path, rate=Fraction(30), start_frame=43, count=5), conformed[43:48]
    )


@pytest.mark.parametrize("rate", [Fraction(24), Fraction(25), Fraction(30000, 1001), Fraction(60)])
def test_cfr_source_keeps_its_native_grid(synth_media, rate: Fraction) -> None:
    info = replace(ffmpeg.probe(synth_media.path("calib")), fps=rate, nominal_fps=rate)
    assert video.source_rate(info, Fraction(30)) == rate


def test_grey_frames_perclip_starts_with_first_stream_frame(synth_media: object) -> None:
    path = synth_media.path("perclip")  # type: ignore[attr-defined]
    frames = video.grey_frames(path)
    direct = _direct_first_frame(path, w=frames.shape[2], h=frames.shape[1])
    changed_region = (slice(20, 32), slice(48, 64))
    assert not np.array_equal(frames[0], frames[1])
    assert video.frame_distance(
        frames[0][changed_region], frames[1][changed_region]
    ) > 0.05
    assert np.array_equal(frames[0], direct)


def test_grey_frames_window_matches_full_decode(synth_media: object) -> None:
    path = synth_media.path("calib")  # type: ignore[attr-defined]
    full = video.grey_frames(path)
    for first in (0, 1, 29, 30, 31):
        assert np.array_equal(
            video.grey_frames(path, start_frame=first, count=9), full[first : first + 9]
        )
    past = video.grey_frames(path, start_frame=len(full) + 10, count=9)
    assert past.shape == (0, full.shape[1], full.shape[2])


def test_locate_sequence_calibration(synth_media: object) -> None:
    shifted = _filtered_video(
        synth_media,
        "video-first-frame-removed",
        "trim=start_frame=1,setpts=PTS-STARTPTS",
    )
    scaled = _filtered_video(synth_media, "video-scaled-160x90", "scale=160:90")
    bright = _filtered_video(synth_media, "video-bright-01", "eq=brightness=0.1")
    source = video.grey_frames(synth_media.path("calib"))  # type: ignore[attr-defined]
    outputs = {
        "shifted": video.grey_frames(shifted),
        "scaled": video.grey_frames(scaled),
        "bright": video.grey_frames(bright),
    }
    frame_spans = [(round(start * 30), round(end * 30)) for start, end in synth.default_spans()]
    for segment in (0, 3, 6):
        source_first, source_last = frame_spans[segment]
        source_middle = (source_first + source_last) // 2
        needle = source[source_middle - 4 : source_middle + 5]
        planned = segment * 72 + 36 - 4
        for name, hay in outputs.items():
            expected = planned - 1 if name == "shifted" else planned
            index, distance, second = video.locate_sequence(
                needle, hay, expect=expected, search=25
            )
            assert index == expected
            assert second > distance


def _nearby(path: Path, middle: int) -> np.ndarray:
    return video.grey_frames(path, start_frame=middle - 2, count=5)


@pytest.mark.parametrize(
    ("fixture", "x_frac"), [("crop_center", 0.5), ("crop_offcenter", 0.2)]
)
def test_estimate_crop_mapping(synth_media: object, fixture: str, x_frac: float) -> None:
    source = _nearby(synth_media.path("calib"), 360)  # type: ignore[attr-defined]
    render = _nearby(synth_media.path(fixture), 360)  # type: ignore[attr-defined]
    mapping = video.estimate_mapping(source, render)
    expected_x = synth.crop_window(320, 180, x_frac)[0] * 64 / 320
    assert mapping.kind == "crop"
    assert mapping.x == pytest.approx(expected_x, abs=2.0)


def test_estimate_fit_mapping(synth_media: object) -> None:
    source = _nearby(synth_media.path("calib_vertical"), 360)  # type: ignore[attr-defined]
    render = _nearby(synth_media.path("fit_vertical"), 360)  # type: ignore[attr-defined]
    mapping = video.estimate_mapping(source, render)
    expected_scale = min(render.shape[2] / source.shape[2], render.shape[1] / source.shape[1])
    expected_x = (render.shape[2] - source.shape[2] * expected_scale) / 2
    expected_y = (render.shape[1] - source.shape[1] * expected_scale) / 2
    assert mapping.kind == "fit"
    assert mapping.x == pytest.approx(expected_x, abs=2.0)
    assert mapping.y == pytest.approx(expected_y, abs=2.0)


def test_apply_mapping_with_different_analysis_widths(synth_media: object) -> None:
    crop_source = video.grey_frames(
        synth_media.path("calib"),  # type: ignore[attr-defined]
        w=64,
        start_frame=358,
        count=5,
    )
    crop_render = video.grey_frames(
        synth_media.path("crop_offcenter"),  # type: ignore[attr-defined]
        w=96,
        start_frame=358,
        count=5,
    )
    crop_mapping = video.estimate_mapping(crop_source, crop_render)
    crop_size = (72, 40)
    mapped_crop_source = video.apply_mapping(
        crop_source, crop_mapping, role="source", size=crop_size
    )
    mapped_crop_render = video.apply_mapping(
        crop_render, crop_mapping, role="render", size=crop_size
    )
    crop_distances = [
        video.frame_distance(source, render)
        for source, render in zip(mapped_crop_source, mapped_crop_render, strict=True)
    ]
    expected_x = synth.crop_window(320, 180, 0.2)[0] * 64 / 320
    assert crop_mapping.kind == "crop"
    assert crop_mapping.x == pytest.approx(expected_x, abs=2.0)
    assert float(np.mean(crop_distances)) < 0.15

    fit_source = video.grey_frames(
        synth_media.path("calib_vertical"),  # type: ignore[attr-defined]
        w=64,
        start_frame=358,
        count=5,
    )
    fit_render = video.grey_frames(
        synth_media.path("fit_vertical"),  # type: ignore[attr-defined]
        w=96,
        start_frame=358,
        count=5,
    )
    fit_mapping = video.estimate_mapping(fit_source, fit_render)
    fit_size = (114, 64)
    mapped_fit_source = video.apply_mapping(
        fit_source, fit_mapping, role="source", size=fit_size
    )
    mapped_fit_render = video.apply_mapping(
        fit_render, fit_mapping, role="render", size=fit_size
    )
    fit_distances = [
        video.frame_distance(source, render)
        for source, render in zip(mapped_fit_source, mapped_fit_render, strict=True)
    ]
    assert fit_mapping.kind == "fit"
    assert float(np.mean(fit_distances)) < 0.25


def test_estimate_full_mapping(synth_media: object) -> None:
    source = _nearby(synth_media.path("calib"), 252)  # type: ignore[attr-defined]
    render = _nearby(synth_media.path("clean"), 180)  # type: ignore[attr-defined]
    mapping = video.estimate_mapping(source, render)
    assert mapping.kind == "full"


def test_rotation_decodes_upright(synth_media: object) -> None:
    rotated_path = synth_media.path("rotated_vertical")  # type: ignore[attr-defined]
    rotated = video.grey_frames(rotated_path)
    physical = video.grey_frames(synth_media.path("calib_vertical"))  # type: ignore[attr-defined]
    assert rotated.shape == physical.shape
    assert rotated.shape[1] > rotated.shape[2]
    distances = [video.frame_distance(a, b) for a, b in zip(rotated, physical, strict=True)]
    assert float(np.mean(distances)) < 0.15
    assert ffmpeg.probe(rotated_path).orientation == "vertical"


def test_frame_distance_and_best_of() -> None:
    first = np.arange(64, dtype=np.float32).reshape(8, 8)
    same_with_grade = first * 2.0 + 20.0
    other = np.flipud(first)
    assert video.frame_distance(first, same_with_grade) == pytest.approx(0.0, abs=1e-6)
    index, distances = video.best_of(first, np.stack((other, same_with_grade)))
    assert index == 1
    assert distances[1] < distances[0]


@pytest.mark.parametrize("fixture", ["clean", "perclip", "clean_crop"])
def test_grey_frames_uint8_matches_previous_float_decode(synth_media, fixture):
    path = synth_media.path(fixture)
    info = ffmpeg.probe(path, count_frames=False)
    frames = video.grey_frames(path)
    _, height, width = frames.shape
    result = ffmpeg.run(
        [
            "ffmpeg", "-loglevel", "error", "-i", str(path), "-map", "0:v:0", "-an", "-vf",
            f"setpts=PTS-STARTPTS,fps={info.fps.numerator}/{info.fps.denominator}"
            f":start_time=0:round=near,scale={width}:{height}:flags=area,format=gray",
            "-pix_fmt", "gray", "-f", "rawvideo", "-",
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
        binary=True,
    )
    previous = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width)
    previous = previous.astype(np.float32)
    assert frames.dtype == np.uint8
    assert frames.flags.writeable
    np.testing.assert_array_equal(frames, previous)


@pytest.mark.parametrize("chunk_size", [1, 7, 233])
def test_small_chunks_and_count_stop_ffmpeg_cleanly(synth_media, monkeypatch, chunk_size):
    path = synth_media.path("perclip")
    full = video.grey_frames(path, w=8, h=4)
    stream = ffmpeg.run_stream
    popen = ffmpeg.subprocess.Popen
    processes, sizes = [], []

    def recorded(command, **kwargs):
        process = popen(command, **kwargs)
        if "-frames:v" in command:
            assert command[command.index("-frames:v") + 1] == "26"
            processes.append(process)
        return process

    def tiny_chunks(command, *, timeout_s):
        for chunk in stream(command, timeout_s=timeout_s, chunk_size=chunk_size):
            sizes.append(len(chunk))
            yield chunk

    monkeypatch.setattr(ffmpeg.subprocess, "Popen", recorded)
    monkeypatch.setattr(ffmpeg, "run_stream", tiny_chunks)
    frames = video.grey_frames(path, w=8, h=4, start_frame=17, count=9)
    np.testing.assert_array_equal(frames, full[17:26])
    assert frames.dtype == np.uint8
    assert sum(sizes) == 26 * 8 * 4
    assert max(sizes) <= chunk_size
    assert len(processes) == 1
    assert processes[0].poll() == 0
    assert processes[0].stdout.closed


@pytest.mark.parametrize("block", [1, 16, 127, 1024])
def test_iter_grey_frames_preserves_indices_and_independent_blocks(synth_media, block):
    path = synth_media.path("perclip")
    full = video.grey_frames(path)
    blocks = list(video.iter_grey_frames(path, block=block))
    assert [first for first, _ in blocks] == list(range(0, len(full), block))
    for first, frames in blocks:
        assert frames.dtype == np.uint8
        assert 0 < len(frames) <= block
        np.testing.assert_array_equal(frames, full[first : first + block])
    if len(blocks) > 1:
        assert not np.shares_memory(blocks[0][1], blocks[1][1])


def test_empty_window_does_not_start_decoder(synth_media, monkeypatch):
    path = synth_media.path("clean")
    monkeypatch.setattr(ffmpeg, "run_stream", lambda *a, **kw: pytest.fail("decoded empty window"))
    frames = video.grey_frames(path, count=0)
    assert frames.shape == (0, 36, 64)
    assert frames.dtype == np.uint8


@pytest.mark.parametrize("kwargs", [{"w": 0}, {"h": 0}, {"start_frame": -1}, {"count": -1}])
def test_grey_frames_rejects_invalid_windows(synth_media, kwargs):
    with pytest.raises(ValidationError):
        video.grey_frames(synth_media.path("clean"), **kwargs)


def test_iter_grey_frames_rejects_empty_blocks():
    with pytest.raises(ValidationError):
        list(video.iter_grey_frames(Path("unused"), block=0))


@pytest.mark.parametrize("kind", ["full", "crop", "fit"])
@pytest.mark.parametrize("role", ["source", "render"])
def test_uint8_mapping_and_distances_match_float_frames(kind, role):
    frames = np.random.default_rng(71).integers(0, 256, (5, 12, 16), dtype=np.uint8)
    floating = frames.astype(np.float32)
    mapping = video.Mapping(kind, 1, 2.25, 1.5, 8, 9, 1)
    mapped = video.apply_mapping(frames, mapping, role=role, size=(7, 10))
    expected = video.apply_mapping(floating, mapping, role=role, size=(7, 10))
    assert mapped.dtype == np.float32
    np.testing.assert_array_equal(mapped, expected)
    assert video.frame_distance(frames[0], frames[1]) == video.frame_distance(
        floating[0], floating[1]
    )
