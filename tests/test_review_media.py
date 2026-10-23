"""Probe review video facts and frame-rate consistency."""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from yap_studio import ffmpeg, receipt
from yap_studio.errors import InputError
from yap_studio.review.media import (
    KnownMedia,
    _picture_duration,
    _tag_duration,
    file_identity,
    is_constant_rate,
    probe_media,
)


def test_known_media_accepts_identity_change_only_for_matching_content(tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"original")
    original = file_identity(video)
    known = KnownMedia()
    sha256 = receipt.sha256_file(video)
    assert not known.changed(video, sha256, original)
    replacement = tmp_path / "replacement.mp4"
    replacement.write_bytes(b"original")
    replacement.replace(video)
    assert not known.changed(video, sha256, original)
    video.write_bytes(b"different")
    assert known.changed(video, sha256, original)


@pytest.mark.parametrize(
    ("reference", "average", "expected"),
    [
        ("30000/1001", "60000/2002", True),
        ("30/1", "30000/1001", False),
        ("0/0", "30/1", False),
        ("bad", "30/1", False),
        ("30/1", None, False),
        ("0/1", "0/1", False),
    ],
)
def test_constant_rate_rule(reference: str | None, average: str | None, expected: bool) -> None:
    assert is_constant_rate(reference, average) is expected


def _make_media(output: Path, source: str, *, video: bool) -> None:
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", source,
    ]
    if video:
        command += ["-map", "0:v:0", "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    else:
        command += ["-map", "0:a:0", "-t", "2", "-c:a", "aac"]
    subprocess.run(
        [*command, str(output)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=60,
        check=True,
    )


def test_probe_constant_rate_h264_video(tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    _make_media(video, "testsrc2=size=64x36:rate=30000/1001", video=True)
    media = probe_media(video)
    assert media.path == video.resolve()
    assert media.name == "sample.mp4"
    assert media.duration_s == pytest.approx(2.0, abs=0.05)
    assert media.fps == Fraction(30000, 1001)
    assert media.cfr is True
    assert (media.codec, media.width, media.height) == ("h264", 64, 36)
    assert media.facts("a" * 64).sha256 == "a" * 64
    assert media.facts("a" * 64).fps == Fraction(30000, 1001)


def _make_packet_timing_video(
    output: Path, *, b_frames: int, dropped_frames: tuple[int, ...] = ()
) -> None:
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=64x36:rate=30:duration=10", "-map", "0:v:0",
    ]
    if dropped_frames:
        selection = "*".join(f"not(eq(n\\,{frame}))" for frame in dropped_frames)
        command += ["-vf", f"select={selection}"]
    command += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", str(b_frames), "-g", "31",
        # Keep the reorder pattern stable and the 240-packet boundary inside a GOP.
        "-x264-params", "b-adapt=0:scenecut=0", "-fps_mode", "passthrough", str(output),
    ]
    subprocess.run(
        command, stdin=subprocess.DEVNULL, capture_output=True, timeout=60, check=True,
    )


@pytest.mark.parametrize("b_frames", [3, 0], ids=["b_frames", "no_b_frames"])
def test_probe_constant_rate_h264_packet_window(tmp_path: Path, b_frames: int) -> None:
    video = tmp_path / "sample.mp4"
    _make_packet_timing_video(video, b_frames=b_frames)
    media = probe_media(video)
    assert media.cfr is True
    assert media.fps == Fraction(30)


@pytest.mark.parametrize(
    "dropped_frames", [(50, 120), (225,)], ids=["early_gaps", "near_window_end"]
)
def test_probe_variable_rate_h264_packet_window(
    tmp_path: Path, dropped_frames: tuple[int, ...]
) -> None:
    # This container preserves nominal summary rates, so packet gaps decide CFR.
    video = tmp_path / "sample.mkv"
    _make_packet_timing_video(video, b_frames=3, dropped_frames=dropped_frames)
    rates = json.loads(ffmpeg.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,avg_frame_rate", "-of", "json", str(video)],
        timeout_s=ffmpeg.PROBE_TIMEOUT_S,
    ).stdout)["streams"][0]
    assert Fraction(rates["r_frame_rate"]) == Fraction(rates["avg_frame_rate"]) == Fraction(30)
    media = probe_media(video)
    assert media.cfr is False
    assert media.fps is None


def test_probe_uses_video_duration_when_audio_outlasts_video(tmp_path: Path) -> None:
    video = tmp_path / "sample.mkv"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-t", "1", "-f", "lavfi", "-i", "testsrc2=size=64x36:rate=30000/1001",
            "-t", "3", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(video),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=60,
        check=True,
    )
    assert ffmpeg.probe(video).duration_s == pytest.approx(3.0, abs=0.1)
    assert probe_media(video).duration_s == pytest.approx(1.0, abs=0.1)


def test_probe_skips_attached_cover_art(tmp_path: Path) -> None:
    art = tmp_path / "cover.png"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "color=c=red:s=64x64", "-map", "0:v:0", "-frames:v", "1", str(art)],
        stdin=subprocess.DEVNULL, capture_output=True, timeout=60, check=True,
    )
    video = tmp_path / "sample.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
         "-i", str(art), "-f", "lavfi", "-i", "testsrc2=size=64x36:rate=30000/1001",
         "-map", "0:v:0", "-map", "1:v:0", "-t", "1",
         "-c:v:0", "png", "-disposition:v:0", "attached_pic",
         "-c:v:1", "libx264", "-pix_fmt:v:1", "yuv420p", str(video)],
        stdin=subprocess.DEVNULL, capture_output=True, timeout=60, check=True,
    )
    streams = json.loads(ffmpeg.run(
        ["ffprobe", "-v", "error", "-select_streams", "v",
         "-show_entries", "stream=index:stream_disposition=attached_pic",
         "-of", "json", str(video)], timeout_s=ffmpeg.PROBE_TIMEOUT_S,
    ).stdout)["streams"]
    assert any(stream["disposition"]["attached_pic"] == 1 for stream in streams)
    assert probe_media(video).fps == Fraction(30000, 1001)


def test_probe_selects_first_non_artwork_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"placeholder")
    info = SimpleNamespace(has_video=True, video_codec="h264", width=64, height=36)
    monkeypatch.setattr(ffmpeg, "probe", lambda *_args, **_kwargs: info)
    monkeypatch.setattr(
        ffmpeg, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout=json.dumps({
            "streams": [
                {"index": 0, "disposition": {"attached_pic": 1},
                 "r_frame_rate": "25/1", "avg_frame_rate": "25/1", "duration": "1"},
                {"index": 1, "disposition": {"attached_pic": 0},
                 "r_frame_rate": "30000/1001", "avg_frame_rate": "30000/1001",
                 "duration": "1.001"},
            ],
        })),
    )
    media = probe_media(source)
    assert media.fps == Fraction(30000, 1001)
    assert media.duration_s == pytest.approx(1.001)


@pytest.mark.parametrize(
    ("points", "constant"),
    [([0, 1000, 2000, 3000], True),
     ([0, 1000, 2100, 3100], False),
     ([0, 2000, 1000, 3000], True)],
)
def test_probe_checks_presentation_packet_gaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, points: list[int], constant: bool
) -> None:
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"placeholder")
    info = SimpleNamespace(has_video=True, video_codec="h264", width=64, height=36)
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, timeout_s: float) -> SimpleNamespace:
        assert timeout_s == ffmpeg.PROBE_TIMEOUT_S
        commands.append(command)
        if "-read_intervals" in command:
            return SimpleNamespace(stdout=json.dumps({
                "packets": [{"pts": str(point)} for point in points],
            }))
        return SimpleNamespace(stdout=json.dumps({"streams": [{
            "index": 2, "r_frame_rate": "30/1", "avg_frame_rate": "30/1",
            "time_base": "1/30000", "duration": "1",
        }]}))

    monkeypatch.setattr(ffmpeg, "probe", lambda *_args, **_kwargs: info)
    monkeypatch.setattr(ffmpeg, "run", fake_run)
    media = probe_media(source)
    assert (media.cfr, media.fps) == (constant, Fraction(30) if constant else None)
    assert commands[1] == [
        "ffprobe", "-v", "error", "-select_streams", "2", "-read_intervals", "%+#272",
        "-show_entries", "packet=pts", "-of", "json", str(source.resolve()),
    ]


@pytest.mark.parametrize("packet_failure", [False, True])
def test_probe_keeps_summary_when_packet_timing_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, packet_failure: bool
) -> None:
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"placeholder")
    info = SimpleNamespace(has_video=True, video_codec="h264", width=64, height=36)

    def fake_run(command: list[str], *, timeout_s: float) -> SimpleNamespace:
        if "-read_intervals" in command:
            if packet_failure:
                raise InputError(str(source), "packet query failed")
            return SimpleNamespace(stdout='{"packets":[{"pts":"0"},{"pts":"N/A"}]}')
        return SimpleNamespace(stdout=json.dumps({"streams": [{
            "index": 0, "r_frame_rate": "30/1", "avg_frame_rate": "30/1",
            "time_base": "1/30000", "duration": "1",
        }]}))

    monkeypatch.setattr(ffmpeg, "probe", lambda *_args, **_kwargs: info)
    monkeypatch.setattr(ffmpeg, "run", fake_run)
    assert probe_media(source).fps == Fraction(30)


def test_audio_only_has_no_video_stream(tmp_path: Path) -> None:
    audio = tmp_path / "tone.m4a"
    _make_media(audio, "sine=frequency=440:sample_rate=48000", video=False)
    with pytest.raises(InputError) as error:
        probe_media(audio)
    assert error.value.path == str(audio.resolve())
    assert error.value.reason == "no video stream"


def test_probe_uses_selected_stream_duration_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"placeholder")
    info = SimpleNamespace(
        has_video=True,
        duration_s=3.0,
        video_codec="h264",
        width=64,
        height=36,
    )
    calls: list[tuple[list[str], float]] = []
    stream_output = {
        "value": '{"streams":[{"index":0,"r_frame_rate":"30/1",'
        '"avg_frame_rate":"29/1","duration":"1.5"}]}'
    }

    def fake_probe(path: Path, *, count_frames: bool) -> SimpleNamespace:
        assert path == source.resolve()
        assert count_frames is False
        return info

    def fake_run(command: list[str], *, timeout_s: float) -> SimpleNamespace:
        calls.append((command, timeout_s))
        return SimpleNamespace(stdout=stream_output["value"])

    monkeypatch.setattr(ffmpeg, "probe", fake_probe)
    monkeypatch.setattr(ffmpeg, "run", fake_run)
    media = probe_media(source)
    assert (media.cfr, media.fps, media.duration_s) == (False, None, 1.5)
    stream_output["value"] = (
        '{"streams":[{"index":0,"r_frame_rate":"30/1","avg_frame_rate":"29/1"}]}'
    )
    assert probe_media(source).duration_s is None
    assert calls == [
        (
            [
                "ffprobe", "-v", "error", "-select_streams", "v",
                "-show_entries",
                "stream=index,r_frame_rate,avg_frame_rate,time_base,duration,nb_frames:"
                "stream_disposition=attached_pic:stream_tags=DURATION",
                "-of", "json", str(source.resolve()),
            ],
            ffmpeg.PROBE_TIMEOUT_S,
        )
    ] * 2


@pytest.mark.parametrize(
    ("tag", "expected"),
    [("00:00:01.250000000", 1.25), ("01:02:03.5", 3723.5),
     ("00:00:00.000000000", None), ("bad", None)],
)
def test_tag_duration(tag: str, expected: float | None) -> None:
    assert _tag_duration(tag) == expected


def test_picture_duration_uses_tag_before_frame_count(tmp_path: Path) -> None:
    stream = {"tags": {"DURATION": "00:00:01.000000000"}, "nb_frames": "90"}
    assert _picture_duration(stream, tmp_path / "sample.mkv", Fraction(30)) == 1.0


def test_picture_duration_uses_frame_count(tmp_path: Path) -> None:
    stream = {"nb_frames": "30"}
    assert _picture_duration(stream, tmp_path / "sample.mkv", Fraction(30)) == 1.0


def test_picture_duration_counts_selected_stream_packets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sample.mkv"
    calls = []

    def fake_run(command: list[str], *, timeout_s: float) -> SimpleNamespace:
        calls.append((command, timeout_s))
        return SimpleNamespace(stdout='{"streams":[{"nb_read_packets":"30"}]}')

    monkeypatch.setattr(ffmpeg, "run", fake_run)
    assert _picture_duration({"index": 2}, source, Fraction(30)) == 1.0
    assert calls == [(
        ["ffprobe", "-v", "error", "-count_packets", "-select_streams", "2",
         "-show_entries", "stream=nb_read_packets", "-of", "json", str(source)],
        ffmpeg.PROBE_TIMEOUT_S,
    )]


def test_picture_duration_unknown_ignores_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ffmpeg, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout='{"streams":[{}]}')
    )
    assert _picture_duration({"index": 0}, tmp_path / "sample.mkv", Fraction(30)) is None
    assert _picture_duration({"index": 0}, tmp_path / "sample.mkv", None) is None
