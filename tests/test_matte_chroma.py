"""Synthetic screen detection, colour keys and edge despill."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import synth
from test_matte_make import PASSTHROUGH, PIXELS, FakeHelper, decode_matte

from yap_studio import ffmpeg
from yap_studio.cli import main
from yap_studio.errors import ValidationError, YapError
from yap_studio.matte import make
from yap_studio.matte.model import load_matte

VISION_BACKEND = {"name": "vision-person-segmentation", "quality": "accurate", "revision": 1,
                  "mask_width": 1080, "mask_height": 1920, "os": "synthetic",
                  "helper_sha256": "a" * 64}


def mint_screen_clip(folder: Path) -> Path:
    """Put a large pale mint person with a soft edge on a green screen."""
    plate = folder / "mint-screen.mkv"
    ffmpeg.run([
        "ffmpeg", "-v", "error", "-filter_complex_threads", "1",
        "-f", "lavfi", "-i", "color=0x30b050:s=1080x1920:r=30,format=gbrp",
        "-f", "lavfi", "-i", "color=0xb9d5cc:s=1080x1920:r=30,format=gbrp",
        "-f", "lavfi", "-i", "color=black:s=1080x1920:r=30,format=gray,"
        "drawbox=x=300:y=700:w=480:h=600:c=white:t=fill,format=gray,"
        "lut=c0='if(gt(val,128),255,0)',gblur=sigma=8,format=gray,format=gbrp",
        "-filter_complex", "[0:v][1:v][2:v]maskedmerge,scale=out_color_matrix=bt709,"
        "format=yuv420p,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709",
        "-frames:v", "1", "-c:v", "ffv1", "-pix_fmt", "yuv420p",
        "-threads", "2", str(plate),
    ], timeout_s=60)
    return plate


def screen_clip(
    folder: Path, screen: str = "green", *, frames: int = 8, vignette: bool = False,
) -> Path:
    """Blur a moving disk over a noisy screen with a vertical light falloff."""
    colours = (48, 176, 80) if screen == "green" else (48, 80, 176)
    light = "1-0.5*pow((2*X/W-1)*(2*Y/H-1),2)" if vignette else "1"
    background = "nullsrc=s=108x192:r=30,format=gbrp,geq=" + ":".join(
        f"{channel}='({value}-12*Y/H+2*sin(X*19+Y*7+N*11))*({light})'"
        for channel, value in zip("rgb", colours, strict=True)
    )
    mint = "lt(abs(X-(32+5*N)),7)*lt(abs(Y-96),7)"
    foreground = "nullsrc=s=108x192:r=30,format=gbrp,geq=" + ":".join(
        f"{channel}='if({mint},{patch},{skin})'"
        for channel, patch, skin in zip("rgb", (185, 213, 204), (197, 142, 120), strict=True)
    )
    mask = ("nullsrc=s=108x192:r=30,format=gray,"
            "geq=lum='255*lt(hypot(X-(32+5*N),Y-96),24)',gblur=sigma=3")
    target = folder / f"{screen}.mkv"
    ffmpeg.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", background,
        "-f", "lavfi", "-i", foreground, "-f", "lavfi", "-i", mask,
        "-filter_complex", "[0:v][1:v][2:v]maskedmerge", "-frames:v", str(frames),
        "-c:v", "ffv1", "-pix_fmt", "gbrp", str(target),
    ], timeout_s=30)
    return target


@pytest.mark.parametrize("screen", ["green", "blue"])
def test_screen_key_tracks_soft_moving_person_and_keeps_mint(tmp_path: Path, screen: str) -> None:
    from yap_studio.matte import chroma

    video = screen_clip(tmp_path, screen)
    found = chroma.detect(video)
    assert found == chroma.detect(video)
    assert found and found["screen"] == screen and found["border_share"] > 0.5
    data = make.make_matte(video, tmp_path / "matte", cleanup=PASSTHROUGH)
    assert data.backend == {"name": "chroma-key", **found,
                            "lo": max(0.12, 0.4 * found["ratio"]),
                            "hi": max(max(0.12, 0.4 * found["ratio"]) + 0.05,
                                      0.7 * found["ratio"]), "requested": "auto"}
    assert load_matte(tmp_path / "matte/matte.json").backend == data.backend
    alpha = np.frombuffer(decode_matte(tmp_path / "matte/matte.mkv"), np.uint8).reshape(8, 192, 108)
    for frame in range(8):
        x = 32 + 5 * frame
        assert alpha[frame, 96, x] >= 250
        assert alpha[frame, 96, x - 12] >= 250
        assert alpha[frame, 4, 4] <= 5
        assert np.any((alpha[frame, 96, x + 19:x + 28] > 5)
                      & (alpha[frame, 96, x + 19:x + 28] < 250))
        assert abs(np.where(alpha[frame, 96] >= 128)[0].mean() - x) < 2
    assert len(data.coverage) == 8 and data.empty_frames == ()
    assert data.times == tuple(i / 30 for i in range(8))


@pytest.mark.parametrize("screen", ["green", "blue"])
def test_ratio_clears_half_brightness_corners_and_keeps_mint(tmp_path: Path, screen: str) -> None:
    video = screen_clip(tmp_path, screen, vignette=True)
    folder = tmp_path / "matte"
    data = make.make_matte(video, folder, cleanup=PASSTHROUGH)
    assert data.backend["name"] == "chroma-key"
    alpha = np.frombuffer(decode_matte(folder / "matte.mkv"), np.uint8).reshape(8, 192, 108)
    for corner in (alpha[:, :8, :8], alpha[:, :8, -8:],
                   alpha[:, -8:, :8], alpha[:, -8:, -8:]):
        assert corner.max() <= 5
    for frame in range(8):
        assert alpha[frame, 96, 32 + 5 * frame] >= 250


def test_detection_seeks_eight_small_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.matte import chroma

    video = screen_clip(tmp_path)
    commands = []
    run_stream = ffmpeg.run_stream

    def record(command, **kwargs):
        commands.append(command)
        return run_stream(command, **kwargs)

    monkeypatch.setattr(ffmpeg, "run_stream", record)
    assert chroma.detect(video) is not None
    assert len(commands) == 8
    for index, command in enumerate(commands):
        assert command.index("-ss") < command.index("-i")
        assert float(command[command.index("-ss") + 1]) == pytest.approx(index / 30)
        assert command[command.index("-frames:v") + 1] == "1"
        assert command[command.index("-vf") + 1] == "scale=108:192:flags=area"


def test_auto_room_counts_source_frames_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = synth.picture_source(tmp_path / "room.mp4", "color=gray:s=16x24:r=30", 3)
    fake = FakeHelper(tmp_path, monkeypatch).install([bytes([255]) * PIXELS] * 3)
    calls = []
    probe = ffmpeg.probe

    def counted(source, **kwargs):
        if Path(source) == video and kwargs.get("count_frames"):
            calls.append(source)
        return probe(source, **kwargs)

    monkeypatch.setattr(ffmpeg, "probe", counted)
    data = make.make_matte(video, tmp_path / "matte", cleanup=PASSTHROUGH)
    assert data.backend["name"] == make.BACKEND_NAME
    assert len(calls) == 1
    assert "requested" not in data.backend
    fake.assert_reaped()


def test_project_cutout_defaults_for_standalone_render(tmp_path: Path) -> None:
    from yap_studio.matte import chroma

    assert chroma.project_request(tmp_path) == "auto"


def test_chroma_joins_do_not_enable_vision_refinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.matte import frames

    video = screen_clip(tmp_path)
    monkeypatch.setattr(frames, "encode_vision", lambda *args: pytest.fail("Vision refinement"))
    data = make.make_matte(video, tmp_path / "joined", joins=(4,))
    assert data.cleanup.refine is False
    assert data.joins == ()
    make.make_matte(video, tmp_path / "continuous")
    assert decode_matte(tmp_path / "joined/matte.mkv") == decode_matte(
        tmp_path / "continuous/matte.mkv",
    )


@pytest.mark.parametrize("joins", [(0,), (8,), (4, 4), (5, 2), (True,)])
def test_chroma_validates_joins_before_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, joins: tuple,
) -> None:
    from yap_studio.matte import chroma

    video = screen_clip(tmp_path)
    monkeypatch.setattr(chroma, "_detect", lambda *args, **kwargs: pytest.fail("detection"))
    with pytest.raises(ValidationError, match="joins"):
        make.make_matte(video, tmp_path / "matte", joins=joins)


def test_room_with_small_green_poster_keeps_person_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.matte import chroma

    video = synth.picture_source(tmp_path / "room.mp4", "color=gray:s=16x24:r=30,"
                                 "drawbox=x=0:y=8:w=2:h=3:c=green:t=fill", 3)
    assert chroma.detect(video) is None
    fake = FakeHelper(tmp_path, monkeypatch).install([bytes([255]) * PIXELS] * 3)
    auto = make.make_matte(video, tmp_path / "auto", cleanup=PASSTHROUGH)
    person = make.make_matte(video, tmp_path / "person", backend="vision", cleanup=PASSTHROUGH)
    auto_doc, person_doc = auto.to_dict(), person.to_dict()
    # FFV1 containers have fresh identities even when their decoded pixels match.
    auto_doc["matte"].pop("sha256")
    person_doc["matte"].pop("sha256")
    assert auto_doc == person_doc
    assert decode_matte(tmp_path / "auto/matte.mkv") == decode_matte(tmp_path / "person/matte.mkv")
    assert "requested" not in auto.backend
    fake.assert_reaped()


def test_forced_chroma_refuses_room_with_person_fix(tmp_path: Path) -> None:
    video = synth.picture_source(tmp_path / "room.mp4", "color=gray:s=108x192:r=30", 3)
    with pytest.raises(YapError, match="No green or blue screen found at the edges") as error:
        make.make_matte(video, tmp_path / "matte", backend="chroma")
    assert "use the person cutout instead" in str(error.value)
    assert "--backend auto" in error.value.suggested_action
    assert not (tmp_path / "matte/matte.json").exists()


def test_chroma_cli_metadata(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    video = screen_clip(tmp_path)
    assert main(["matte", "make", str(video), "--out", str(tmp_path / "matte"),
                 "--backend", "chroma", "--json"]) == 0
    backend = json.loads(capsys.readouterr().out)["backend"]
    assert backend["name"] == "chroma-key" and backend["requested"] == "chroma"


def test_despill_rejects_unknown_cached_screen() -> None:
    from yap_studio.matte import chroma
    from yap_studio.stages.assemble_picture import _Graph

    graph = _Graph()
    with pytest.raises(ValidationError, match="green or blue"):
        chroma.despill(graph, "person", "mask", "red")
    assert graph.filters == []


@pytest.mark.parametrize("screen", ["green", "blue"])
def test_despill_changes_only_soft_rim_over_background(tmp_path: Path, screen: str) -> None:
    from yap_studio.matte import chroma
    from yap_studio.stages.assemble_picture import _Graph

    video = screen_clip(tmp_path, screen)
    matte_dir = tmp_path / "matte"
    data = make.make_matte(video, matte_dir, cleanup=PASSTHROUGH)
    graph = _Graph()
    graph.filters.append("[0:v]split[original][person]")
    graph.filters.append("[1:v]split[plain_mask][edge_mask]")
    person, mask = chroma.despill(graph, "person", "edge_mask", screen)
    graph.filters.append(f"[{person}][{mask}]alphamerge[keyed]")
    graph.filters.append("[original][plain_mask]alphamerge[plain]")
    graph.filters.append("[2:v]split[bg1][bg2]")
    graph.filters.append("[bg1][keyed]overlay=format=rgb[after]")
    graph.filters.append("[bg2][plain]overlay=format=rgb[before]")
    graph.filters.append("[before][after]hstack,format=rgb24[out]")
    raw = ffmpeg.run([
        "ffmpeg", "-v", "error", "-i", str(video), "-i", str(matte_dir / "matte.mkv"),
        "-f", "lavfi", "-i", "color=gray:s=108x192:r=30",
        "-filter_complex", ";".join(graph.filters), "-map", "[out]", "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ], timeout_s=30, binary=True).stdout
    pixels = np.frombuffer(raw, np.uint8).reshape(192, 216, 3).astype(int)
    assert np.max(np.abs(pixels[96, 32] - pixels[96, 140])) <= 1
    channel = 1 if screen == "green" else 2
    def excess(pixel: np.ndarray) -> int:
        return int(pixel[channel] - np.delete(pixel, channel).max())
    assert any(excess(pixels[96, x + 108]) < excess(pixels[96, x]) - 3
               for x in range(51, 61))
    assert data.backend["screen"] == screen
