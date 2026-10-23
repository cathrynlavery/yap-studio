"""Real ffmpeg pipelines fed by deterministic streaming person mask helpers."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from yap_studio import ffmpeg, paths, receipt, tools
from yap_studio.errors import DependencyError, OutputPathError, ProcessingError
from yap_studio.matte import hyperframes, make, vision
from yap_studio.matte.model import Cleanup, load_matte


@pytest.mark.parametrize("cleanup", [Cleanup(), Cleanup(low=0, high=1, feather_sigma=8)])
def test_spatial_cleanup_preserves_all_zero_frames(tmp_path: Path, cleanup: Cleanup) -> None:
    from yap_studio.faces.model import VideoIdentity

    video = VideoIdentity("zero", "a" * 64, 1, Fraction(30), WIDTH, HEIGHT)
    output = tmp_path / "zero.mkv"
    make._encode_frames(iter([bytes(PIXELS)]), output, video, cleanup, time.monotonic() + 30)
    assert decode_matte(output) == bytes(PIXELS)

WIDTH, HEIGHT = 16, 24
PIXELS = WIDTH * HEIGHT
PASSTHROUGH = Cleanup(median=False, low=0, high=255, feather_sigma=0, refine=False)


def make_clip(path: Path, *, frames: int = 3, fps: Fraction | int = 30) -> Path:
    """Create a small synthetic input with an exact frame count."""
    ffmpeg.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={WIDTH}x{HEIGHT}:rate={fps}",
            "-frames:v",
            str(frames),
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        timeout_s=30,
    )
    return path


def decode_matte(path: Path) -> bytes:
    """Decode the lossless gray pixels to test the actual cleanup result."""
    return ffmpeg.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ],
        timeout_s=30,
        binary=True,
    ).stdout


class FakeHelper:
    """Install an executable that emits independently chosen pixels and records."""

    def __init__(self, folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.folder = folder
        self.helper = folder / "fake-person-matte"
        self.pid_file = folder / "helper.pid"
        self.monkeypatch = monkeypatch

    def install(
        self,
        frames: list[bytes],
        *,
        edit: Callable[[list[dict[str, Any]]], None] | None = None,
        times: list[float] | None = None,
        raw: bytes | None = None,
        failure: str | None = None,
    ) -> FakeHelper:
        """Write a fake helper and patch only the executable and availability seams."""
        times = times if times is not None else [i / 30 for i in range(len(frames))]
        records = [
            {
                "type": "header",
                "width": WIDTH,
                "height": HEIGHT,
                "mask_width": 12,
                "mask_height": 18,
                "quality": "accurate",
                "revision": 1,
            }
        ]
        records.extend(
            {
                "type": "frame",
                "frame": i,
                "pts": times[i],
                "coverage": sum(pixel >= 128 for pixel in frame) / len(frame),
            }
            for i, frame in enumerate(frames)
        )
        records.append({"type": "end", "frames": len(frames), "end_pts": times[-1] + 1 / 30})
        if edit is not None:
            edit(records)
        payload = b"".join(frames) if raw is None else raw
        script = f"""#!{sys.executable}
import json
import os
import sys
import time
from pathlib import Path

Path({str(self.pid_file)!r}).write_text(str(os.getpid()))
failure = {failure!r}
if failure == 'exit':
    sys.stderr.write('Vision failed: deliberate segmentation failure\\n')
    sys.exit(4)
if failure == 'hang':
    time.sleep(60)
arguments = sys.argv[1:]
records = json.loads({json.dumps(records)!r})
if records and records[0].get('type') == 'header' and records[0].get('quality') == 'accurate':
    records[0]['quality'] = arguments[arguments.index('--quality') + 1]
output = Path(arguments[arguments.index('--records') + 1])
output.write_text(''.join(json.dumps(record, sort_keys=True) + '\\n' for record in records))
sys.stdout.buffer.write(bytes.fromhex({payload.hex()!r}))
sys.stdout.buffer.flush()
"""
        self.helper.write_text(script, encoding="utf-8")
        self.helper.chmod(0o755)
        self.monkeypatch.setattr(vision, "helper_executable", lambda: self.helper)
        self.monkeypatch.setattr(vision, "unavailable_reason", lambda: None)
        return self

    def assert_reaped(self) -> None:
        """Check that the operating system no longer has the helper process."""
        pid = int(self.pid_file.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    return make_clip(tmp_path / "clip.mp4")


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHelper:
    return FakeHelper(tmp_path, monkeypatch)


def test_person_request_is_present_in_only_document_write(
    clip: Path, tmp_path: Path, fake: FakeHelper, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake.install([bytes([255]) * PIXELS] * 3)
    writes = []
    write = paths.atomic_write_text

    def record(document: Path, payload: str, **kwargs) -> None:
        writes.append(json.loads(payload))
        write(document, payload, **kwargs)

    monkeypatch.setattr(paths, "atomic_write_text", record)
    data = make.make_matte(clip, tmp_path / "matte", backend="vision", requested="person",
                           cleanup=PASSTHROUGH)
    assert len(writes) == 1
    assert writes[0]["backend"]["requested"] == "person"
    assert load_matte(tmp_path / "matte/matte.json") == data
    fake.assert_reaped()


def test_matte_lossless_pixels_count_size_and_identity(
    clip: Path, tmp_path: Path, fake: FakeHelper
) -> None:
    frames = [bytes((pixel + offset) % 256 for pixel in range(PIXELS)) for offset in (0, 53, 99)]
    fake.install(frames, times=[0.0, 0.07, 0.1])
    before = clip.read_bytes()
    out = tmp_path / "new" / "matte"
    data = make.make_matte(clip, out, cleanup=PASSTHROUGH, quality="balanced")
    assert data.status == "measured"
    assert data.video.frames == 3
    assert (data.video.width, data.video.height) == (WIDTH, HEIGHT)
    assert data.video.sha256 == receipt.sha256_file(clip)
    assert clip.read_bytes() == before
    assert decode_matte(out / "matte.mkv") == b"".join(frames)
    info = ffmpeg.probe(out / "matte.mkv", count_frames=True)
    assert (info.video_frames, info.width, info.height) == (3, WIDTH, HEIGHT)
    assert (info.video_codec, info.pix_fmt) == ("ffv1", "gray")
    assert load_matte(out / "matte.json") == data
    assert data.times == (0.0, 0.07, 0.1)
    assert data.end_time == pytest.approx(0.1 + 1 / 30)
    assert data.coverage == tuple(sum(p >= 128 for p in f) / PIXELS for f in frames)
    assert data.backend["quality"] == "balanced"
    assert data.backend["helper_sha256"] == receipt.sha256_file(vision.helper_source())
    assert data.matte is not None
    assert data.matte["sha256"] == receipt.sha256_file(out / "matte.mkv")
    assert sorted(p.name for p in out.iterdir()) == ["matte.json", "matte.mkv"]
    fake.assert_reaped()


@pytest.mark.parametrize("frames", [1, 2, 3, 5])
def test_default_median_preserves_endpoints_and_fills_dropout(
    tmp_path: Path, fake: FakeHelper, frames: int
) -> None:
    clip = make_clip(tmp_path / "clip.mp4", frames=frames)
    masks = [bytes([255]) * PIXELS for _ in range(frames)]
    if frames >= 3:
        masks[1] = bytes(PIXELS)
    fake.install(masks)
    out = tmp_path / "out"
    data = make.make_matte(clip, out)
    assert decode_matte(out / "matte.mkv") == bytes([255]) * PIXELS * frames
    assert ffmpeg.probe(out / "matte.mkv").video_frames == frames
    assert data.coverage == tuple(0.0 if frame == bytes(PIXELS) else 1.0 for frame in masks)


def test_levels_clip_and_map_low_high(clip: Path, tmp_path: Path, fake: FakeHelper) -> None:
    pixels = bytes([0, 8, 9, 127, 246, 247, 248, 255]) * (PIXELS // 8)
    fake.install([pixels] * 3)
    out = tmp_path / "out"
    make.make_matte(clip, out, cleanup=Cleanup(median=False, feather_sigma=0, refine=False))
    expected = bytes(max(0, min(255, int((value - 8) * 255 / (247 - 8)))) for value in pixels)
    assert decode_matte(out / "matte.mkv") == expected * 3


@pytest.mark.parametrize(
    "failure",
    [
        "missing-header",
        "gap",
        "decreasing-pts",
        "wrong-count",
        "missing-end",
        "after-end",
        "coverage-bool",
        "pts-nan",
        "end-before-last",
        "coverage-range",
        "wrong-quality",
        "bad-dimensions",
        "duplicate-header",
    ],
)
def test_invalid_records_are_rejected_without_publication(
    clip: Path, tmp_path: Path, fake: FakeHelper, failure: str
) -> None:
    def edit(records: list[dict[str, Any]]) -> None:
        if failure == "missing-header":
            records.pop(0)
        elif failure == "gap":
            records[2]["frame"] = 2
        elif failure == "decreasing-pts":
            records[3]["pts"] = 0.0
        elif failure == "wrong-count":
            records[-1]["frames"] = 4
        elif failure == "missing-end":
            records.pop()
        elif failure == "after-end":
            records.append(records[1].copy())
        elif failure == "coverage-bool":
            records[1]["coverage"] = True
        elif failure == "pts-nan":
            records[1]["pts"] = float("nan")
        elif failure == "end-before-last":
            records[-1]["end_pts"] = 0.0
        elif failure == "coverage-range":
            records[1]["coverage"] = 1.01
        elif failure == "wrong-quality":
            records[0]["quality"] = "fast"
        elif failure == "bad-dimensions":
            records[0]["width"] = WIDTH + 1
        elif failure == "duplicate-header":
            records.insert(1, records[0].copy())

    fake.install([bytes(PIXELS)] * 3, edit=edit)
    before = clip.read_bytes()
    out = tmp_path / "out"
    with pytest.raises(ProcessingError) as caught:
        make.make_matte(clip, out)
    assert caught.value.code == "person_matte_output"
    assert list(out.iterdir()) == []
    assert clip.read_bytes() == before
    fake.assert_reaped()


def test_helper_exit_preserves_reason(clip: Path, tmp_path: Path, fake: FakeHelper) -> None:
    fake.install([bytes(PIXELS)] * 3, failure="exit")
    with pytest.raises(ProcessingError) as caught:
        make.make_matte(clip, tmp_path / "out")
    assert caught.value.code == "person_matte_exit_4"
    assert "Vision failed: deliberate segmentation failure" in caught.value.message
    assert list((tmp_path / "out").iterdir()) == []
    fake.assert_reaped()


def test_helper_timeout_kills_children_and_removes_outputs(
    clip: Path, tmp_path: Path, fake: FakeHelper, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake.install([bytes(PIXELS)] * 3, failure="hang")
    monkeypatch.setattr(make, "_deadline_seconds", lambda frames: 0.3)
    started = time.monotonic()
    with pytest.raises(ProcessingError) as caught:
        make.make_matte(clip, tmp_path / "out")
    assert caught.value.code == "person_matte_timeout"
    assert time.monotonic() - started < 5
    assert list((tmp_path / "out").iterdir()) == []
    fake.assert_reaped()


@pytest.mark.parametrize("failure", ["exit", "hang"])
def test_encoder_error_or_timeout_reaps_both_children(
    clip: Path,
    tmp_path: Path,
    fake: FakeHelper,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    fake.install([bytes(PIXELS)] * 3, failure="hang" if failure == "exit" else None)
    streams: list[vision.HelperStream] = []
    stream_helper = vision.stream_helper

    @contextmanager
    def capture_helper(video: Path, *, quality: str) -> Iterator[vision.HelperStream]:
        with stream_helper(video, quality=quality) as stream:
            streams.append(stream)
            yield stream

    monkeypatch.setattr(vision, "stream_helper", capture_helper)
    encoder = tmp_path / "deliberate-encoder"
    encoder_pid = tmp_path / "encoder.pid"
    encoder.write_text(
        f"#!{sys.executable}\n"
        "import os, sys, time\n"
        "from pathlib import Path\n"
        f"Path({str(encoder_pid)!r}).write_text(str(os.getpid()))\n"
        + (
            "time.sleep(60)\n"
            if failure == "hang"
            else "sys.stderr.write('deliberate encoder failure\\n')\nsys.exit(7)\n"
        ),
        encoding="utf-8",
    )
    encoder.chmod(0o755)
    safe_command = ffmpeg._safe_command
    monkeypatch.setattr(
        ffmpeg,
        "_safe_command",
        lambda command: [str(encoder)]
        if "-vf" in command and "pipe:0" in command else safe_command(command),
    )
    monkeypatch.setattr(make, "_deadline_seconds", lambda frames: 0.5 if failure == "hang" else 10)
    started = time.monotonic()
    with pytest.raises(ProcessingError) as caught:
        make.make_matte(clip, tmp_path / "out")
    if failure == "hang":
        assert caught.value.code == "person_matte_timeout"
    else:
        assert caught.value.code == "ffmpeg_exit_7"
        assert "deliberate encoder failure" in caught.value.message
    assert time.monotonic() - started < 5
    assert len(streams) == 1 and streams[0].process.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.kill(streams[0].process.pid, 0)
    with pytest.raises(ProcessLookupError):
        os.kill(int(encoder_pid.read_text()), 0)
    assert list((tmp_path / "out").iterdir()) == []


def test_encoded_frame_count_mismatch_is_rejected(
    clip: Path, tmp_path: Path, fake: FakeHelper
) -> None:
    fake.install([bytes(PIXELS)] * 3, raw=bytes(PIXELS * 2))
    with pytest.raises(ProcessingError) as caught:
        make.make_matte(clip, tmp_path / "out", cleanup=PASSTHROUGH)
    assert caught.value.code == "matte_frame_count"
    assert list((tmp_path / "out").iterdir()) == []


def test_encoded_dimensions_mismatch_is_rejected(
    clip: Path, tmp_path: Path, fake: FakeHelper, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake.install([bytes(PIXELS)] * 3)
    probe = ffmpeg.probe

    def wrong_size(path: Path, *, count_frames: bool = True) -> ffmpeg.MediaInfo:
        info = probe(path, count_frames=count_frames)
        return replace(info, width=WIDTH + 1) if path.suffix == ".mkv" else info

    monkeypatch.setattr(ffmpeg, "probe", wrong_size)
    with pytest.raises(ProcessingError) as caught:
        make.make_matte(clip, tmp_path / "out")
    assert caught.value.code == "matte_frame_count"
    assert list((tmp_path / "out").iterdir()) == []


@pytest.mark.parametrize("filename", ["matte.mkv", "matte.json"])
def test_existing_unowned_output_is_untouched(
    clip: Path, tmp_path: Path, fake: FakeHelper, filename: str
) -> None:
    fake.install([bytes(PIXELS)] * 3)
    out = tmp_path / "out"
    out.mkdir()
    existing = out / filename
    existing.write_bytes(b"keep me")
    with pytest.raises(OutputPathError) as caught:
        make.make_matte(clip, out)
    assert caught.value.code == "unsafe_path"
    assert existing.read_bytes() == b"keep me"
    assert not fake.pid_file.exists()
    assert list(out.iterdir()) == [existing]


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_input_is_protected_from_output_aliases(
    tmp_path: Path, fake: FakeHelper, alias: str
) -> None:
    clip = make_clip(tmp_path / ("matte.mkv" if alias == "same" else "clip.mp4"))
    fake.install([bytes(PIXELS)] * 3)
    before = clip.read_bytes()
    out = tmp_path if alias == "same" else tmp_path / "out"
    out.mkdir(exist_ok=True)
    if alias == "symlink":
        (out / "matte.mkv").symlink_to(clip)
    elif alias == "hardlink":
        (out / "matte.mkv").hardlink_to(clip)
    with pytest.raises(OutputPathError, match="input file"):
        make.make_matte(clip, out)
    assert clip.read_bytes() == before
    assert not fake.pid_file.exists()


def test_document_failure_rolls_back_matte(
    clip: Path, tmp_path: Path, fake: FakeHelper, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake.install([bytes(PIXELS)] * 3)

    def fail(*args: Any, **kwargs: Any) -> Path:
        raise OutputPathError("deliberate publication failure")

    monkeypatch.setattr(paths, "atomic_write_text", fail)
    with pytest.raises(OutputPathError, match="deliberate"):
        make.make_matte(clip, tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []


def test_unavailable_auto_writes_only_document(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reason = "Vision is unavailable here"
    monkeypatch.setattr(vision, "unavailable_reason", lambda: reason)
    monkeypatch.setattr(hyperframes, "unavailable_reason", lambda: "no renderer")
    monkeypatch.setattr(vision, "helper_executable", lambda: pytest.fail("unexpected helper"))
    out = tmp_path / "out"
    data = make.make_matte(clip, out)
    assert data.status == "unavailable" and data.reason == f"{reason}; no renderer"
    assert data.backend == {} and data.matte is None and data.times is None
    assert data.end_time is None and data.coverage is None
    assert load_matte(out / "matte.json") == data
    assert [path.name for path in out.iterdir()] == ["matte.json"]


def test_auto_prefers_vision_without_querying_renderer(
    clip: Path, tmp_path: Path, fake: FakeHelper, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake.install([bytes(PIXELS)] * 3)
    monkeypatch.setattr(
        hyperframes, "unavailable_reason", lambda: pytest.fail("unexpected fallback query")
    )
    data = make.make_matte(clip, tmp_path / "out", cleanup=PASSTHROUGH)
    assert data.backend["name"] == "vision-person-segmentation"
    assert data.empty_frames == ()
    fake.assert_reaped()


def test_explicit_vision_unavailable_is_dependency_error(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vision, "unavailable_reason", lambda: "no Vision")
    with pytest.raises(DependencyError, match="no Vision") as caught:
        make.make_matte(clip, tmp_path / "out", backend="vision")
    assert caught.value.suggested_action
    assert not (tmp_path / "out").exists()


_REAL_HELPER = pytest.mark.skipif(
    sys.platform != "darwin" or tools.find_tool("xcrun") is None,
    reason="real Vision helper needs macOS and xcrun",
)


@pytest.fixture
def real_helper(_owner_folders: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Reuse the linked helper and keep compiler caches inside this test's owner folder."""
    monkeypatch.setenv("CLANG_MODULE_CACHE_PATH", str(_owner_folders / "clang"))
    monkeypatch.setenv("SWIFT_MODULECACHE_PATH", str(_owner_folders / "swift"))
    return vision.helper_executable()


@pytest.mark.helpers
@_REAL_HELPER
@pytest.mark.parametrize("quality", ["accurate", "balanced"])
def test_real_vision_helper_valid_records_and_matte(
    tmp_path: Path, real_helper: Path, quality: str
) -> None:
    clip = make_clip(tmp_path / "clip.mp4")
    assert vision.helper_executable() == real_helper
    try:
        data = make.make_matte(clip, tmp_path / "out", backend="vision", quality=quality)
    except ProcessingError as exc:
        if (
            exc.code == "person_matte_exit_2"
            and "sysctlbyname for kern.hv_vmm_present failed" in exc.full_stderr
            and "Cannot Decode" in exc.full_stderr
        ):
            pytest.skip("the execution sandbox blocks the native macOS video decoder")
        raise
    assert data.status == "measured"
    info = ffmpeg.probe(tmp_path / "out" / "matte.mkv", count_frames=True)
    assert (info.video_frames, info.width, info.height) == (3, WIDTH, HEIGHT)
    assert data.times == pytest.approx((0, 1 / 30, 2 / 30), abs=0.000001)
    assert data.end_time == pytest.approx(3 / 30, abs=0.000001)
    assert data.coverage is not None and len(data.coverage) == 3
    assert all(0 <= fraction <= 1 for fraction in data.coverage)
    assert data.backend["mask_width"] > 0 and data.backend["mask_height"] > 0
    assert load_matte(tmp_path / "out" / "matte.json") == data


@pytest.mark.helpers
@_REAL_HELPER
def test_real_vision_helper_refuses_rotation(tmp_path: Path, real_helper: Path) -> None:
    clip = make_clip(tmp_path / "clip.mp4")
    rotated = tmp_path / "rotated.mp4"
    ffmpeg.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-display_rotation:v:0",
            "90",
            "-i",
            str(clip),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            str(rotated),
        ],
        timeout_s=30,
    )
    assert ffmpeg.probe(rotated).rotation in (90, 270)
    assert vision.helper_executable() == real_helper
    with pytest.raises(ProcessingError) as caught:
        make.make_matte(rotated, tmp_path / "out", backend="vision")
    assert caught.value.code == "person_matte_exit_3"
    assert "upright" in caught.value.message
    assert list((tmp_path / "out").iterdir()) == []
