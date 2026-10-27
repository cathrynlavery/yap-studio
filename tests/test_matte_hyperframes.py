"""Real VP9 alpha streams from a fake renderer, with no browser or downloads."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from fractions import Fraction
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from test_matte_make import PASSTHROUGH, decode_matte
from test_matte_timing import timestamp_plate

from yap_studio import ffmpeg, paths, receipt
from yap_studio.errors import DependencyError, InputError, OutputPathError, ProcessingError
from yap_studio.faces.model import VideoIdentity
from yap_studio.matte import hyperframes, make, vision
from yap_studio.matte.model import Cleanup, load_matte
from yap_studio.render import hyperframes as renderer

pytestmark = pytest.mark.usefixtures("_no_real_node")

WIDTH, HEIGHT, FRAMES, FPS = 96, 128, 12, 10
VALUES = (30, 180, 60, 210, 90, 240, 120, 45, 195, 75, 225, 105)


def person_alpha(index: int) -> np.ndarray:
    """An ellipse and a body reaching the bottom, with crisp lossless alpha."""
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    center = WIDTH // 2 + index % 3 - 1
    shape = ((x - center) / 23) ** 2 + ((y - 65) / 45) ** 2 <= 1
    shape |= (np.abs(x - center) <= 18) & (y >= 65)
    return shape.astype(np.uint8) * 255


_FAKE_SCRIPT = r"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

mode = os.environ.get('FAKE_HF_MODE', '')
if mode == 'timeout-before-pid':
    time.sleep(60)

import numpy as np

arguments = sys.argv[1:]
assert arguments[0] == 'remove-background'
assert arguments[-1] == '--json'
assert sys.stdin.buffer.read() == b''
for key in ('HYPERFRAMES_NO_TELEMETRY', 'HYPERFRAMES_NO_UPDATE_CHECK',
            'HYPERFRAMES_NO_AUTO_INSTALL', 'HYPERFRAMES_SKIP_SKILLS'):
    assert os.environ[key] == '1'
Path(os.environ['FAKE_HF_PID']).write_text(str(os.getpid()))
if mode == 'exit':
    sys.stderr.write('deliberate matte failure\n')
    sys.exit(3)
if mode == 'timeout':
    time.sleep(60)
if mode == 'bad-json':
    print('{"ok": false}')
    sys.exit(0)
if mode == 'invalid-json':
    print('not JSON')
    sys.exit(0)
if mode == 'missing-output':
    print('{"ok": true}')
    sys.exit(0)
video = arguments[1]
output = arguments[arguments.index('-o') + 1]
width, height = 96, 128
raw = subprocess.run(
    [os.environ['FAKE_HF_FFMPEG'], '-nostdin', '-v', 'error', '-i', video,
     '-map', '0:v:0', '-an', '-fps_mode', 'passthrough', '-f', 'rawvideo',
     '-pix_fmt', 'rgb24', 'pipe:1'], check=True, capture_output=True,
).stdout
frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)
rgba = np.empty((len(frames), height, width, 4), dtype=np.uint8)
rgba[..., :3] = frames
y, x = np.mgrid[:height, :width]
stripes = {int(i) for i in os.environ.get('FAKE_HF_STRIPES', '').split(',') if i}
for i in range(len(frames)):
    center = width // 2 + i % 3 - 1
    shape = ((x - center) / 23) ** 2 + ((y - 65) / 45) ** 2 <= 1
    shape |= (np.abs(x - center) <= 18) & (y >= 65)
    if i in stripes:
        shape = (x >= width // 3) & (x < 2 * width // 3)
    rgba[i, ..., 3] = shape.astype(np.uint8) * 255
if mode == 'drop':
    rgba = rgba[:-1]
if mode == 'extra':
    rgba = np.concatenate((rgba, rgba[-1:]))
if mode == 'shift':
    rgba = np.concatenate((rgba[:1], rgba[:-1]))
subprocess.run(
    [os.environ['FAKE_HF_FFMPEG'], '-nostdin', '-v', 'error', '-y', '-f', 'rawvideo',
     '-pix_fmt', 'rgba', '-s', f'{width}x{height}', '-framerate', '10', '-i', 'pipe:0',
     '-an', '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p', '-lossless', '1',
     '-auto-alt-ref', '0', '-threads', '1', output],
    input=rgba.tobytes(), check=True, stdout=subprocess.DEVNULL,
)
# Deliberately do not report the actual count. Only decoded frames are evidence.
print(json.dumps({'ok': True, 'framesProcessed': 999999}))
"""


@cache
def _vp9_available() -> bool:
    encoders = ffmpeg.run(["ffmpeg", "-encoders"], timeout_s=30).stdout
    return "libvpx-vp9" in encoders


@pytest.fixture(scope="session")
def vp9_encoder() -> None:
    """Check encoder support once, skipping only the tests which need WebM."""
    if not _vp9_available():
        pytest.skip("ffmpeg has no libvpx-vp9 encoder")


class FakeRenderer:
    """An installed CLI fixture which proves the environment and stdin contract."""

    def __init__(self, folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.pid_file = folder / "hyperframes.pid"
        executable = folder / "hyperframes"
        executable.write_text(f"#!{sys.executable}\n" + _FAKE_SCRIPT, encoding="utf-8")
        executable.chmod(0o755)
        self.install = renderer.RendererInstall("test-version", folder, executable, None, "22")
        monkeypatch.setattr(renderer, "installed", lambda: self.install)
        monkeypatch.setenv("HOME", str(folder / "home"))
        monkeypatch.setenv("FAKE_HF_PID", str(self.pid_file))
        monkeypatch.setenv("FAKE_HF_FFMPEG", ffmpeg.find_tool("ffmpeg"))
        self.child: subprocess.Popen[bytes] | None = None
        popen = subprocess.Popen

        def start(command: list[str], *args: object, **kwargs: object):
            child = popen(command, *args, **kwargs)
            if command[0] == str(executable):
                self.child = child
            return child

        monkeypatch.setattr(subprocess, "Popen", start)

    def assert_reaped(self) -> None:
        # Startup can time out before the child writes its PID file.
        assert self.child is not None, "renderer was never launched"
        assert self.child.returncode is not None
        with pytest.raises(ProcessLookupError):
            os.kill(self.child.pid, 0)
        with pytest.raises(ChildProcessError):
            os.waitpid(self.child.pid, os.WNOHANG)


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeRenderer:
    return FakeRenderer(tmp_path, monkeypatch)


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    video = tmp_path / "clip.mkv"
    frames = np.empty((FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8)
    # Distinct temporal luma and a spatial pattern make a one-frame shift clear.
    x = np.arange(WIDTH) % 8
    for i, value in enumerate(VALUES):
        frames[i] = (value + x[None, :, None]).astype(np.uint8)
    subprocess.run(
        ffmpeg._safe_command(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{WIDTH}x{HEIGHT}",
                "-framerate",
                str(FPS),
                "-i",
                "pipe:0",
                "-c:v",
                "ffv1",
                str(video),
            ]
        ),
        input=frames.tobytes(),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return video


@pytest.mark.usefixtures("vp9_encoder")
def test_happy_path_alpha_only_identity_and_alignment(
    clip: Path, tmp_path: Path, fake: FakeRenderer
) -> None:
    out = tmp_path / "out"
    data = make.make_matte(clip, out, backend="hyperframes", cleanup=PASSTHROUGH)
    assert data.backend == {
        "name": "hyperframes-remove-background",
        "version": "test-version",
        "alignment": 1.0,
    }
    assert data.video.sha256 == receipt.sha256_file(clip)
    assert data.times == pytest.approx(tuple(i / FPS for i in range(FRAMES)))
    assert data.end_time == FRAMES / FPS
    assert data.coverage == tuple(float((person_alpha(i) >= 128).mean()) for i in range(FRAMES))
    assert data.empty_frames == () and data.warnings == ()
    assert decode_matte(out / "matte.mkv") == b"".join(
        person_alpha(i).tobytes() for i in range(FRAMES)
    )
    info = ffmpeg.probe(out / "matte.mkv")
    assert (info.video_frames, info.width, info.height) == (FRAMES, WIDTH, HEIGHT)
    assert (info.video_codec, info.pix_fmt) == ("ffv1", "gray")
    assert load_matte(out / "matte.json") == data
    assert sorted(p.name for p in out.iterdir()) == ["matte.json", "matte.mkv"]
    fake.assert_reaped()


def test_variable_rate_refused_before_cli(
    clip: Path, tmp_path: Path, fake: FakeRenderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = ffmpeg.probe
    monkeypatch.setattr(ffmpeg, "probe", lambda *a, **kw: replace(probe(*a, **kw), nominal_fps=12))
    with pytest.raises(InputError) as caught:
        make.make_matte(clip, tmp_path / "out", backend="hyperframes")
    assert caught.value.code == "matte_variable_rate"
    assert caught.value.suggested_action == "run it on the cut, which is constant rate"
    assert not fake.pid_file.exists()
    assert not (tmp_path / "out").exists()


@pytest.mark.usefixtures("vp9_encoder")
def test_irregular_timestamps_refused_before_renderer(
    tmp_path: Path, fake: FakeRenderer
) -> None:
    plate = timestamp_plate(tmp_path / "irregular.mkv", width=WIDTH, height=HEIGHT)
    info = ffmpeg.probe(plate)
    assert info.fps == info.nominal_fps == Fraction(30)
    with pytest.raises(InputError) as caught:
        make.make_matte(plate, tmp_path / "out", backend="hyperframes", notice=None)
    assert caught.value.code == "matte_variable_rate"
    assert not fake.pid_file.exists()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("count", [1, 2, 3, 7])
def test_python_temporal_median_matches_ffmpeg_clone_padding(count: int) -> None:
    frames = np.random.default_rng(7).integers(0, 256, (count, 4, 6), dtype=np.uint8)
    command = ffmpeg._safe_command([
        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", "6x4",
        "-framerate", "30", "-i", "pipe:0", "-vf",
        "tpad=start=1:stop=1:start_mode=clone:stop_mode=clone,tmedian=radius=1",
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ])
    expected = subprocess.run(command, input=frames.tobytes(), capture_output=True,
                              check=True).stdout
    actual = b"".join(hyperframes.temporal_median(iter(frame.tobytes() for frame in frames)))
    assert actual == expected


@pytest.mark.usefixtures("vp9_encoder")
def test_default_cleanup_keeps_guarded_middle_frame_empty(
    clip: Path, tmp_path: Path, fake: FakeRenderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    short = tmp_path / "short.mkv"
    ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(clip), "-frames:v", "3",
                "-c:v", "copy", str(short)], timeout_s=30)
    monkeypatch.setenv("FAKE_HF_STRIPES", "1")
    data = make.make_matte(short, tmp_path / "out", backend="hyperframes", notice=None)
    assert data.empty_frames == (1,)
    raw = [person_alpha(0).tobytes(), bytes(WIDTH * HEIGHT), person_alpha(2).tobytes()]
    expected = tmp_path / "old-cleanup.mkv"
    subprocess.run(make._encoder_command(expected, data.video, Cleanup()),
                   input=b"".join(raw), check=True, capture_output=True)
    old = np.frombuffer(decode_matte(expected), dtype=np.uint8).reshape(3, HEIGHT, WIDTH)
    actual = np.frombuffer(decode_matte(tmp_path / "out/matte.mkv"),
                           dtype=np.uint8).reshape(3, HEIGHT, WIDTH)
    assert old[1].any()
    assert not actual[1].any()
    np.testing.assert_array_equal(actual[[0, 2]], old[[0, 2]])
    fake.assert_reaped()


@pytest.mark.usefixtures("vp9_encoder")
@pytest.mark.parametrize(
    "mode,code",
    [
        ("drop", "matte_frame_count"),
        ("extra", "matte_frame_count"),
        ("shift", "matte_alignment"),
    ],
)
def test_bad_count_or_alignment_rolls_back(
    clip: Path,
    tmp_path: Path,
    fake: FakeRenderer,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    code: str,
) -> None:
    monkeypatch.setenv("FAKE_HF_MODE", mode)
    with pytest.raises(ProcessingError) as caught:
        make.make_matte(clip, tmp_path / "out", backend="hyperframes", cleanup=PASSTHROUGH)
    assert caught.value.code == code
    if mode == "shift":
        assert "alignment share" in caught.value.message and "99%" in caught.value.message
    assert list((tmp_path / "out").iterdir()) == []
    fake.assert_reaped()


@pytest.mark.usefixtures("vp9_encoder")
def test_stripes_are_zeroed_with_raw_coverage_and_warning(
    clip: Path, tmp_path: Path, fake: FakeRenderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_HF_STRIPES", "2,5,9")
    out = tmp_path / "out"
    data = make.make_matte(clip, out, backend="hyperframes", cleanup=PASSTHROUGH)
    assert data.empty_frames == (2, 5, 9)
    assert len(data.warnings) == 1
    assert "3 frame(s)" in data.warnings[0]
    assert all(t in data.warnings[0] for t in ("0.200 s", "0.500 s", "0.900 s"))
    assert data.backend["alignment"] == 1.0
    assert data.coverage is not None
    assert all(data.coverage[i] == 1 / 3 for i in data.empty_frames)
    assert decode_matte(out / "matte.mkv") == b"".join(
        bytes(WIDTH * HEIGHT) if i in data.empty_frames else person_alpha(i).tobytes()
        for i in range(FRAMES)
    )
    assert load_matte(out / "matte.json") == data
    fake.assert_reaped()


@pytest.mark.parametrize(
    "mode,code",
    [
        ("exit", "hyperframes_matte_exit_3"),
        ("bad-json", "hyperframes_matte_output"),
        ("invalid-json", "hyperframes_matte_output"),
        ("missing-output", "hyperframes_matte_output"),
        ("timeout", "hyperframes_matte_timeout"),
        ("timeout-before-pid", "hyperframes_matte_timeout"),
    ],
)
def test_cli_failures_are_reaped_and_leave_no_outputs(
    clip: Path,
    tmp_path: Path,
    fake: FakeRenderer,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    code: str,
) -> None:
    monkeypatch.setenv("FAKE_HF_MODE", mode)
    if code == "hyperframes_matte_timeout":
        monkeypatch.setattr(hyperframes, "_deadline_seconds", lambda frames: 0.5)
    started = time.monotonic()
    with pytest.raises(ProcessingError) as caught:
        make.make_matte(clip, tmp_path / "out", backend="hyperframes")
    assert caught.value.code == code
    assert time.monotonic() - started < 5
    assert list((tmp_path / "out").iterdir()) == []
    fake.assert_reaped()
    if code == "hyperframes_matte_timeout":
        assert fake.child is not None and fake.child.returncode == -signal.SIGKILL
    if mode == "timeout-before-pid":
        assert not fake.pid_file.exists()


@pytest.mark.parametrize(
    "top,bottom,coverage", [(True, True, False), (False, False, False), (False, True, True)]
)
def test_no_person_guard_rules(top: bool, bottom: bool, coverage: bool) -> None:
    alpha = person_alpha(0)
    if top:
        alpha[:3, : WIDTH // 3] = 255
    if not bottom:
        alpha[-2:] = 0
    if coverage:
        alpha[3:] = 255
    assert hyperframes._no_person(alpha)
    assert not hyperframes._no_person(person_alpha(0))


def test_guard_thresholds_and_soft_alpha() -> None:
    alpha = person_alpha(0)
    alpha[:3, : WIDTH // 4] = 128
    assert not hyperframes._no_person(alpha)  # Exactly 25% is permitted.
    alpha[:3, WIDTH // 4] = 127
    assert not hyperframes._no_person(alpha)
    alpha[:3, WIDTH // 4] = 128
    assert hyperframes._no_person(alpha)


@pytest.mark.parametrize("wrong,expected", [(1, 0.99), (2, 0.98), (0, 1.0)])
def test_alignment_requires_at_least_99_percent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wrong: int, expected: float
) -> None:
    count = 102
    plate = tmp_path / "plate"
    webm = tmp_path / "webm"
    identity = VideoIdentity("plate", "a" * 64, count, Fraction(FPS), WIDTH, HEIGHT)
    install = renderer.RendererInstall("test", tmp_path, tmp_path / "fake", None, "22")
    stream = hyperframes.MatteStream(install, time.monotonic() + 10, iter(()))

    def frames(video: Path, *args: object, alpha: bool = False, **kwargs: object):
        for i in range(count):
            if alpha:
                yield person_alpha(0)
            else:
                offset = 1 if video == webm and 1 <= i <= wrong else 0
                yield np.full((HEIGHT, WIDTH), 50 + 100 * ((i - offset) % 2), dtype=np.uint8)

    monkeypatch.setattr(hyperframes, "_frames", frames)
    if expected < 0.99:
        with pytest.raises(ProcessingError) as caught:
            for _ in hyperframes._guarded_frames(plate, webm, identity, stream):
                pass
        assert caught.value.code == "matte_alignment"
        assert "98.00%" in caught.value.message
    else:
        assert sum(1 for _ in hyperframes._guarded_frames(plate, webm, identity, stream)) == count
    assert stream.compared == 100 and stream.alignment == expected


@pytest.mark.parametrize("difference", [0.0, 0.24, 0.26])
def test_alignment_accepts_ties_and_quarter_level_tolerance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, difference: float
) -> None:
    plate = tmp_path / "plate"
    identity = VideoIdentity("plate", "a" * 64, 3, Fraction(FPS), WIDTH, HEIGHT)
    install = renderer.RendererInstall("test", tmp_path, tmp_path / "fake", None, "22")
    stream = hyperframes.MatteStream(install, time.monotonic() + 10, iter(()))
    alpha = person_alpha(0)
    # Set a fraction of the sampled solid pixels one level brighter than the
    # previous frame. The mean difference straddles the 0.25 level allowance.
    rows, columns = np.nonzero(hyperframes._solid_samples(alpha))
    current = np.zeros_like(alpha)
    ones = round(len(rows) * difference)
    current[rows[:ones] * 4, columns[:ones] * 4] = 1

    def frames(video: Path, *args: object, alpha: bool = False, **kwargs: object):
        if alpha:
            yield from (person_alpha(0) for _ in range(3))
        elif video == plate:
            yield np.zeros_like(current)
            yield current
            yield np.full_like(current, 100)
        else:
            yield from (np.zeros_like(current) for _ in range(3))

    monkeypatch.setattr(hyperframes, "_frames", frames)
    if difference > 0.25:
        with pytest.raises(ProcessingError, match="alignment share"):
            for _ in hyperframes._guarded_frames(plate, tmp_path / "webm", identity, stream):
                pass
    else:
        assert (
            sum(1 for _ in hyperframes._guarded_frames(plate, tmp_path / "webm", identity, stream))
            == 3
        )
        assert stream.alignment == 1.0


@pytest.mark.usefixtures("vp9_encoder")
@pytest.mark.parametrize("frames", [1, 2])
def test_short_clip_needs_no_alignment_neighbours(
    clip: Path, tmp_path: Path, fake: FakeRenderer, frames: int
) -> None:
    short = tmp_path / "short.mkv"
    ffmpeg.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(clip),
            "-frames:v",
            str(frames),
            "-c:v",
            "copy",
            str(short),
        ],
        timeout_s=30,
    )
    data = make.make_matte(short, tmp_path / "out", backend="hyperframes", cleanup=PASSTHROUGH)
    assert data.backend["alignment"] == 1.0
    assert decode_matte(tmp_path / "out/matte.mkv") == b"".join(
        person_alpha(i).tobytes() for i in range(frames)
    )


def test_unavailable_reason_uses_only_existing_install(
    fake: FakeRenderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert hyperframes.unavailable_reason() is None
    monkeypatch.setattr(renderer, "installed", lambda: None)
    assert hyperframes.unavailable_reason() == (
        "the HyperFrames matte needs the renderer; run yap-studio doctor"
    )


@pytest.mark.usefixtures("vp9_encoder")
def test_auto_falls_back_to_hyperframes(
    clip: Path, tmp_path: Path, fake: FakeRenderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vision, "unavailable_reason", lambda: "no Vision")
    data = make.make_matte(clip, tmp_path / "out", cleanup=PASSTHROUGH)
    assert data.backend["name"] == "hyperframes-remove-background"
    fake.assert_reaped()


@pytest.mark.parametrize("backend", ["vision", "hyperframes"])
def test_explicit_missing_backend_never_falls_back(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    monkeypatch.setattr(vision, "unavailable_reason", lambda: "no Vision")
    monkeypatch.setattr(renderer, "installed", lambda: None)
    with pytest.raises(DependencyError) as caught:
        make.make_matte(clip, tmp_path / "out", backend=backend)
    assert ("no Vision" if backend == "vision" else "run yap-studio doctor") in caught.value.message
    assert not (tmp_path / "out").exists()


@pytest.mark.usefixtures("vp9_encoder")
def test_shared_cleanup_matches_cleaning_guarded_alpha(
    clip: Path, tmp_path: Path, fake: FakeRenderer
) -> None:
    cleanup = Cleanup(median=True, low=20, high=230, feather_sigma=0.6)
    make.make_matte(clip, tmp_path / "out", backend="hyperframes", cleanup=cleanup)
    expected = tmp_path / "expected.mkv"
    subprocess.run(
        make._encoder_command(expected, load_matte(tmp_path / "out/matte.json").video, cleanup),
        input=b"".join(person_alpha(i).tobytes() for i in range(FRAMES)),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert decode_matte(tmp_path / "out/matte.mkv") == decode_matte(expected)


@pytest.mark.usefixtures("vp9_encoder")
def test_document_failure_rolls_back_hyperframes_matte(
    clip: Path, tmp_path: Path, fake: FakeRenderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise OutputPathError("deliberate publication failure")

    monkeypatch.setattr(paths, "atomic_write_text", fail)
    with pytest.raises(OutputPathError, match="deliberate"):
        make.make_matte(clip, tmp_path / "out", backend="hyperframes")
    assert list((tmp_path / "out").iterdir()) == []
