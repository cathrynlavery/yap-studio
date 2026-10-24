"""Safe ffmpeg runner and probe tests."""

from __future__ import annotations

import json
import os
import shlex
import sys
from collections.abc import Callable, Generator
from dataclasses import fields, replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import synth
from test_install_info import build_checkout

from yap_studio import __version__, cli, ffmpeg, setup_checks
from yap_studio.cli import main
from yap_studio.errors import DependencyError, InputError, ProcessingError
from yap_studio.ffmpeg import MediaInfo, has_filters, probe, run, run_stream, tool_versions
from yap_studio.review import phone


@pytest.fixture(autouse=True)
def clear_version_cache() -> Generator[None, None, None]:
    """Keep executable revision checks independent between tests."""
    ffmpeg._version_cache.clear()
    yield
    ffmpeg._version_cache.clear()


@pytest.fixture
def doctor_phone_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phone, "_tailscale_binary", lambda: None)
    monkeypatch.setattr(phone, "tailscale_name", lambda: None)
    monkeypatch.setattr(phone, "lan_address", lambda: "192.168.1.10")


@pytest.fixture
def fake_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[str, str], Path]:
    """Select a small executable that prints a banner and records invocations."""
    def create(name: str, banner: str) -> Path:
        script = tmp_path / name
        script.write_text(
            '#!/bin/sh\nprintf "%s\\n" "$1" >> "$0.calls"\n'
            f"printf '%s\\n' {shlex.quote(banner)}\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        monkeypatch.setenv(f"YAP_STUDIO_{name.upper()}", str(script))
        return script

    return create


@pytest.mark.parametrize("release", ["4.4.2-0ubuntu0.22.04.1", "5.1.6"])
def test_find_tool_refuses_old_ffmpeg(fake_tool: Callable[[str, str], Path], release: str) -> None:
    fake_tool("ffmpeg", f"ffmpeg version {release}")
    with pytest.raises(DependencyError) as caught:
        ffmpeg.find_tool("ffmpeg")
    assert caught.value.code == "ffmpeg_too_old"
    assert release in caught.value.message


@pytest.mark.parametrize(
    "banner",
    [
        "ffmpeg version 6.1",
        "ffmpeg version 6.1.1-3ubuntu5",
        "ffmpeg version n7.0.2",
        "ffmpeg version 9.0.2",
        "ffmpeg version N-117000-g1234abcd",
        "ffmpeg version g1234abcd",
        "garbage",
        "",
    ],
)
def test_find_tool_accepts_supported_or_unknown_version(
    fake_tool: Callable[[str, str], Path], banner: str
) -> None:
    script = fake_tool("ffmpeg", banner)
    assert ffmpeg.find_tool("ffmpeg") == str(script)


def test_find_tool_refuses_old_ffprobe(fake_tool: Callable[[str, str], Path]) -> None:
    fake_tool("ffprobe", "ffprobe version 4.4.2")
    with pytest.raises(DependencyError) as caught:
        ffmpeg.find_tool("ffprobe")
    assert caught.value.code == "ffprobe_too_old"


@pytest.mark.parametrize("name", ["ffmpeg", "ffprobe"])
def test_tool_versions_rejects_wrong_banner(
    fake_tool: Callable[[str, str], Path], name: str
) -> None:
    """Do not report another executable's release as the requested tool's version."""
    other = "ffprobe" if name == "ffmpeg" else "ffmpeg"
    fake_tool(name, f"{other} version 6.1.1")
    assert tool_versions()[name] is None


@pytest.mark.parametrize("name", ["ffmpeg", "ffprobe"])
@pytest.mark.parametrize("release", ["6.1.1", "4.4.2"])
def test_find_tool_rejects_wrong_banner(
    fake_tool: Callable[[str, str], Path], name: str, release: str
) -> None:
    """Report a wrong executable before checking whether its release is supported."""
    other = "ffprobe" if name == "ffmpeg" else "ffmpeg"
    script = fake_tool(name, f"{other} version {release}")
    with pytest.raises(DependencyError) as caught:
        ffmpeg.find_tool(name)
    assert caught.value.code == f"{name}_wrong_tool"
    assert caught.value.error_type == "dependency_error"
    assert caught.value.message == f"{script} is not {name}."
    assert caught.value.suggested_action == (
        f"Point YAP_STUDIO_{name.upper()} at the {name} executable."
    )


@pytest.mark.parametrize("release", ["6.1.1", "4.4.2"])
def test_version_command_is_cached(
    fake_tool: Callable[[str, str], Path], release: str
) -> None:
    script = fake_tool("ffmpeg", f"ffmpeg version {release}")
    for _ in range(3):
        if release == "4.4.2":
            with pytest.raises(DependencyError):
                ffmpeg.find_tool("ffmpeg")
        else:
            assert ffmpeg.find_tool("ffmpeg") == str(script)
    assert tool_versions()["ffmpeg"] == release
    assert script.with_suffix(".calls").read_text() == "-version\n"


def test_version_cache_follows_realpath(
    fake_tool: Callable[[str, str], Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = fake_tool("ffmpeg", "ffmpeg version 6.1.1")
    assert ffmpeg.find_tool("ffmpeg") == str(script)
    alias = tmp_path / "alias"
    alias.symlink_to(script)
    monkeypatch.setenv("YAP_STUDIO_FFMPEG", str(alias))
    assert ffmpeg.find_tool("ffmpeg") == str(alias)
    assert script.with_suffix(".calls").read_text() == "-version\n"


def test_version_cache_rechecks_changed_executable(fake_tool: Callable[[str, str], Path]) -> None:
    script = fake_tool("ffmpeg", "ffmpeg version 6.1.1")
    assert ffmpeg.find_tool("ffmpeg") == str(script)
    previous = script.stat()
    fake_tool("ffmpeg", "ffmpeg version 4.4.2")
    os.utime(script, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1))
    with pytest.raises(DependencyError):
        ffmpeg.find_tool("ffmpeg")
    assert script.with_suffix(".calls").read_text() == "-version\n-version\n"


def test_version_cache_rechecks_changed_size_with_same_mtime(
    fake_tool: Callable[[str, str], Path],
) -> None:
    """Re-read a replacement with a different size even when its timestamp is preserved."""
    script = fake_tool("ffmpeg", "ffmpeg version 6.1.1")
    assert ffmpeg.find_tool("ffmpeg") == str(script)
    previous = script.stat()
    fake_tool("ffmpeg", "ffmpeg version 4.4.2-legacy")
    os.utime(script, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    current = script.stat()
    assert current.st_mtime_ns == previous.st_mtime_ns
    assert current.st_size != previous.st_size
    assert current.st_ino == previous.st_ino
    assert current.st_dev == previous.st_dev
    with pytest.raises(DependencyError) as caught:
        ffmpeg.find_tool("ffmpeg")
    assert caught.value.code == "ffmpeg_too_old"
    assert "4.4.2-legacy" in caught.value.message
    assert script.with_suffix(".calls").read_text() == "-version\n-version\n"


def test_find_tool_accepts_failed_version_command(fake_tool: Callable[[str, str], Path]) -> None:
    script = fake_tool("ffmpeg", "ffmpeg version 4.4.2")
    with script.open("a", encoding="utf-8") as output:
        output.write("exit 1\n")
    assert ffmpeg.find_tool("ffmpeg") == str(script)


def test_find_tool_accepts_version_timeout(
    fake_tool: Callable[[str, str], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = fake_tool("ffmpeg", "ffmpeg version 6.1.1")
    calls = []

    def timeout(command: list[str], *, timeout_s: float, tool: str) -> None:
        calls.append(command)
        assert timeout_s <= 5
        raise ProcessingError(tool, -1, "timed out", code="ffmpeg_timeout")

    monkeypatch.setattr(ffmpeg.process, "spawn", timeout)
    assert ffmpeg.find_tool("ffmpeg") == str(script)
    assert ffmpeg.find_tool("ffmpeg") == str(script)
    assert calls == [[str(script), "-version"]]


@pytest.mark.parametrize("name", ["ffmpeg", "ffprobe"])
def test_doctor_reports_old_version(
    fake_tool: Callable[[str, str], Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], doctor_phone_network: None, name: str,
) -> None:
    for tool in ("ffmpeg", "ffprobe"):
        fake_tool(tool, f"{tool} version {'4.4.2' if tool == name else '6.1.1'}")
    monkeypatch.setattr(cli, "has_filters", lambda names: dict.fromkeys(names, True))
    assert main(["doctor"]) == 2
    output = capsys.readouterr().out
    assert f"{name} 4.4.2 is too old." in output
    assert any(line.startswith(name) and "4.4.2" in line for line in output.splitlines())
    assert "not found" not in output
    assert "ok: ffmpeg is usable" not in output
    if name == "ffmpeg":
        assert "filters  not checked (ffmpeg is too old)" in output


def _consume_stream(cmd, **kwargs):
    return b"".join(run_stream(cmd, **kwargs))


@pytest.mark.parametrize("runner", [run, _consume_stream])
def test_runner_times_out(runner) -> None:
    with pytest.raises(ProcessingError) as caught:
        runner(
            [
                "ffmpeg",
                "-re",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=duration=60",
                "-map",
                "0:v:0",
                "-f",
                "null",
                "-",
            ],
            timeout_s=0.2,
        )
    assert caught.value.code == "ffmpeg_timeout"


@pytest.mark.parametrize("runner", [run, _consume_stream])
def test_runner_maps_missing_input(runner) -> None:
    with pytest.raises(InputError) as caught:
        runner(
            ["ffmpeg", "-i", "surely-not-present.mov", "-map", "0:v:0", "-f", "null", "-"],
            timeout_s=10,
        )
    assert caught.value.path == "surely-not-present.mov"
    assert caught.value.reason == "no file at this path"


@pytest.fixture
def streamed_processes(monkeypatch):
    ffmpeg.find_tool("ffmpeg")
    processes = []
    popen = ffmpeg.subprocess.Popen

    def recorded(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(ffmpeg.subprocess, "Popen", recorded)
    yield processes
    assert processes
    assert all(process.poll() is not None for process in processes)
    assert all(process.stdout.closed for process in processes)


def test_stream_matches_stdout_in_bounded_chunks(streamed_processes):
    cmd = [
        "ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
        "testsrc2=size=16x16:rate=30:duration=0.3", "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ]
    expected = run(cmd, timeout_s=10, binary=True).stdout
    chunks = list(run_stream(cmd, timeout_s=10, chunk_size=7))
    assert chunks and all(0 < len(chunk) <= 7 for chunk in chunks)
    assert b"".join(chunks) == expected
    assert all(process.returncode == 0 for process in streamed_processes)


def test_closing_stream_reaps_child(streamed_processes):
    stream = run_stream(
        ["ffmpeg", "-f", "lavfi", "-i", "testsrc2=size=16x16", "-f", "rawvideo", "-"],
        timeout_s=10,
        chunk_size=7,
    )
    assert len(next(stream)) == 7
    assert streamed_processes[0].poll() is None
    stream.close()
    assert streamed_processes[0].poll() is not None


@pytest.mark.parametrize("close_stdout", [False, True])
def test_stream_deadline_reaps_a_stalled_child(monkeypatch, streamed_processes, close_stdout):
    script = "import os, time; " + ("os.close(1); " if close_stdout else "") + "time.sleep(60)"
    monkeypatch.setattr(ffmpeg, "_safe_command", lambda _: [sys.executable, "-c", script])
    with pytest.raises(ProcessingError) as caught:
        list(run_stream(["ffmpeg"], timeout_s=0.2))
    assert caught.value.code == "ffmpeg_timeout"


def test_stream_captures_large_stderr_without_deadlock(monkeypatch, streamed_processes):
    script = "import os; os.write(2, b'x' * 100000 + b'\\xff'); raise SystemExit(1)"
    monkeypatch.setattr(ffmpeg, "_safe_command", lambda _: [sys.executable, "-c", script])
    with pytest.raises(ProcessingError) as expected:
        run(["ffmpeg"], timeout_s=10)
    with pytest.raises(ProcessingError) as streamed:
        list(run_stream(["ffmpeg"], timeout_s=10, chunk_size=7))
    assert str(streamed.value) == str(expected.value)


def test_probe_rotation_and_stream_facts(synth_media: object) -> None:
    path = synth_media.path("rotated_vertical")  # type: ignore[attr-defined]
    info = probe(path)
    assert info.has_video and info.has_audio
    assert info.video_index == 0
    assert info.audio_index == 1
    assert info.fps == Fraction(30, 1)
    assert (info.coded_width, info.coded_height) == (320, 180)
    assert (info.width, info.height) == (180, 320)
    assert info.rotation in {90, 270}
    assert info.orientation == "vertical"
    assert info.sample_rate == 48_000
    assert info.video_frames == 900
    assert info.to_dict()["fps"] == "30/1"
    assert info.to_dict()["video_index"] == 0
    assert info.to_dict()["audio_index"] == 1


@pytest.mark.parametrize(("chunks", "expected"), [
    ([b"0.06", b"6733,\n"], 0.066733),
    ([b"0.000000\n"], 0.0),
    ([b"N/A\n"], None),
    ([b"\n"], None),
    ([b"invalid\n"], None),
    ([], None),
])
def test_first_video_timestamp_stops_and_closes_stream(
    monkeypatch: pytest.MonkeyPatch, chunks: list[bytes], expected: float | None,
) -> None:
    closed = []

    def stream(command, **kwargs):
        assert command[command.index("-select_streams") + 1] == "2"
        try:
            yield from chunks
            if chunks:
                pytest.fail("read past the first decoded frame")
        finally:
            closed.append(True)

    monkeypatch.setattr(ffmpeg, "run_stream", stream)
    assert ffmpeg._first_video_frame_time(Path("source.mov"), 2) == expected
    assert closed == [True]


@pytest.mark.parametrize("error", [
    InputError("source.mov", "unreadable frame"),
    ProcessingError("ffprobe", -1, "timed out", code="ffmpeg_timeout"),
])
def test_first_video_timestamp_failure_disables_seek(
    monkeypatch: pytest.MonkeyPatch, error: Exception,
) -> None:
    def stream(command, **kwargs):
        raise error

    monkeypatch.setattr(ffmpeg, "run_stream", stream)
    assert ffmpeg._first_video_frame_time(Path("source.mov"), 0) is None


def test_probe_first_frame_is_opt_in(synth_media, monkeypatch: pytest.MonkeyPatch) -> None:
    path = synth_media.path("calib")
    calls = []

    def first_frame(source, index):
        calls.append((source, index))
        return 0.0

    monkeypatch.setattr(ffmpeg, "_first_video_frame_time", first_frame)
    assert probe(path).video_first_frame_s is None
    assert calls == []
    info = probe(path, first_frame=True)
    assert info.video_first_frame_s == info.video_start_s == 0.0
    assert info.to_dict()["video_first_frame_s"] == 0.0
    assert calls == [(Path(path), info.video_index)]


def test_probe_clock_offset(synth_media: object) -> None:
    path = synth_media.path("clean_shift_start")  # type: ignore[attr-defined]
    info = probe(path)
    assert info.audio_start_s is not None
    assert info.audio_offset_s == pytest.approx(
        info.audio_start_s - info.format_start_s, abs=0.000001
    )
    assert info.audio_offset_s > 0.05


@pytest.mark.parametrize(("average", "nominal", "expected"), [
    (Fraction(299, 10), Fraction(30), True),
    (Fraction(30), Fraction(30), False),
    (None, Fraction(30), False),
    (Fraction(30), None, False),
    (Fraction(0), Fraction(30), False),
    (Fraction(30), Fraction(0), False),
    (Fraction(-30), Fraction(30), False),
    (Fraction(30), Fraction(-30), False),
])
def test_variable_frame_rate_requires_two_positive_differing_rates(
    synth_media, average: Fraction | None, nominal: Fraction | None, expected: bool,
) -> None:
    assert ffmpeg.variable_frame_rate(average, nominal) is expected
    info = replace(probe(synth_media.path("calib")), fps=average, nominal_fps=nominal)
    assert info.variable_frame_rate is expected


def test_probe_vfr_rate_serializes_without_changing_fps(synth_media) -> None:
    path = synth_media.cached("vfr", synth.vfr_source, suffix=".mov")
    info = probe(path)
    assert info.variable_frame_rate
    assert info.nominal_fps == 30
    assert info.fps != info.nominal_fps
    facts = info.to_dict()
    assert facts["nominal_fps"] == "30/1"
    assert Fraction(facts["fps"]) == info.fps
    assert json.loads(json.dumps(facts)) == facts


def test_probe_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="no file at this path"):
        probe(tmp_path / "missing.mp4")


def test_tool_versions_and_filters() -> None:
    versions = tool_versions()
    assert versions["ffmpeg"]
    assert versions["ffprobe"]
    assert all(has_filters(["ebur128", "blackdetect", "freezedetect", "aresample"]).values())


def test_doctor(capsys: pytest.CaptureFixture[str], doctor_phone_network: None) -> None:
    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert "ok: ffmpeg is usable" in output
    assert f"install  yap-studio {__version__}" in output


def test_doctor_reports_missing_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    doctor_phone_network: None,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_FFMPEG", str(tmp_path / "missing-ffmpeg"))
    assert main(["doctor"]) == 2
    assert "problem:" in capsys.readouterr().out


@pytest.fixture
def doctor_dependencies(monkeypatch: pytest.MonkeyPatch, doctor_phone_network: None) -> None:
    """Keep install warnings independent of optional runtime dependencies."""
    monkeypatch.setattr(cli, "resolve_tool", lambda name: name)
    monkeypatch.setattr(cli, "check_version", lambda path: None)
    monkeypatch.setattr(cli, "tool_versions", lambda: {"ffmpeg": "fixture", "ffprobe": "fixture"})
    monkeypatch.setattr(cli, "has_filters", lambda names: dict.fromkeys(names, True))
    monkeypatch.setattr(cli, "available_engines", lambda: {})
    monkeypatch.setattr(cli, "resolve_backend", lambda: None)
    monkeypatch.setattr(setup_checks, "synced_service", lambda path: None)
    monkeypatch.setattr(setup_checks, "free_disk_gb", lambda path: 20.0)
    monkeypatch.setattr(setup_checks, "cutout_reason", lambda: None)


@pytest.mark.parametrize("usable", [True, False])
@pytest.mark.parametrize("source", ["wheel", "elsewhere"])
def test_doctor_warns_for_newer_checkout_without_changing_exit_code(
    doctor_dependencies: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], usable: bool, source: str,
) -> None:
    checkout = tmp_path / "checkout with spaces"
    package, _ = build_checkout(checkout, version="0.10.0")
    running = None if source == "wheel" else tmp_path / "running"
    monkeypatch.setattr("yap_studio.install_info.__version__", "0.1.0")
    monkeypatch.setattr("yap_studio.install_info.source_checkout", lambda: running)
    if not usable:
        monkeypatch.setattr(cli, "has_filters", lambda names: dict.fromkeys(names, False))
    baseline = main(["doctor"])
    capsys.readouterr()
    monkeypatch.chdir(package)
    assert main(["doctor"]) == baseline == (0 if usable else 2)
    output = capsys.readouterr().out
    assert "install  yap-studio 0.1.0" in output
    assert (
        "warning: the installed yap-studio 0.1.0 is older than this checkout (0.10.0). "
        "Run the checkout with uv run, or reinstall with: "
        f"uv tool install --force '{checkout.resolve()}[asr,mcp]'"
    ) in output


@pytest.mark.parametrize("version", ["0.1.0", "0.0.9", "unknown"])
def test_doctor_does_not_warn_for_equal_older_or_unknown_checkout(
    doctor_dependencies: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], version: str,
) -> None:
    package, _ = build_checkout(tmp_path, version=version)
    monkeypatch.chdir(package)
    monkeypatch.setattr("yap_studio.install_info.__version__", "0.1.0")
    monkeypatch.setattr("yap_studio.install_info.source_checkout", lambda: None)
    assert main(["doctor"]) == 0
    assert "warning:" not in capsys.readouterr().out


def test_doctor_does_not_warn_outside_checkout(
    doctor_dependencies: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("yap_studio.install_info.source_checkout", lambda: None)
    assert main(["doctor"]) == 0
    assert "warning:" not in capsys.readouterr().out


@pytest.mark.parametrize("version", ["0.1.1", "0.1.2", "0.1.3"])
def test_doctor_compares_running_source_version(
    doctor_dependencies: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], version: str,
) -> None:
    """Doctor compares the version of source loaded from a different checkout."""
    running_package, commit = build_checkout(tmp_path / "running", version="0.1.2")
    package, _ = build_checkout(tmp_path / "checkout", version=version)
    monkeypatch.chdir(package)
    monkeypatch.setattr("yap_studio.install_info.__version__", "0.1.0")
    monkeypatch.setattr(
        "yap_studio.install_info.__file__", str(running_package / "install_info.py")
    )
    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert f"install  yap-studio 0.1.2 (source {commit})" in output
    if version == "0.1.3":
        assert (
            "warning: the installed yap-studio 0.1.2 is older than this checkout (0.1.3)."
        ) in output
    else:
        assert "warning:" not in output


def test_doctor_does_not_warn_for_running_source_checkout(
    doctor_dependencies: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    package, _ = build_checkout(tmp_path, version="0.10.0")
    monkeypatch.chdir(package)
    monkeypatch.setattr("yap_studio.install_info.__version__", "0.1.0")
    monkeypatch.setattr("yap_studio.install_info.source_checkout", lambda: tmp_path.resolve())
    assert main(["doctor"]) == 0
    assert "warning:" not in capsys.readouterr().out


@pytest.mark.parametrize("tagged", [False, True])
def test_probe_colour_fields_and_format_tags(tmp_path: Path, tagged: bool) -> None:
    path = tmp_path / "colours.mp4"
    colour_args = (
        ["-vf", "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=limited"]
        if tagged else []
    )
    run(
        [
            "ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
            "testsrc2=size=32x32:rate=30:duration=0.1", "-c:v", "libx264", "-pix_fmt",
            "yuv420p", *colour_args, "-metadata", "title=colour fixture", str(path),
        ],
        timeout_s=10,
    )
    info = probe(path)
    assert info.pix_fmt == "yuv420p"
    assert info.color_transfer == ("bt709" if tagged else None)
    assert info.color_primaries == ("bt709" if tagged else None)
    assert info.color_space == ("bt709" if tagged else None)
    assert info.color_range == ("tv" if tagged else None)
    assert info.format_tags is not None
    assert info.format_tags["title"] == "colour fixture"
    assert info.to_dict()["color_transfer"] == info.color_transfer
    assert "format_tags" not in info.to_dict()

    @lru_cache(maxsize=2)
    def title(media: MediaInfo) -> str:
        assert media.format_tags is not None
        return media.format_tags["title"]

    assert title(info) == title(info) == "colour fixture"
    assert title.cache_info().hits == 1
    other = replace(info, format_tags={"title": "other fixture"})
    assert other != info
    assert title(other) == "other fixture"


def test_audio_only_probe_has_no_video_colour_fields(tmp_path: Path) -> None:
    path = tmp_path / "audio.wav"
    run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=duration=0.1", str(path)],
        timeout_s=10,
    )
    info = probe(path)
    assert not info.has_video
    assert info.pix_fmt is None
    assert info.color_transfer is None
    assert info.color_primaries is None
    assert info.color_space is None
    assert info.color_range is None


def test_media_info_old_positional_construction_still_works(tmp_path: Path) -> None:
    path = tmp_path / "video.mp4"
    run(
        [
            "ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
            "testsrc2=size=32x32:duration=0.1", str(path),
        ],
        timeout_s=10,
    )
    info = probe(path)
    original_fields = fields(MediaInfo)[:26]
    legacy = MediaInfo(*(getattr(info, field.name) for field in original_fields))
    assert legacy.path == info.path
    assert legacy.audio_duration_s == info.audio_duration_s
    assert legacy.pix_fmt is None
    assert legacy.color_transfer is None
    assert legacy.color_primaries is None
    assert legacy.color_space is None
    assert legacy.color_range is None
    assert legacy.format_tags is None
    assert legacy.nominal_fps is None
    assert legacy.variable_frame_rate is False


@pytest.fixture
def edit_dependencies(monkeypatch: pytest.MonkeyPatch) -> Callable[[bool], None]:
    """Make Node, the renderer and its browser, and speech present or missing."""
    from yap_studio.render import hyperframes

    monkeypatch.setattr(setup_checks, "synced_service", lambda path: None)
    monkeypatch.setattr(setup_checks, "free_disk_gb", lambda path: 20.0)
    monkeypatch.setattr(setup_checks, "cutout_reason", lambda: None)

    def apply(present: bool) -> None:
        monkeypatch.setattr(cli.tools, "find_tool", lambda name: name if present else None)
        monkeypatch.setattr(cli.tools, "run_tool", lambda *_args, **_kwargs: SimpleNamespace(
            stdout="v22.1.0\n"))
        install = object() if present else None
        monkeypatch.setattr(hyperframes, "_npm_installed", lambda: install)
        monkeypatch.setattr(hyperframes, "installed", lambda: install)
        monkeypatch.setattr(cli, "resolve_backend",
                            lambda: SimpleNamespace(
                                name="fixture", download_notice=lambda: None
                            ) if present else None)

    return apply


@pytest.mark.parametrize("edit", [False, True])
def test_doctor_edit_fails_only_when_asked_and_editing_tools_are_missing(
    doctor_dependencies: None, edit_dependencies: Callable[[bool], None],
    capsys: pytest.CaptureFixture[str], edit: bool,
) -> None:
    edit_dependencies(False)
    flag = ["--edit"] if edit else []
    assert main(["doctor", *flag]) == (2 if edit else 0)
    output = capsys.readouterr().out
    assert "edit     not ready" in output
    if edit:
        assert "problem: not ready to edit; missing node, renderer, browser, speech" in output
    else:
        assert "ok: ffmpeg is usable" in output
    assert main(["doctor", "--json", *flag]) == (2 if edit else 0)
    report = json.loads(capsys.readouterr().out)
    assert report["ready_to_edit"] is False and report["problems"] == []
    assert report["edit_missing"] == ["node", "renderer", "browser", "speech"]


def test_doctor_edit_passes_when_every_editing_tool_is_ready(
    doctor_dependencies: None, edit_dependencies: Callable[[bool], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    edit_dependencies(True)
    assert main(["doctor", "--edit"]) == 0
    output = capsys.readouterr().out
    assert "edit     ready" in output and "ok: ready to edit" in output
    assert main(["doctor", "--edit", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ready_to_edit"] is True and report["edit_missing"] == []


def test_doctor_edit_reports_ffmpeg_problems_first(
    doctor_dependencies: None, edit_dependencies: Callable[[bool], None],
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    edit_dependencies(True)
    monkeypatch.setattr(cli, "has_filters", lambda names: dict.fromkeys(names, False))
    assert main(["doctor", "--edit"]) == 2
    assert "problem: missing required filters" in capsys.readouterr().out
    assert main(["doctor", "--edit", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["ready_to_edit"] is False
