"""Allowlisted tool and shared process behavior tests."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from yap_studio import process, tools
from yap_studio.errors import DependencyError, ProcessingError
from yap_studio.transcript.backends import ParakeetMlxBackend

pytestmark = pytest.mark.usefixtures("_no_real_node")


@pytest.mark.parametrize(("platform", "brew_mode", "on_path"), [
    ("darwin", 0o755, False), ("darwin", 0o755, True),
    ("darwin", None, False), ("darwin", 0o644, False), ("linux", 0o755, False),
])
def test_add_homebrew_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    platform: str, brew_mode: int | None, on_path: bool,
) -> None:
    folder = tmp_path / "homebrew/bin"
    folder.mkdir(parents=True)
    if brew_mode is not None:
        brew = folder / "brew"
        brew.touch()
        brew.chmod(brew_mode)
    original = str(tmp_path / "preferred")
    if on_path:
        original += os.pathsep + str(folder)
    monkeypatch.setenv("PATH", original)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(tools, "HOMEBREW_BIN_DIRS", (str(folder),))
    tools.add_homebrew_path()
    expected = original
    if platform == "darwin" and brew_mode == 0o755 and not on_path:
        expected += os.pathsep + str(folder)
    assert os.environ["PATH"] == expected
    tools.add_homebrew_path()
    assert os.environ["PATH"] == expected


def test_cli_adds_homebrew_path_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from yap_studio import cli

    brew = tmp_path / "brew"
    brew.touch()
    brew.chmod(0o755)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(tools, "HOMEBREW_BIN_DIRS", (str(tmp_path),))
    monkeypatch.setenv("PATH", str(tmp_path / "preferred"))
    assert cli.main(["--version"]) == 0
    capsys.readouterr()
    assert os.environ["PATH"] == os.pathsep.join([str(tmp_path / "preferred"), str(tmp_path)])


@pytest.fixture
def fake_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    _no_real_elevenlabs: Callable[[Path], None],
) -> Callable[[str, str], Path]:
    """Put fake executables on a test-local PATH."""
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))

    def build(name: str, body: str) -> Path:
        executable = tmp_path / name
        if name == "node":
            body = ("import sys\nif sys.argv[1:] == ['--version']:\n"
                    "    print('v22.0.0')\n    sys.exit(0)\n" + body)
            npm = tmp_path / "npm"
            npm.write_text("#!/usr/bin/env node\n")
            npm.chmod(0o755)
        executable.write_text(f"#!{sys.executable}\n" + body, encoding="utf-8")
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        if name == "elevenlabs-pp-cli":
            _no_real_elevenlabs(executable)
        return executable

    return build


@pytest.mark.parametrize(("version", "returncode", "npm"), [
    ("v22.0.0", 1, True), ("v20.0.0", 0, True), ("v22.0.0", 0, False),
    ("invalid", 0, True),
])
def test_node_skips_unusable_first_candidate(
    node_bins: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    version: str, returncode: int, npm: bool,
) -> None:
    first = node_bins("first", version, returncode=returncode, npm=npm)
    working = node_bins("working")
    monkeypatch.setenv("PATH", os.pathsep.join([str(first.parent), str(working.parent)]))
    assert tools.find_tool("node") == str(working)
    assert tools.find_tool("npm") == str(working.parent / "npm")


def test_node_finds_install_place_off_path(
    node_bins: Callable[..., Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    working = node_bins("keg")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(tools, "NODE_INSTALL_PLACES", (str(working.parent),))
    assert tools.find_tool("node") == str(working)
    assert tools.find_tool("npm") == str(working.parent / "npm")


def test_node_requires_a_working_pair(
    node_bins: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken = node_bins("broken", returncode=1)
    monkeypatch.setenv("PATH", str(broken.parent))
    assert tools.find_tool("node") is None
    assert tools.find_tool("npm") is None


@pytest.mark.parametrize("later", [False, True])
@pytest.mark.parametrize("name", ["node", "npm", "hyperframes"])
def test_node_children_use_working_path(
    node_bins: Callable[..., Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    later: bool, name: str,
) -> None:
    first = node_bins("first", returncode=int(later))
    working = node_bins("working") if later else first
    original_path = os.pathsep.join([str(first.parent), str(working.parent), str(tmp_path)])
    monkeypatch.setenv("PATH", original_path)
    executable = tmp_path / "hyperframes"
    executable.write_text('#!/bin/sh\n/usr/bin/env node\nprintf "%s\\n" "$PATH"\n')
    executable.chmod(0o755)
    result = tools.run_tool(name, [], timeout_s=5)
    expected = str(working.parent) + os.pathsep + original_path if later else original_path
    assert result.stdout.splitlines() == ([str(working), expected] if name == "hyperframes"
                                          else [str(working)])
    assert tools.node_env().get("PATH", original_path) == expected
    assert os.environ["PATH"] == original_path


def test_node_cache_keys_path_and_home_and_can_reset(
    node_bins: Callable[..., Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    first, second = node_bins("first"), node_bins("second")
    monkeypatch.setenv("PATH", str(first.parent))
    assert tools.find_tool("node") == str(first)
    first.write_text(f"#!{sys.executable}\nraise SystemExit(1)\n")
    assert tools.find_tool("npm") == str(first.parent / "npm")
    monkeypatch.setenv("HOME", str(tmp_path / "other-home"))
    assert tools.find_tool("node") is None
    monkeypatch.setenv("PATH", str(second.parent))
    assert tools.find_tool("node") == str(second)
    second.write_text(f"#!{sys.executable}\nraise SystemExit(1)\n")
    tools._find_node.cache_clear()
    assert tools.find_tool("node") is None


def test_node_install_places_prefer_path_then_newest_version(
    node_bins: Callable[..., Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    path_node = node_bins("on-path")
    node_bins("opt/node@22/bin")
    newest = node_bins("opt/node@26/bin")
    node_bins("opt/node@9/bin")
    monkeypatch.setenv("PATH", str(path_node.parent))
    monkeypatch.setattr(tools, "NODE_INSTALL_PLACES", (str(tmp_path / "opt/node@*/bin"),))
    assert tools.find_tool("node") == str(path_node)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert tools.find_tool("node") == str(newest)


def test_node_finds_newest_nvm_version_under_home(
    node_bins: Callable[..., Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    node_bins("home/.nvm/versions/node/v22.9.0/bin")
    newest = node_bins("home/.nvm/versions/node/v22.10.0/bin")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(tools, "NODE_INSTALL_PLACES", ("~/.nvm/versions/node/*/bin",))
    assert tools.find_tool("node") == str(newest)


@pytest.mark.parametrize("failure", [OSError("unloadable library"),
                                     subprocess.TimeoutExpired("node", 5)])
def test_node_skips_failed_probes_and_deduplicates_resolved_files(
    node_bins: Callable[..., Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    failure: Exception,
) -> None:
    first, working = node_bins("first"), node_bins("working")
    alias = tmp_path / "alias"
    alias.symlink_to(first.parent, target_is_directory=True)
    monkeypatch.setenv("PATH", os.pathsep.join(map(str, [first.parent, alias, working.parent])))
    run = tools.subprocess.run
    probes = []

    def probe(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        probes.append(args[0])
        assert kwargs["timeout"] == 5
        if args[0] == str(first):
            raise failure
        return run(args, **kwargs)

    monkeypatch.setattr(tools.subprocess, "run", probe)
    assert tools.find_tool("node") == str(working)
    assert tools.find_tool("npm") == str(working.parent / "npm")
    assert tools.node_env()["PATH"].startswith(str(working.parent) + os.pathsep)
    assert probes == [str(first), str(working)]


def test_node_skips_nonexecutable_npm(
    node_bins: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, working = node_bins("first"), node_bins("working")
    (first.parent / "npm").chmod(0o644)
    monkeypatch.setenv("PATH", os.pathsep.join(map(str, [first.parent, working.parent])))
    assert tools.find_tool("node") == str(working)


@pytest.mark.parametrize("available", [False, True])
def test_node_version_reuses_plain_probe(
    node_bins: Callable[..., Path], monkeypatch: pytest.MonkeyPatch, available: bool,
) -> None:
    node = node_bins("version", "v22.23.1", returncode=0 if available else 1)
    monkeypatch.setenv("PATH", str(node.parent))
    run = tools.subprocess.run
    probes = []

    def probe(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        probes.append(args)
        assert kwargs.get("env") is None
        return run(args, **kwargs)

    monkeypatch.setattr(tools.subprocess, "run", probe)
    assert tools.node_version() == ("v22.23.1" if available else None)
    assert tools.find_tool("node") == (str(node) if available else None)
    assert tools.find_tool("npm") == (str(node.parent / "npm") if available else None)
    tools.node_env()
    assert tools.node_version() == ("v22.23.1" if available else None)
    assert probes == [[str(node), "--version"]]


def test_tool_returns_stdout_with_overlay_cwd_and_literal_args(
    fake_tool: Callable[[str, str], Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = fake_tool(
        "node",
        "import json, os, sys\n"
        "print(json.dumps([os.environ['YAP_TEST_INHERITED'], "
        "os.environ['YAP_TEST_OVERLAY'], os.getcwd(), sys.argv[1:]]))\n",
    )
    monkeypatch.setenv("YAP_TEST_INHERITED", "inherited")
    monkeypatch.setenv("YAP_TEST_OVERLAY", "original")
    args = ["argument with spaces", "$(exit 9)"]
    result = tools.run_tool(
        "node", args, timeout_s=10, env={"YAP_TEST_OVERLAY": "changed"}, cwd=tmp_path
    )
    assert result.returncode == 0
    assert result.args == [str(executable), *args]
    assert result.stderr == ""
    assert json.loads(result.stdout) == ["inherited", "changed", str(tmp_path.resolve()), args]
    assert os.environ["YAP_TEST_OVERLAY"] == "original"
    assert tools.find_tool("node") == str(executable)


@pytest.mark.parametrize("name", ["hyperframes", "yap-person-matte", "yap-face-boxes"])
def test_cached_executable_runs_outside_path(
    fake_tool: Callable[[str, str], Path], monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    executable = fake_tool("cached-helper", "print('cached')\n")
    monkeypatch.setattr(tools, "find_tool", lambda _: None)
    result = tools.run_tool(name, [], timeout_s=10, executable=executable)
    assert result.stdout == "cached\n"


@pytest.mark.parametrize("name", ["unlisted-tool", "npx"])
def test_unknown_tool_is_rejected_before_lookup_or_spawn(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("unsupported tool reached lookup or spawn")

    monkeypatch.setattr(tools.shutil, "which", unexpected)
    monkeypatch.setattr(process, "spawn", unexpected)
    with pytest.raises(ValueError, match="unsupported tool"):
        tools.find_tool(name)
    with pytest.raises(ValueError, match="unsupported tool"):
        tools.run_tool(name, [], timeout_s=10, executable=sys.executable)


@pytest.mark.parametrize("name", sorted(tools.ALLOWED_TOOLS - {"elevenlabs-pp-cli"}))
def test_payload_is_rejected_before_lookup_or_spawn(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("unsupported payload reached lookup or spawn")

    monkeypatch.setattr(tools, "find_tool", unexpected)
    monkeypatch.setattr(process, "spawn", unexpected)
    with pytest.raises(ValueError, match="stdin payloads"):
        tools.run_tool(name, [], timeout_s=10, input_bytes=b"")


@pytest.mark.parametrize("payload", [b'{"request": "hello"}\n', b"\x00\xff\xc3\xa9", b""])
def test_elevenlabs_payload_reaches_stdin_exactly(
    fake_tool: Callable[[str, str], Path], payload: bytes
) -> None:
    fake_tool("elevenlabs-pp-cli", "import sys\nprint(sys.stdin.buffer.read().hex())\n")
    result = tools.run_tool("elevenlabs-pp-cli", [], timeout_s=10, input_bytes=payload)
    assert result.stdout.strip() == payload.hex()
    assert isinstance(result.stderr, str)


def test_stdout_path_keeps_binary_stdout_out_of_result(
    fake_tool: Callable[[str, str], Path], tmp_path: Path,
) -> None:
    fake_tool("elevenlabs-pp-cli", "import sys\n"
              "sys.stdout.buffer.write(b'RIFF\\x00\\xff\\xc3\\xa9')\n"
              "sys.stderr.write('diagnostic')\n")
    target = tmp_path / "stdout.bin"
    result = tools.run_tool("elevenlabs-pp-cli", [], timeout_s=10, stdout_path=target)
    assert target.read_bytes() == b"RIFF\x00\xff\xc3\xa9"
    assert result.stdout is None
    assert result.stderr == "diagnostic"


def test_stdout_path_keeps_binary_stdout_out_of_processing_error(
    fake_tool: Callable[[str, str], Path], tmp_path: Path,
) -> None:
    fake_tool("elevenlabs-pp-cli", "import sys\n"
              "sys.stdout.buffer.write(b'RIFF\\x00\\xff')\n"
              "sys.stderr.write('request failed')\n"
              "sys.exit(5)\n")
    target = tmp_path / "stdout.bin"
    with pytest.raises(ProcessingError) as caught:
        tools.run_tool("elevenlabs-pp-cli", [], timeout_s=10, stdout_path=target)
    assert target.read_bytes() == b"RIFF\x00\xff"
    assert caught.value.full_stderr == "request failed"
    assert "RIFF" not in caught.value.message


def test_process_spawn_stdout_path_with_payload(tmp_path: Path) -> None:
    target = tmp_path / "stdout.bin"
    result = process.spawn(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"],
        timeout_s=10, tool="fixture", input_bytes=b"\xff\x00RIFF", stdout_path=target,
    )
    assert target.read_bytes() == b"\xff\x00RIFF"
    assert result.stdout is None
    assert result.stderr == ""


@pytest.mark.parametrize("name", sorted(tools.ALLOWED_TOOLS - {"elevenlabs-pp-cli"}))
def test_stdout_path_is_rejected_for_other_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str,
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("unsupported stdout path reached lookup or spawn")

    monkeypatch.setattr(tools, "find_tool", unexpected)
    monkeypatch.setattr(process, "spawn", unexpected)
    with pytest.raises(ValueError, match="stdout_path"):
        tools.run_tool(name, [], timeout_s=10, stdout_path=tmp_path / "audio.bin")


@pytest.mark.parametrize("name", ["node", "elevenlabs-pp-cli", "git"])
def test_call_without_payload_gets_closed_stdin(
    fake_tool: Callable[[str, str], Path], name: str
) -> None:
    fake_tool(name, "import sys\nprint(repr(sys.stdin.buffer.read()))\n")
    assert tools.run_tool(name, [], timeout_s=10).stdout == "b''\n"


def test_git_is_allowlisted() -> None:
    assert "git" in tools.ALLOWED_TOOLS
    assert tools.run_tool("git", ["--version"], timeout_s=2).stdout.startswith("git version ")


def test_tool_timeout_has_stable_code(fake_tool: Callable[[str, str], Path]) -> None:
    fake_tool("node", "import time\ntime.sleep(60)\n")
    with pytest.raises(ProcessingError) as caught:
        tools.run_tool("node", [], timeout_s=0.1)
    assert caught.value.code == "node_timeout"
    assert caught.value.returncode == -1
    assert "timed out after 0.1 s" in caught.value.full_stderr


def test_nonzero_exit_caps_stored_stderr_and_message(
    fake_tool: Callable[[str, str], Path],
) -> None:
    fake_tool(
        "node", "import sys\nsys.stderr.write('start' + 'x' * 100000 + 'finish')\nsys.exit(7)\n"
    )
    with pytest.raises(ProcessingError) as caught:
        tools.run_tool("node", [], timeout_s=10)
    error = caught.value
    assert error.code == "node_exit_7"
    assert error.returncode == 7
    assert error.full_stderr == ("x" * 100000 + "finish")[-64 * 1024:]
    assert error.message == (
        "The media tool exited with return code 7. Last error output: " + error.full_stderr[-400:]
    )


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("node", "node_not_found"),
        ("npm", "node_not_found"),
        ("hyperframes", "renderer_not_found"),
        ("xcrun", "xcode_tools_not_found"),
        ("swiftc", "xcode_tools_not_found"),
        ("elevenlabs-pp-cli", "elevenlabs_cli_not_found"),
        ("yap-person-matte", "yap_person_matte_not_found"),
        ("yap-face-boxes", "yap_face_boxes_not_found"),
        ("git", "git_not_found"),
    ],
)
def test_missing_tool_has_matching_dependency_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str, code: str
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    assert tools.find_tool(name) is None
    with pytest.raises(DependencyError) as caught:
        tools.run_tool(name, [], timeout_s=10)
    assert caught.value.code == code
    assert caught.value.suggested_action


@pytest.mark.parametrize("kind", ["missing", "directory", "not-executable"])
def test_unusable_explicit_executable_is_a_dependency_error(tmp_path: Path, kind: str) -> None:
    executable = tmp_path / "helper"
    if kind == "directory":
        executable.mkdir()
    elif kind == "not-executable":
        executable.write_text("unused", encoding="utf-8")
        executable.chmod(0o600)
    with pytest.raises(DependencyError) as caught:
        tools.run_tool("hyperframes", [], timeout_s=10, executable=executable)
    assert caught.value.code == "renderer_not_found"


def test_exec_failure_is_a_processing_error(tmp_path: Path) -> None:
    executable = tmp_path / "broken"
    executable.write_text("not an executable format", encoding="utf-8")
    executable.chmod(0o700)
    with pytest.raises(ProcessingError) as caught:
        tools.run_tool("node", [], timeout_s=10, executable=executable)
    assert caught.value.code == "node_exec_failed"
    assert caught.value.returncode == -1


def test_timeout_is_required() -> None:
    with pytest.raises(TypeError, match="timeout_s"):
        tools.run_tool("node", [])  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="timeout_s"):
        process.spawn([sys.executable], tool="python")  # type: ignore[call-arg]


@pytest.mark.parametrize("binary", [False, True])
def test_spawn_returns_nonzero_result_for_caller_to_map(binary: bool) -> None:
    result = process.spawn(
        [sys.executable, "-c", "import os; os.write(1, b'ok'); os.write(2, b'bad'); exit(4)"],
        timeout_s=10,
        tool="fixture",
        binary=binary,
    )
    assert result.returncode == 4
    assert result.stdout == (b"ok" if binary else "ok")
    assert result.stderr == (b"bad" if binary else "bad")


@pytest.mark.parametrize("binary", [False, True])
def test_spawn_payload_preserves_bytes_with_text_or_binary_output(binary: bool) -> None:
    result = process.spawn(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"],
        timeout_s=10,
        tool="fixture",
        input_bytes=b"\xff\x00hello",
        binary=binary,
    )
    assert result.stdout == (b"\xff\x00hello" if binary else "\ufffd\x00hello")


@pytest.mark.parametrize("failure", ["exit", "timeout", "exec"])
def test_speech_backend_keeps_error_mapping_on_shared_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    wav = tmp_path / "input.wav"
    wav.write_bytes(b"")
    executable = tmp_path / "engine"
    executable.write_text("not an executable format", encoding="utf-8")
    executable.chmod(0o700)

    @contextmanager
    def converted(paths: Sequence[Path]) -> Iterator[tuple[list[Path], list[Path], list[float]]]:
        yield [wav], [wav], [0.2]

    monkeypatch.setattr("yap_studio.transcript.backends._converted_wavs", converted)
    monkeypatch.setattr(ParakeetMlxBackend, "executable", staticmethod(lambda: executable))
    original_spawn = process.spawn
    calls: list[tuple[float, str]] = []

    def quick_spawn(
        command: list[str], *, timeout_s: float, tool: str
    ) -> subprocess.CompletedProcess[Any]:
        calls.append((timeout_s, tool))
        if failure == "exec":
            child = command
        else:
            script = "import sys; sys.stderr.write('engine failure'); sys.exit(4)"
            if failure == "timeout":
                script = "import time; time.sleep(60)"
            child = [sys.executable, "-c", script]
        return original_spawn(child, timeout_s=0.1 if failure == "timeout" else 10, tool=tool)

    monkeypatch.setattr(process, "spawn", quick_spawn)
    with pytest.raises(ProcessingError) as caught:
        ParakeetMlxBackend(model="fixture").transcribe_many([wav])
    assert calls == [(330, "speech_engine")]
    assert caught.value.code == "speech_engine_failed"
    assert caught.value.returncode == (4 if failure == "exit" else -1)
    if failure == "exit":
        assert caught.value.full_stderr == "engine failure"


@pytest.mark.parametrize("location", ["override", "xdg-bin", "xdg-data/../bin", ".local/bin",
                                      "homebrew/bin"])
def test_osxphotos_outside_app_path(tmp_path, monkeypatch, location, _no_real_osxphotos):
    monkeypatch.setattr(tools.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "preferred"))
    monkeypatch.setenv("UV_TOOL_BIN_DIR", str(tmp_path / "override"))
    monkeypatch.setenv("XDG_BIN_HOME", str(tmp_path / "xdg-bin"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    (tmp_path / "xdg-data").mkdir()
    monkeypatch.setattr(tools, "HOMEBREW_BIN_DIRS", (str(tmp_path / "homebrew/bin"),))
    binary = tmp_path / location / "osxphotos"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("#!/bin/sh\nprintf 'photos found'\n")
    _no_real_osxphotos(binary)
    binary.chmod(0o644)
    assert tools.find_tool("osxphotos") is None
    binary.chmod(0o755)
    assert Path(tools.find_tool("osxphotos")).resolve() == binary.resolve()
    assert tools.run_tool("osxphotos", [], timeout_s=5).stdout == "photos found"
    preferred = tmp_path / "preferred/osxphotos"
    preferred.parent.mkdir()
    preferred.write_text("#!/bin/sh\n")
    preferred.chmod(0o755)
    _no_real_osxphotos(preferred)
    assert tools.find_tool("osxphotos") == str(preferred)
