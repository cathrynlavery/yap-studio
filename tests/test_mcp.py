"""MCP operation exposure and stdio protocol tests."""

from __future__ import annotations

import asyncio
import builtins
import inspect
import json
import queue
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp.shared.memory import create_connected_server_and_client_session

from yap_studio.checks.structure import check_structure
from yap_studio.cli import main
from yap_studio.mcp_server import build_server
from yap_studio.registry import REGISTRY
from yap_studio.verify import verify

ROOT = Path(__file__).parents[1]


def _edit_list(path: Path, source: Path, spans) -> Path:
    path.write_text(
        json.dumps(
            {
                "source": str(source),
                "segments": [{"start": start, "end": end} for start, end in spans],
            }
        )
    )
    return path


def _quiet_clean(synth_media) -> tuple[Path, float]:
    def build(output: Path) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(synth_media.path("clean")),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-c:v",
                "copy",
                "-af",
                "volume=0.5",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(output),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=120,
            check=True,
        )

    path = synth_media.cached("clean_quiet", build)
    integrated = check_structure(path).integrated_lufs
    assert integrated is not None
    return path, round(integrated, 1)


async def _list_tools():
    async with create_connected_server_and_client_session(build_server()) as client:
        return await client.list_tools()


async def _call_tool(arguments: dict):
    async with create_connected_server_and_client_session(build_server()) as client:
        return await client.call_tool("verify", arguments)


def test_mcp_schema_registry_and_signature_do_not_drift() -> None:
    listed = asyncio.run(_list_tools())
    assert [tool.name for tool in listed.tools] == ["verify"]
    mcp_names = set(listed.tools[0].inputSchema["properties"])
    registry_names = {param.name for param in REGISTRY["verify"].params()}
    signature_names = set(inspect.signature(verify).parameters) - REGISTRY["verify"].excluded_params
    assert mcp_names == registry_names == signature_names
    loudness = listed.tools[0].inputSchema["properties"]["loudness"]
    assert loudness["type"] == "string"
    assert loudness["enum"] == ["check", "note"]
    assert loudness["default"] == "check"
    assert "intermediate render" in loudness["description"]


@pytest.mark.parametrize("loudness", ["check", "note"])
def test_mcp_verify_returns_passing_structured_result(
    synth_media, tmp_path: Path, loudness: str,
) -> None:
    media, loudness_target = _quiet_clean(synth_media)
    edl = _edit_list(tmp_path / "edit.json", synth_media.path("calib"), synth_media.spans)
    result = asyncio.run(
        _call_tool(
            {
                "media": str(media),
                "edl": str(edl),
                "asr": "none",
                "loudness_target": loudness_target if loudness == "check" else -14.0,
                "loudness": loudness,
            }
        )
    )
    assert result.structuredContent["success"] is True
    assert result.structuredContent["status"] == "pass"
    if loudness == "note":
        notes = [
            row for row in result.structuredContent["findings"]
            if row["check"] in {"loudness.integrated", "loudness.true_peak"}
        ]
        assert len(notes) == 2
        assert all(row["severity"] == "info" and row["suggested_action"] is None for row in notes)


def test_mcp_missing_media_returns_structured_input_error(tmp_path: Path) -> None:
    result = asyncio.run(
        _call_tool({"media": str(tmp_path / "missing.mp4"), "asr": "none"})
    )
    content = result.structuredContent
    assert content["success"] is False
    assert content["error"]["code"] == "invalid_input"
    assert content["error"]["suggested_action"]


def test_mcp_unexpected_exception_returns_internal_error(monkeypatch: pytest.MonkeyPatch) -> None:
    original = REGISTRY["verify"]

    def explode(media: str) -> dict:
        """Always fail unexpectedly.

        Args:
            media: Input path.
        """
        raise RuntimeError("boom")

    monkeypatch.setitem(REGISTRY, "verify", replace(original, fn=explode))
    result = asyncio.run(_call_tool({"media": "anything.mp4"}))
    assert result.structuredContent == {
        "success": False,
        "error": {
            "type": "internal_error",
            "code": "internal_error",
            "message": "Unexpected error; see the server log.",
        },
    }


def test_cli_mcp_without_extra_prints_install_hint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_import = builtins.__import__

    def missing_mcp(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mcp.server.fastmcp":
            raise ModuleNotFoundError("No module named 'mcp'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_mcp)
    assert main(["mcp"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: MCP support is not installed." in captured.err
    assert "fix: pip install 'yap-studio[mcp]'" in captured.err


@pytest.mark.e2e
def test_mcp_stdio_emits_only_json_rpc_on_stdout() -> None:
    initialize_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "1"},
        },
    }
    initialized_notification = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    }
    tools_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }
    process = subprocess.Popen(
        [str(ROOT / ".venv" / "bin" / "yap-studio"), "mcp"],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    stdout_lines: queue.Queue[str | None] = queue.Queue()
    stderr_lines: list[str] = []

    def read_stdout() -> None:
        for line in process.stdout:
            stdout_lines.put(line)
        stdout_lines.put(None)

    def read_stderr() -> None:
        stderr_lines.extend(process.stderr)

    stdout_thread = threading.Thread(target=read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    deadline = time.monotonic() + 60

    def next_stdout_line() -> str | None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pytest.fail(f"timed out waiting for MCP server; stderr: {''.join(stderr_lines)}")
        try:
            return stdout_lines.get(timeout=remaining)
        except queue.Empty:
            pytest.fail(f"timed out waiting for MCP server; stderr: {''.join(stderr_lines)}")

    def parse_reply(line: str) -> dict:
        reply = json.loads(line)
        assert reply.get("jsonrpc") == "2.0"
        return reply

    def read_until_reply(reply_id: int) -> dict:
        while True:
            line = next_stdout_line()
            assert line is not None, (
                f"MCP server exited before replying to id {reply_id}; "
                f"stderr: {''.join(stderr_lines)}"
            )
            reply = parse_reply(line)
            if reply.get("id") == reply_id:
                return reply

    def send(message: dict) -> None:
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()

    try:
        send(initialize_request)
        read_until_reply(1)

        send(initialized_notification)
        send(tools_request)
        tools_reply = read_until_reply(2)
        assert [tool["name"] for tool in tools_reply["result"]["tools"]] == [
            "verify"
        ]

        process.stdin.close()
        remaining = deadline - time.monotonic()
        assert remaining > 0, "timed out waiting for MCP server to exit"
        assert process.wait(timeout=remaining) == 0

        while (line := next_stdout_line()) is not None:
            parse_reply(line)
    finally:
        if not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
